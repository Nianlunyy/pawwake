"""
数据库模块 —— 负责所有跟 PostgreSQL 打交道的事情
==============================================
包括：
- 创建表结构
- 存储对话记录
- 存储/检索记忆（带中文分词和加权排序）
"""

import os
import re
import json
import logging
import uuid
from typing import Optional, List
from datetime import datetime, date, timedelta, timezone as dt_timezone

import asyncpg

import shared

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

HAS_PGVECTOR = False  # 在init_tables时检测


# ============================================================
# 连接池管理
# ============================================================

_pool: Optional[asyncpg.Pool] = None


class BrokenMergeReferencesError(ValueError):
    """备份前发现 merged_from 引用了不存在的记忆。"""

    def __init__(self, count: int):
        self.count = count
        super().__init__(
            f"检测到 {count} 条记忆的合并来源已失效，可修复断裂引用后重新导出"
        )


class BrokenSupersessionReferencesError(ValueError):
    """备份前发现 superseded_by 指向不存在的后继记忆。"""

    def __init__(self, count: int):
        self.count = count
        super().__init__(f"检测到 {count} 条记忆的版本后继已失效，无法导出完整备份")


class DatabaseDisabled(RuntimeError):
    """数据库总闸关闭时拒绝创建或返回连接池。"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if not shared.DATABASE_ENABLED:
        raise DatabaseDisabled("DATABASE_ENABLED=false，数据库连接已停用")
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未设置！")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, statement_cache_size=0)
        print("✅ 数据库连接池已创建")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ 数据库连接池已关闭")


# ============================================================
# 表结构初始化
# ============================================================

async def init_tables():
    global HAS_PGVECTOR
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT,
                model           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                metadata        TEXT,
                deleted_at      TIMESTAMPTZ DEFAULT NULL,
                deletion_scope  TEXT DEFAULT NULL
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                importance      INTEGER DEFAULT 5,
                source_session  TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                last_accessed   TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_fts
            ON memories
            USING gin(to_tsvector('simple', content));
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations (session_id, created_at);
        """)

        # 工具调用支持：加 metadata 字段（已有表自动迁移）
        await conn.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS metadata TEXT;
        """)

        # 对话回收站：整段与批量删除只标记 deleted_at，老库自动补列。
        await conn.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
        """)
        await conn.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS deletion_scope TEXT;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_deleted_at
            ON conversations (deleted_at)
            WHERE deleted_at IS NOT NULL;
        """)

        # content 允许 NULL（工具调用时 assistant 的 content 可能为空）
        await conn.execute("""
            ALTER TABLE conversations ALTER COLUMN content DROP NOT NULL;
        """)

        # 原始对话召回索引。NULL 同时作为可恢复 backfill 的持久账本。
        await conn.execute("""
            ALTER TABLE conversations ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_content_tsv
            ON conversations USING GIN (content_tsv);
        """)

        # 网关配置表（存储运行时可变配置）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS gateway_config (
                key     TEXT PRIMARY KEY,
                value   TEXT DEFAULT ''
            );
        """)
        await conn.execute(
            """INSERT INTO gateway_config (key, value)
               VALUES ('database_instance_id', $1)
               ON CONFLICT (key) DO NOTHING""",
            str(uuid.uuid4()),
        )

        # 分区缓存状态表（存储每个session的轮转状态）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS session_cache_state (
                session_id      TEXT PRIMARY KEY,
                summary         TEXT DEFAULT '',
                a_start_round   INTEGER DEFAULT 0,
                deleted_summary TEXT,
                deleted_a_start_round INTEGER,
                deleted_cache_valid BOOLEAN,
                seen_fragment_ids TEXT[] DEFAULT '{}',
                seen_fragment_times JSONB DEFAULT '{}'::jsonb,
                seen_memory_times JSONB DEFAULT '{}'::jsonb,
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS seen_fragment_ids TEXT[] DEFAULT '{}';
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS seen_fragment_times JSONB DEFAULT '{}'::jsonb;
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS seen_memory_times JSONB DEFAULT '{}'::jsonb;
        """)
        # deleted_cache_valid: NULL=没有寄存，TRUE=可原样恢复，FALSE=旧段已变更需重算。
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS deleted_summary TEXT;
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS deleted_a_start_round INTEGER;
        """)
        await conn.execute("""
            ALTER TABLE session_cache_state
            ADD COLUMN IF NOT EXISTS deleted_cache_valid BOOLEAN;
        """)
        await conn.execute("""
            UPDATE session_cache_state AS scs
            SET seen_fragment_times = (
                SELECT COALESCE(
                    jsonb_object_agg(fragment_id, to_jsonb(scs.updated_at)),
                    '{}'::jsonb
                )
                FROM unnest(COALESCE(scs.seen_fragment_ids, '{}'::text[])) AS fragment_id
            )
            WHERE COALESCE(scs.seen_fragment_times, '{}'::jsonb) = '{}'::jsonb
              AND cardinality(COALESCE(scs.seen_fragment_ids, '{}'::text[])) > 0;
        """)

        # ---- 三层记忆架构字段（layer / title / is_active / merged_from / event_date）----
        # layer: 1=原始碎片, 2=事件记忆, 3=核心记忆
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS layer INTEGER DEFAULT 1;
        """)

        # title: 记忆标题（语义锚点，用于搜索加权）
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS title TEXT DEFAULT NULL;
        """)

        # is_active: 是否参与搜索（碎片合并后变为 false）
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;
        """)

        # merged_from: 合并来源的碎片ID列表
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS merged_from INTEGER[] DEFAULT NULL;
        """)

        # event_date: 事件日期（用于按天整理）
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS event_date DATE DEFAULT NULL;
        """)

        # 自动提取碎片的逐条对话来源；老记忆与非提取写入保持来源未知。
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_message_ids INTEGER[] DEFAULT NULL;
        """)
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_content_intact BOOLEAN DEFAULT FALSE;
        """)

        # remind_at: 对话里明确的未来约定，到期后强制注入一次；claimed/delivered 记录处理时间与送达时间
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS remind_at TIMESTAMPTZ DEFAULT NULL;
        """)
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS reminder_claimed_at TIMESTAMPTZ DEFAULT NULL;
        """)
        await conn.execute("""
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS reminder_delivered_at TIMESTAMPTZ DEFAULT NULL;
        """)

        # external_id: 调用方提供的稳定幂等键
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS external_id TEXT;
        """)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_external_id
            ON memories (external_id)
            WHERE external_id IS NOT NULL;
        """)

        # superseded_by: 自动冲突接管后的后继记忆
        await conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS superseded_by INTEGER;
        """)
        await conn.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_memories_superseded_by'
                      AND conrelid = 'memories'::regclass
                ) THEN
                    ALTER TABLE memories
                    ADD CONSTRAINT fk_memories_superseded_by
                    FOREIGN KEY (superseded_by) REFERENCES memories(id);
                END IF;
            END $$;
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_superseded_by
            ON memories (superseded_by)
            WHERE superseded_by IS NOT NULL;
        """)

        # 三层记忆索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories (layer);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_active ON memories (is_active);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_event_date ON memories (event_date);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_remind_due
            ON memories (remind_at)
            WHERE remind_at IS NOT NULL AND reminder_delivered_at IS NULL;
        """)

        # 尝试启用pgvector扩展（向量搜索）
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            HAS_PGVECTOR = True
            print("✅ pgvector扩展已启用")

            # 对话表向量列
            await conn.execute(f"""
                ALTER TABLE conversations
                ADD COLUMN IF NOT EXISTS embedding vector({shared.EMBEDDING_DIM});
            """)

            # 记忆表向量列
            await conn.execute(f"""
                ALTER TABLE memories
                ADD COLUMN IF NOT EXISTS embedding vector({shared.EMBEDDING_DIM});
            """)
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding
                    ON memories USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10);
                """)
            except Exception:
                pass  # ivfflat需要一定行数才能建索引，初期跳过
        except Exception as e:
            HAS_PGVECTOR = False
            print(f"⚠️ pgvector不可用（{e}），向量搜索将使用Python端计算")

            # 回退：用TEXT列存JSON格式的向量
            await conn.execute("""
                ALTER TABLE conversations ADD COLUMN IF NOT EXISTS embedding_json TEXT;
            """)
            await conn.execute("""
                ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_json TEXT;
            """)

    print("✅ 数据库表结构已就绪")


async def _vector_dimensions(conn) -> dict:
    return {
        table: await conn.fetchval(
            """SELECT atttypmod FROM pg_attribute
               WHERE attrelid = to_regclass($1) AND attname = 'embedding'
                 AND NOT attisdropped""",
            table,
        )
        for table in ("memories", "conversations")
    }


async def get_vector_dimensions() -> dict:
    if not HAS_PGVECTOR:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await _vector_dimensions(conn)


async def get_embedding_vector_presence() -> dict:
    """Check both tables, including inactive and deleted rows, before a clear."""
    column = "embedding" if HAS_PGVECTOR else "embedding_json"
    pool = await get_pool()
    async with pool.acquire() as conn:
        return {
            table: await conn.fetchval(
                f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)"
            )
            for table in ("memories", "conversations")
        }


async def save_embedding_settings(base_url: str, model: str, dim: int,
                                  confirmed: bool, *, allow_empty: bool = False,
                                  retain_vectors: bool = False) -> dict:
    """Save settings and update vector provenance atomically on approval."""
    if dim <= 0:
        raise ValueError("Embedding 维度必须大于 0")
    incomplete = not base_url.strip() or not model.strip()
    if incomplete and not allow_empty:
        raise ValueError("Embedding 地址和模型不能为空")
    target = [base_url.strip().rstrip("/"), model.strip(), dim]
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            dimensions = await _vector_dimensions(conn) if HAS_PGVECTOR else {}
            rebuilt = [table for table, current in dimensions.items() if current != dim]
            column = "embedding" if HAS_PGVECTOR else "embedding_json"
            has_vectors = any([
                await conn.fetchval(
                    f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL)"
                )
                for table in ("memories", "conversations")
            ])
            raw_source = await conn.fetchval(
                "SELECT value FROM gateway_config WHERE key = 'embedding_vector_source' FOR UPDATE"
            )
            source = json.loads(raw_source) if raw_source else [
                shared.EMBEDDING_BASE_URL.strip().rstrip("/"),
                shared.EMBEDDING_MODEL.strip(), shared.EMBEDDING_DIM,
            ]
            if retain_vectors:
                if (not has_vectors or source[0] == target[0]
                        or source[1:] != target[1:] or rebuilt):
                    raise ValueError("只有地址变更且模型和维度未变时才能保留旧向量")
                raw_detection = await conn.fetchval(
                    "SELECT value FROM gateway_config WHERE key = 'embedding_detection'"
                )
                detection = json.loads(raw_detection) if raw_detection else None
                if (not detection or detection.get("reprobe")
                        or detection.get("base_url") != target[0]
                        or detection.get("model") != target[1]
                        or detection.get("dimension") != dim):
                    raise ValueError("请先用新地址完成向量维度检测，再选择保留旧向量")
            needs_migration = source != target or bool(rebuilt)
            applied = not incomplete and needs_migration and (
                retain_vectors or confirmed or not has_vectors
            )
            if applied:
                if not retain_vectors:
                    for table in ("memories", "conversations"):
                        if table in rebuilt:
                            await conn.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS embedding")
                            await conn.execute(f"ALTER TABLE {table} ADD COLUMN embedding vector({dim})")
                        else:
                            await conn.execute(f"UPDATE {table} SET {column} = NULL WHERE {column} IS NOT NULL")
                await conn.execute("""
                    INSERT INTO gateway_config (key, value)
                    VALUES ('embedding_vector_source', $1)
                    ON CONFLICT (key) DO UPDATE SET value = $1
                """, json.dumps(target))
                source = target
            for key, value in (
                ("EMBEDDING_BASE_URL", base_url),
                ("EMBEDDING_MODEL", model),
                ("EMBEDDING_DIM", str(dim)),
            ):
                await conn.execute("""
                    INSERT INTO gateway_config (key, value) VALUES ($1, $2)
                    ON CONFLICT (key) DO UPDATE SET value = $2
                """, key, value)

        if applied and "memories" in rebuilt:
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding
                    ON memories USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10);
                """)
            except Exception:
                pass  # 搜索仍可用；初期数据不足时索引可能建不起来
    return {
        "source": source,
        "dimensions": {table: dim if applied else current
                       for table, current in dimensions.items()},
        "applied": applied,
        "retained": retain_vectors and applied,
        "rebuilt": rebuilt if applied else [],
        "cleared": applied and has_vectors and not retain_vectors,
        "pending": needs_migration and not applied,
        "has_vectors": has_vectors,
    }


# ============================================================
# 网关配置
# ============================================================

async def get_gateway_config(key: str, default: str = "") -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT value FROM gateway_config WHERE key = $1", key)
        return row['value'] if row else default


async def set_gateway_config(key: str, value: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO gateway_config (key, value) VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = $2
        """, key, value)


async def get_all_gateway_config() -> dict:
    """获取所有配置项"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT key, value FROM gateway_config")
        return {r['key']: r['value'] for r in rows}
