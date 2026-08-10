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
from typing import Optional, List
from datetime import datetime, date, timedelta, timezone as dt_timezone

import asyncpg

# 时区偏移（和 main.py 保持一致）
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))

DATABASE_URL = os.getenv("DATABASE_URL", "")

HAS_PGVECTOR = False  # 在init_tables时检测

# Embedding 配置（向量搜索用）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))

# 记忆向量搜索开关（需要同时设置 EMBEDDING_API_KEY）
MEMORY_VECTOR_ENABLED = os.getenv("MEMORY_VECTOR_ENABLED", "false").lower() == "true"

# 原始对话召回总开关。关闭时不写对话检索索引，也不触发对话向量补算。
CONVERSATION_RECALL_ENABLED = os.getenv(
    "CONVERSATION_RECALL_ENABLED", "false"
).lower() == "true"

# 对话召回使用原始余弦相似度做归一化前过滤，避免弱候选池被拉成满分。
CONVERSATION_MIN_SCORE_THRESHOLD = float(
    os.getenv("CONVERSATION_MIN_SCORE_THRESHOLD", "0.7")
)

CONVERSATION_HW_KEYWORD = float(os.getenv("CONVERSATION_HW_KEYWORD", "0.45"))
CONVERSATION_HW_SEMANTIC = float(os.getenv("CONVERSATION_HW_SEMANTIC", "0.35"))
CONVERSATION_HW_RECENCY = float(os.getenv("CONVERSATION_HW_RECENCY", "0.2"))
_CONVERSATION_CANDIDATE_POOL = 20

# 记忆搜索权重（纯关键词模式）
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.5"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.3"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.2"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.15"))

# 记忆混合搜索权重（MEMORY_VECTOR_ENABLED=true 时生效）
MEMORY_HW_KEYWORD = float(os.getenv("MEMORY_HW_KEYWORD", "0.35"))
MEMORY_HW_SEMANTIC = float(os.getenv("MEMORY_HW_SEMANTIC", "0.35"))
MEMORY_HW_IMPORTANCE = float(os.getenv("MEMORY_HW_IMPORTANCE", "0.15"))
MEMORY_HW_RECENCY = float(os.getenv("MEMORY_HW_RECENCY", "0.15"))
MEMORY_SEMANTIC_THRESHOLD = float(os.getenv("MEMORY_SEMANTIC_THRESHOLD", "0.5"))


# ============================================================
# 连接池管理
# ============================================================

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
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
                metadata        TEXT
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
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'conversations' AND column_name = 'metadata'
                ) THEN
                    ALTER TABLE conversations ADD COLUMN metadata TEXT;
                END IF;
            END $$;
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
        
        # 分区缓存状态表（存储每个session的轮转状态）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS session_cache_state (
                session_id      TEXT PRIMARY KEY,
                summary         TEXT DEFAULT '',
                a_start_round   INTEGER DEFAULT 0,
                seen_fragment_ids TEXT[] DEFAULT '{}',
                seen_fragment_times JSONB DEFAULT '{}'::jsonb,
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
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memories' AND column_name = 'layer'
                ) THEN
                    ALTER TABLE memories ADD COLUMN layer INTEGER DEFAULT 1;
                END IF;
            END $$;
        """)
        
        # title: 记忆标题（语义锚点，用于搜索加权）
        await conn.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memories' AND column_name = 'title'
                ) THEN
                    ALTER TABLE memories ADD COLUMN title TEXT DEFAULT NULL;
                END IF;
            END $$;
        """)
        
        # is_active: 是否参与搜索（碎片合并后变为 false）
        await conn.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memories' AND column_name = 'is_active'
                ) THEN
                    ALTER TABLE memories ADD COLUMN is_active BOOLEAN DEFAULT TRUE;
                END IF;
            END $$;
        """)
        
        # merged_from: 合并来源的碎片ID列表
        await conn.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memories' AND column_name = 'merged_from'
                ) THEN
                    ALTER TABLE memories ADD COLUMN merged_from INTEGER[] DEFAULT NULL;
                END IF;
            END $$;
        """)
        
        # event_date: 事件日期（用于按天整理）
        await conn.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memories' AND column_name = 'event_date'
                ) THEN
                    ALTER TABLE memories ADD COLUMN event_date DATE DEFAULT NULL;
                END IF;
            END $$;
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
        
        # 尝试启用pgvector扩展（向量搜索）
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            HAS_PGVECTOR = True
            print("✅ pgvector扩展已启用")
            
            # 对话表向量列
            await conn.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'conversations' AND column_name = 'embedding'
                    ) THEN
                        ALTER TABLE conversations ADD COLUMN embedding vector({EMBEDDING_DIM});
                    END IF;
                END $$;
            """)
            
            # 记忆表向量列
            await conn.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'memories' AND column_name = 'embedding'
                    ) THEN
                        ALTER TABLE memories ADD COLUMN embedding vector({EMBEDDING_DIM});
                    END IF;
                END $$;
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
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'conversations' AND column_name = 'embedding_json'
                    ) THEN
                        ALTER TABLE conversations ADD COLUMN embedding_json TEXT;
                    END IF;
                END $$;
            """)
            await conn.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'memories' AND column_name = 'embedding_json'
                    ) THEN
                        ALTER TABLE memories ADD COLUMN embedding_json TEXT;
                    END IF;
                END $$;
            """)
    
    print("✅ 数据库表结构已就绪")


# ============================================================
# 中文分词工具（基于 jieba）
# ============================================================

import jieba
import jieba.analyse

# 静默加载词典
jieba.setLogLevel(jieba.logging.INFO)

EN_WORD_PATTERN = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')
NUM_PATTERN = re.compile(r'\d{2,}')
# 清理查询开头的时间戳（如 "2026-05-02 20:26"）
TIMESTAMP_PATTERN = re.compile(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*\d{1,2}:\d{1,2}\s*')

# 中文停用词（高频但无搜索价值的词）
_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "们",
    "这", "那", "有", "和", "与", "也", "都", "又", "就", "但",
    "而", "或", "到", "被", "把", "让", "从", "对", "为", "以",
    "及", "等", "个", "不", "没", "很", "太", "吗", "呢", "吧",
    "啊", "嗯", "哦", "哈", "呀", "嘛", "么", "啦", "哇", "喔",
    "会", "能", "要", "想", "去", "来", "说", "做", "看", "给",
    "上", "下", "里", "中", "大", "小", "多", "少", "好", "可以",
    "什么", "怎么", "如何", "哪里", "哪个", "为什么", "还是",
    "然后", "因为", "所以", "虽然", "但是", "可以", "已经",
    "一个", "一些", "一下", "一点", "一起", "一样",
    "比较", "应该", "可能", "如果", "这个", "那个",
    "自己", "知道", "觉得", "感觉", "时候", "现在",
})

# jieba 用户词典补充（默认词典缺失的词）
for _w in ["手账", "手帐", "搭子", "种草", "拔草", "安利", "内卷", "摆烂", "emo", "网关"]:
    jieba.add_word(_w)


def extract_search_keywords(query: str) -> List[str]:
    """
    从查询中提取搜索关键词（TF-IDF + 正则）

    1. 去掉开头的时间戳噪音
    2. 用 jieba.analyse.extract_tags (TF-IDF) 提取中文关键词
    3. 正则提取英文单词
    4. 保留4位以上数字（年份等，过滤短数字噪音）

    例如：
    "2026-05-02 20:26 写写手账看看书 放松大脑" → ["手账", "放松", "大脑"]
    "我昨天在手机上部署了Render然后吃了晚饭" → ["手机", "部署", "Render", "晚饭"]
    "春节干了什么" → ["春节"]
    "2026除夕"    → ["2026", "除夕"]
    """
    # 去掉时间戳前缀
    cleaned = TIMESTAMP_PATTERN.sub('', query).strip()
    if not cleaned:
        cleaned = query

    keywords = set()

    # 英文单词（2字符以上）
    for match in EN_WORD_PATTERN.finditer(cleaned):
        word = match.group()
        if len(word) >= 2:
            keywords.add(word)

    # 数字串（只保留4位以上，过滤 "05" "20" 这种时间噪音）
    for match in NUM_PATTERN.finditer(cleaned):
        num = match.group()
        if len(num) >= 4:
            keywords.add(num)

    # TF-IDF 关键词提取（比手动分词+停用词好很多）
    tags = jieba.analyse.extract_tags(cleaned, topK=10)
    for tag in tags:
        # 跳过纯英文/数字（已在上面处理）
        if EN_WORD_PATTERN.fullmatch(tag) or NUM_PATTERN.fullmatch(tag):
            continue
        if tag in _STOP_WORDS:
            continue
        keywords.add(tag)

    return list(keywords)


def jieba_tokenize_for_tsv(text: str) -> str:
    """把文本转换为 PostgreSQL simple tsvector 的分词输入。"""
    if not text:
        return ""
    return " ".join(
        word.lower()
        for raw_word in jieba.cut(text, cut_all=False)
        if (word := raw_word.strip()) and word not in _STOP_WORDS
    )


def _conversation_query_terms(query: str) -> tuple[list[str], bool]:
    """对话关键词统一词表；TF-IDF 失效时只保留连续未知中文词组。"""
    keywords = sorted(extract_search_keywords(query))
    if keywords:
        return keywords, False

    phrases = []
    current = []
    for raw_word in jieba.cut(query, cut_all=False):
        word = raw_word.strip()
        if (
            len(word) == 1
            and "\u4e00" <= word <= "\u9fff"
            and word not in _STOP_WORDS
        ):
            current.append(word)
            continue
        if len(current) >= 2:
            phrases.append("".join(current))
        current = []
    if len(current) >= 2:
        phrases.append("".join(current))
    return sorted(set(phrases)), True


def build_tsquery(query: str) -> str:
    """把对话搜索词编码成 tsquery；未知中文词组改走精确子串后备。"""
    tokens, exact_phrase_fallback = _conversation_query_terms(query)
    if exact_phrase_fallback:
        return ""
    escaped = [
        "'" + token.replace("\\", "\\\\").replace("'", "''") + "'"
        for token in tokens
    ]
    return " & ".join(escaped)


# ============================================================
# 向量搜索（OpenAI 兼容 Embedding API）
# ============================================================

async def compute_embedding(text: str) -> list:
    """调用 OpenAI 兼容的 Embedding API 计算文本向量"""
    if not EMBEDDING_API_KEY:
        return []
    
    try:
        import httpx
        
        if len(text) > 4000:
            text = text[:4000]
        
        body = {
            "model": EMBEDDING_MODEL,
            "input": text,
        }
        if EMBEDDING_DIM > 0:
            body["dimensions"] = EMBEDDING_DIM
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{EMBEDDING_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"⚠️ Embedding计算失败: {e}")
        return []


# 搜索记忆与对话时复用同一条 query embedding。持久写入与 backfill 绕过缓存。
QUERY_EMBED_CACHE_TTL = float(os.getenv("QUERY_EMBED_CACHE_TTL", "5"))
QUERY_EMBED_CACHE_MAX = 128
_query_embed_cache = {}
_query_embed_inflight = {}
_query_embed_locks = {}


def _get_query_embed_lock():
    import asyncio

    loop = asyncio.get_running_loop()
    lock = _query_embed_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _query_embed_locks[loop] = lock
    return lock


async def _query_embed_worker(key, query: str) -> list:
    import time

    try:
        vector = await compute_embedding(query)
        if vector:
            if len(_query_embed_cache) >= QUERY_EMBED_CACHE_MAX:
                _query_embed_cache.pop(next(iter(_query_embed_cache)), None)
            _query_embed_cache[key] = (
                time.monotonic() + QUERY_EMBED_CACHE_TTL,
                tuple(vector),
            )
        return vector
    finally:
        _query_embed_inflight.pop(key, None)


async def get_query_embedding(query: str) -> list:
    """短窗口复用 query 向量；失败和空向量不进入缓存。"""
    import asyncio
    import time

    if not EMBEDDING_API_KEY:
        return []
    normalized_query = query.strip()
    if not normalized_query:
        return []

    key = (
        normalized_query,
        EMBEDDING_BASE_URL.rstrip("/"),
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
    )
    lock = _get_query_embed_lock()
    async with lock:
        hit = _query_embed_cache.get(key)
        if hit is not None:
            expires_at, vector = hit
            if time.monotonic() < expires_at:
                _query_embed_cache.pop(key, None)
                _query_embed_cache[key] = (expires_at, vector)
                return list(vector)
            _query_embed_cache.pop(key, None)

        task = _query_embed_inflight.get(key)
        if task is None or task.done():
            task = asyncio.get_running_loop().create_task(
                _query_embed_worker(key, normalized_query)
            )
            _query_embed_inflight[key] = task

    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"⚠️ query向量共享任务失败: {e}")
        return []
    return list(result)


async def save_memory_embedding(conn, memory_id: int, embedding: list):
    """保存记忆向量到memories表"""
    if not embedding:
        return
    
    if HAS_PGVECTOR:
        vec_str = '[' + ','.join(str(f) for f in embedding) + ']'
        await conn.execute(
            "UPDATE memories SET embedding = $1::vector WHERE id = $2",
            vec_str, memory_id
        )
    else:
        import json
        await conn.execute(
            "UPDATE memories SET embedding_json = $1 WHERE id = $2",
            json.dumps(embedding), memory_id
        )


async def save_conversation_embedding(conn, message_id: int, embedding: list):
    """保存单条原始对话向量。"""
    if not embedding:
        return
    if HAS_PGVECTOR:
        vector_text = "[" + ",".join(str(value) for value in embedding) + "]"
        await conn.execute(
            "UPDATE conversations SET embedding = $1::vector WHERE id = $2",
            vector_text,
            message_id,
        )
    else:
        import json

        await conn.execute(
            "UPDATE conversations SET embedding_json = $1 WHERE id = $2",
            json.dumps(embedding),
            message_id,
        )


def _cosine_sim(a, b):
    """余弦相似度（纯Python）"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def _min_max_normalize(scores: dict) -> dict:
    """min-max归一化到0-1"""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    spread = max_v - min_v
    if spread == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / spread for k, v in scores.items()}


# ============================================================
# 对话记录操作
# ============================================================

async def save_message(session_id: str, role: str, content: str, model: str = "", metadata: str = None):
    tsv_text = (
        jieba_tokenize_for_tsv(content or "")
        if CONVERSATION_RECALL_ENABLED
        else None
    )
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO conversations (
                   session_id, role, content, model, metadata, content_tsv
               ) VALUES (
                   $1, $2, $3, $4, $5,
                   array_to_tsvector(string_to_array($6, ' '))
               )
               RETURNING id""",
            session_id, role, content, model, metadata, tsv_text,
        )
    message_id = row["id"] if row else None
    if (
        message_id is not None
        and CONVERSATION_RECALL_ENABLED
        and EMBEDDING_API_KEY
        and content
        and content.strip()
    ):
        kick_embedding_backfill()
    return message_id


async def get_last_user_content(session_id: str) -> str:
    """获取指定session最后一条user消息的content"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT content FROM conversations
            WHERE session_id = $1 AND role = 'user'
            ORDER BY created_at DESC
            LIMIT 1
        """, session_id)
        return row['content'] if row else ""


async def update_last_assistant_message(session_id: str, new_content: str, model: str = ""):
    """覆盖指定session最后一条assistant消息的content（用于re-roll去重）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM conversations
            WHERE session_id = $1 AND role = 'assistant'
            ORDER BY created_at DESC
            LIMIT 1
        """, session_id)
        if row:
            embedding_column = "embedding" if HAS_PGVECTOR else "embedding_json"
            tsv_text = (
                jieba_tokenize_for_tsv(new_content or "")
                if CONVERSATION_RECALL_ENABLED
                else None
            )
            await conn.execute(
                f"""UPDATE conversations
                   SET content = $1,
                       model = $2,
                       content_tsv = array_to_tsvector(string_to_array($3, ' ')),
                       {embedding_column} = NULL
                   WHERE id = $4""",
                new_content, model, tsv_text, row['id']
            )
            if (
                CONVERSATION_RECALL_ENABLED
                and EMBEDDING_API_KEY
                and new_content
                and new_content.strip()
            ):
                kick_embedding_backfill()
            return True
        return False


async def get_recent_messages(session_id: str, limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content, metadata, created_at FROM conversations WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, limit,
        )
        return list(reversed(rows))


async def search_conversations(query: str, limit: int = 20, offset: int = 0):
    """搜索对话内容，返回匹配的session列表"""
    keywords = extract_search_keywords(query)
    if not keywords:
        return [], 0
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        where_parts = []
        params = []
        for i, kw in enumerate(keywords):
            where_parts.append(f"c.content ILIKE '%' || ${i+1} || '%'")
            params.append(kw)
        where_clause = " OR ".join(where_parts)
        
        count_sql = f"""
            SELECT COUNT(DISTINCT c.session_id) as total
            FROM conversations c
            WHERE {where_clause}
        """
        total_row = await conn.fetchrow(count_sql, *params)
        total = total_row['total'] if total_row else 0
        
        if total == 0:
            return [], 0
        
        limit_idx = len(params) + 1
        offset_idx = len(params) + 2
        params.extend([limit, offset])
        
        sql = f"""
            WITH matched_sessions AS (
                SELECT DISTINCT c.session_id
                FROM conversations c
                WHERE {where_clause}
            ),
            session_info AS (
                SELECT 
                    ms.session_id,
                    MIN(c.created_at) as first_time,
                    MAX(c.created_at) as last_time,
                    COUNT(*) as message_count
                FROM matched_sessions ms
                JOIN conversations c ON c.session_id = ms.session_id
                GROUP BY ms.session_id
            )
            SELECT 
                si.session_id,
                si.first_time,
                si.last_time,
                si.message_count
            FROM session_info si
            ORDER BY si.last_time DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """
        rows = await conn.fetch(sql, *params)
        
        results = []
        for r in rows:
            results.append({
                'session_id': r['session_id'],
                'first_time': r['first_time'].isoformat() if r['first_time'] else None,
                'last_time': r['last_time'].isoformat() if r['last_time'] else None,
                'message_count': r['message_count'],
            })
        
        return results, total


async def update_message_content(message_id: int, new_content: str):
    """更新单条对话消息的内容"""
    tsv_text = (
        jieba_tokenize_for_tsv(new_content or "")
        if CONVERSATION_RECALL_ENABLED
        else None
    )
    pool = await get_pool()
    async with pool.acquire() as conn:
        embedding_column = "embedding" if HAS_PGVECTOR else "embedding_json"
        result = await conn.execute(
            f"""UPDATE conversations
               SET content = $1,
                   content_tsv = array_to_tsvector(string_to_array($2, ' ')),
                   {embedding_column} = NULL
               WHERE id = $3""",
            new_content, tsv_text, message_id,
        )
    updated = int(result.split()[-1]) if result else 0
    if (
        updated
        and CONVERSATION_RECALL_ENABLED
        and EMBEDDING_API_KEY
        and new_content
        and new_content.strip()
    ):
        kick_embedding_backfill()
    return updated


async def delete_single_message(message_id: int):
    """删除单条对话消息（硬删除）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM conversations WHERE id = $1",
            message_id,
        )
        return int(result.split()[-1]) if result else 0


# ============================================================
# 原始对话片段召回
# ============================================================

def _fragment_id(anchor_ids) -> str | None:
    """由命中消息的持久 conversations.id 生成稳定片段 ID。"""
    import hashlib

    ids = sorted({message_id for message_id in anchor_ids if message_id is not None})
    if not ids:
        return None
    payload = "chat-fragment:v1:" + ",".join(str(message_id) for message_id in ids)
    return f"v1:{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


def _assemble_fragments(all_messages, sorted_indices, matched_indices):
    fragments = []
    fragment_ids = []
    current = []
    current_anchors = []
    previous_index = -2

    for index in sorted_indices:
        if index != previous_index + 1 and current:
            fragments.append(current)
            fragment_ids.append(_fragment_id(current_anchors))
            current = []
            current_anchors = []

        message = all_messages[index]
        content = message["content"] or ""
        is_match = index in matched_indices
        max_chars = 200 if is_match else 80
        if len(content) > max_chars:
            content = content[:max_chars] + "…（省略）"
        created_at = message["created_at"]
        if created_at is not None and hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()
        current.append({
            "role": message["role"],
            "content": content,
            "created_at": created_at,
            "is_match": is_match,
        })
        if is_match:
            current_anchors.append(message["id"])
        previous_index = index

    if current:
        fragments.append(current)
        fragment_ids.append(_fragment_id(current_anchors))
    return fragments, fragment_ids


async def _conversation_tsv_ready(conn) -> bool:
    pending = await conn.fetchval(
        r"""SELECT COUNT(*) FROM conversations
            WHERE content_tsv IS NULL
              AND content IS NOT NULL AND content !~ '^\s*$'"""
    )
    return not pending


def _session_exclusion_sql(exclude_session_ids: list, param_index: int) -> tuple[str, list]:
    if not exclude_session_ids:
        return "", []
    return f" AND NOT (session_id = ANY(${param_index}::text[]))", [exclude_session_ids]


async def _keyword_session_scores(conn, tsquery: str, keyword_terms: list[str],
                                  pool_size: int, exclude_session_ids: list):
    if not keyword_terms:
        return {}

    if not tsquery or not await _conversation_tsv_ready(conn):
        conditions = [
            f"content ILIKE '%' || ${index + 1} || '%'"
            for index in range(len(keyword_terms))
        ]
        params = list(keyword_terms)
        exclusion_sql, exclusion_params = _session_exclusion_sql(
            exclude_session_ids, len(params) + 1
        )
        params.extend(exclusion_params)
        rows = await conn.fetch(
            f"""SELECT session_id, COUNT(*)::float AS score,
                       MAX(created_at) AS latest_match
                FROM conversations
                WHERE ({' AND '.join(conditions)}) {exclusion_sql}
                GROUP BY session_id
                ORDER BY score DESC
                LIMIT {int(pool_size)}""",
            *params,
        )
    else:
        params = [tsquery]
        exclusion_sql, exclusion_params = _session_exclusion_sql(
            exclude_session_ids, len(params) + 1
        )
        params.extend(exclusion_params)
        params.append(pool_size)
        rows = await conn.fetch(
            f"""SELECT session_id,
                       MAX(ts_rank(content_tsv, $1::tsquery, 2)) AS score,
                       MAX(created_at) AS latest_match
                FROM conversations
                WHERE content_tsv @@ $1::tsquery {exclusion_sql}
                GROUP BY session_id
                ORDER BY score DESC
                LIMIT ${len(params)}""",
            *params,
        )
    return {
        row["session_id"]: {
            "score": float(row["score"]),
            "latest": row["latest_match"],
        }
        for row in rows
    }


async def _semantic_session_scores(conn, query_embedding: list, pool_size: int,
                                   exclude_session_ids: list):
    """先按原始余弦阈值过滤，再交给融合层归一化。"""
    if not query_embedding:
        return {}

    if HAS_PGVECTOR:
        vector_text = "[" + ",".join(str(value) for value in query_embedding) + "]"
        params = [vector_text]
        exclusion_sql, exclusion_params = _session_exclusion_sql(
            exclude_session_ids, len(params) + 1
        )
        params.extend(exclusion_params)
        ranked_limit = max(100, pool_size * 10)
        params.extend([ranked_limit, CONVERSATION_MIN_SCORE_THRESHOLD, pool_size])
        ranked_limit_index = len(params) - 2
        threshold_index = len(params) - 1
        pool_index = len(params)
        rows = await conn.fetch(
            f"""WITH ranked AS (
                    SELECT session_id,
                           1 - (embedding <=> $1::vector) AS similarity,
                           created_at
                    FROM conversations
                    WHERE embedding IS NOT NULL {exclusion_sql}
                    ORDER BY embedding <=> $1::vector
                    LIMIT ${ranked_limit_index}
                )
                SELECT session_id, MAX(similarity) AS score,
                       MAX(created_at) AS latest_match
                FROM ranked
                WHERE similarity >= ${threshold_index}
                GROUP BY session_id
                ORDER BY score DESC
                LIMIT ${pool_index}""",
            *params,
        )
        return {
            row["session_id"]: {
                "score": float(row["score"]),
                "latest": row["latest_match"],
            }
            for row in rows
        }

    import json

    params = []
    exclusion_sql, exclusion_params = _session_exclusion_sql(
        exclude_session_ids, 1
    )
    params.extend(exclusion_params)
    rows = await conn.fetch(
        f"""SELECT session_id, created_at, embedding_json
            FROM conversations
            WHERE embedding_json IS NOT NULL {exclusion_sql}""",
        *params,
    )
    session_best = {}
    for row in rows:
        try:
            similarity = _cosine_sim(query_embedding, json.loads(row["embedding_json"]))
        except Exception:
            continue
        if similarity < CONVERSATION_MIN_SCORE_THRESHOLD:
            continue
        current = session_best.get(row["session_id"])
        if current is None or similarity > current["score"]:
            session_best[row["session_id"]] = {
                "score": similarity,
                "latest": row["created_at"],
            }
        elif row["created_at"] and row["created_at"] > current["latest"]:
            current["latest"] = row["created_at"]
    return dict(
        sorted(session_best.items(), key=lambda item: -item[1]["score"])[:pool_size]
    )


async def search_chat_fragments(
    query: str,
    max_sessions: int = 3,
    max_matches_per_session: int = 1,
    context: int = 1,
    mode: str = "hybrid",
    exclude_session_ids: list | None = None,
    exclude_fragment_ids: list | None = None,
):
    """检索历史对话。raw API 无状态，排除集合完全由调用方传入。"""
    from datetime import datetime, timezone

    if not CONVERSATION_RECALL_ENABLED:
        return [], 0
    query = query.strip()
    if not query or mode not in {"keyword", "hybrid"}:
        return [], 0

    exclude_session_ids = sorted({str(value) for value in (exclude_session_ids or []) if value})
    excluded_fragments = {str(value) for value in (exclude_fragment_ids or []) if value}
    max_sessions = min(50, max(1, int(max_sessions)))
    max_matches_per_session = min(5, max(1, int(max_matches_per_session)))
    context = min(5, max(0, int(context)))

    keyword_terms, _ = _conversation_query_terms(query)
    tsquery = build_tsquery(query)
    query_embedding = (
        await get_query_embedding(query)
        if mode == "hybrid" and EMBEDDING_API_KEY
        else []
    )
    pool_size = max(_CONVERSATION_CANDIDATE_POOL, max_sessions * 3)
    pool = await get_pool()
    async with pool.acquire() as conn:
        keyword_scores = await _keyword_session_scores(
            conn, tsquery, keyword_terms, pool_size, exclude_session_ids
        )
        semantic_scores = (
            await _semantic_session_scores(
                conn, query_embedding, pool_size, exclude_session_ids
            )
            if mode == "hybrid"
            else {}
        )

    session_ids = set(keyword_scores) | set(semantic_scores)
    if not session_ids:
        return [], 0

    keyword_normalized = _min_max_normalize({
        session_id: item["score"] for session_id, item in keyword_scores.items()
    })
    semantic_normalized = _min_max_normalize({
        session_id: item["score"] for session_id, item in semantic_scores.items()
    })
    now = datetime.now(timezone.utc)
    recency = {}
    for session_id in session_ids:
        timestamps = [
            source[session_id]["latest"]
            for source in (keyword_scores, semantic_scores)
            if session_id in source and source[session_id]["latest"]
        ]
        if timestamps:
            age_days = (now - max(timestamps)).total_seconds() / 86400.0
            recency[session_id] = 1.0 / (1.0 + max(0.0, age_days))
        else:
            recency[session_id] = 0.0
    recency_normalized = _min_max_normalize(recency)

    if mode == "keyword":
        final_scores = keyword_normalized
    else:
        final_scores = {
            session_id: (
                CONVERSATION_HW_KEYWORD * keyword_normalized.get(session_id, 0.0)
                + CONVERSATION_HW_SEMANTIC * semantic_normalized.get(session_id, 0.0)
                + CONVERSATION_HW_RECENCY * recency_normalized.get(session_id, 0.0)
            )
            for session_id in session_ids
        }
    ranked = sorted(final_scores.items(), key=lambda item: -item[1])

    vector_text = None
    if HAS_PGVECTOR and query_embedding:
        vector_text = "[" + ",".join(str(value) for value in query_embedding) + "]"
    results = []
    async with pool.acquire() as conn:
        for session_id, final_score in ranked:
            if len(results) >= max_sessions:
                break
            if HAS_PGVECTOR and vector_text:
                messages = await conn.fetch(
                    """SELECT id, role, content, created_at,
                              CASE WHEN embedding IS NOT NULL
                                   THEN 1 - (embedding <=> $2::vector)
                                   ELSE 0 END AS sem_sim
                       FROM conversations
                       WHERE session_id = $1
                       ORDER BY created_at ASC, id ASC""",
                    session_id, vector_text,
                )
            else:
                embedding_column = "embedding_json" if not HAS_PGVECTOR else "NULL::text"
                messages = await conn.fetch(
                    f"""SELECT id, role, content, created_at,
                               {embedding_column} AS embedding_json
                        FROM conversations
                        WHERE session_id = $1
                        ORDER BY created_at ASC, id ASC""",
                    session_id,
                )

            marked = []
            for message in messages:
                lowered = (message["content"] or "").lower()
                keyword_match = bool(keyword_terms) and all(
                    term.lower() in lowered for term in keyword_terms
                )
                semantic_similarity = 0.0
                if HAS_PGVECTOR and vector_text:
                    semantic_similarity = float(message["sem_sim"] or 0)
                elif query_embedding and message["embedding_json"]:
                    try:
                        import json
                        semantic_similarity = _cosine_sim(
                            query_embedding, json.loads(message["embedding_json"])
                        )
                    except Exception:
                        semantic_similarity = 0.0
                semantic_match = (
                    mode == "hybrid"
                    and semantic_similarity >= CONVERSATION_MIN_SCORE_THRESHOLD
                )
                marked.append({
                    "id": message["id"],
                    "role": message["role"],
                    "content": message["content"] or "",
                    "created_at": message["created_at"],
                    "is_match": keyword_match or semantic_match,
                    "relevance": 1.0 if keyword_match else semantic_similarity,
                })

            match_candidates = [
                (index, item["relevance"])
                for index, item in enumerate(marked)
                if item["is_match"]
            ]
            match_candidates.sort(key=lambda item: -item[1])
            if not match_candidates:
                continue
            total_matched = len(match_candidates)
            kept = []
            for match_index, _ in match_candidates:
                context_indices = range(
                    max(0, match_index - context),
                    min(len(marked), match_index + context + 1),
                )
                fragments, fragment_ids = _assemble_fragments(
                    marked, list(context_indices), {match_index}
                )
                fragment_id = fragment_ids[0] if fragment_ids else None
                if not fragment_id or fragment_id in excluded_fragments:
                    continue
                kept.append((fragments[0], fragment_id))
                if len(kept) >= max_matches_per_session:
                    break
            if not kept:
                continue
            results.append({
                "session_id": session_id,
                "title": session_id,
                "total_messages": len(marked),
                "match_count": total_matched,
                "fragments": [item[0] for item in kept],
                "fragment_ids": [item[1] for item in kept],
                "has_more_matches": total_matched > len(kept),
                "hybrid_scores": {
                    "kw_raw": round(keyword_scores.get(session_id, {}).get("score", 0.0), 6),
                    "kw": round(keyword_normalized.get(session_id, 0.0), 3),
                    "sem_raw": round(semantic_scores.get(session_id, {}).get("score", 0.0), 6),
                    "sem": round(semantic_normalized.get(session_id, 0.0), 3),
                    "rec": round(recency_normalized.get(session_id, 0.0), 3),
                    "final": round(final_score, 3),
                },
            })

    return results, len(results)


# ============================================================
# 记忆操作
# ============================================================

async def save_memory(content: str, importance: int = 5, source_session: str = "",
                      created_at: datetime = None):
    """created_at 传入时保留原时间（备份恢复用），否则落库默认 NOW()"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO memories (content, importance, source_session, created_at) "
            "VALUES ($1, $2, $3, COALESCE($4, NOW())) RETURNING id",
            content, importance, source_session, created_at,
        )
        
        # MEMORY_VECTOR_ENABLED 时自动计算 embedding
        if MEMORY_VECTOR_ENABLED and row:
            try:
                embedding = await compute_embedding(content)
                if embedding:
                    await save_memory_embedding(conn, row['id'], embedding)
            except Exception as e:
                print(f"⚠️ 记忆 {row['id']} embedding自动计算失败: {e}")
        return row["id"] if row else None


async def search_memories(query: str, limit: int = 10):
    """
    搜索相关记忆
    
    MEMORY_VECTOR_ENABLED=true 时走混合搜索（关键词 + 向量）
    否则走纯关键词搜索
    """
    if MEMORY_VECTOR_ENABLED:
        return await search_memories_hybrid(query, limit)
    
    # ---- 纯关键词搜索 ----
    keywords = extract_search_keywords(query)
    
    if not keywords:
        return []
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 每个关键词命中得1分
        case_parts = []
        params = []
        for i, kw in enumerate(keywords):
            case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
            params.append(kw)
        
        hit_count_expr = " + ".join(case_parts)
        max_hits = len(keywords)
        
        # 至少命中一个关键词（只搜索活跃记忆）
        where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
        where_clause = f"is_active = TRUE AND ({' OR '.join(where_parts)})"
        
        limit_idx = len(keywords) + 1
        params.append(limit)
        
        # 时效天数：事件记忆按本地日历日差（AT TIME ZONE 'UTC' 拿到无时区的 UTC 挂钟，
        # 加偏移后取日期，不依赖数据库会话时区）；普通碎片按 created_at 精确时长
        recency_days_expr = (
            "CASE WHEN event_date IS NOT NULL "
            f"THEN GREATEST(0, ((NOW() AT TIME ZONE 'UTC' + INTERVAL '{TIMEZONE_HOURS} hours')::date - event_date))::float "
            "ELSE GREATEST(0, EXTRACT(EPOCH FROM (NOW() - created_at))) / 86400.0 END"
        )
        sql = f"""
            SELECT
                id, content, importance, created_at, event_date,
                ({hit_count_expr}) AS hit_count,
                ({recency_days_expr}) AS effective_days,
                (
                    {WEIGHT_KEYWORD} * ({hit_count_expr})::float / {max_hits}.0 +
                    {WEIGHT_IMPORTANCE} * importance::float / 10.0 +
                    {WEIGHT_RECENCY} * (1.0 / (1.0 + ({recency_days_expr})))
                ) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY score DESC, importance DESC, effective_days ASC
            LIMIT ${limit_idx}
        """
        
        results = await conn.fetch(sql, *params)
        
        # 过滤低分记忆
        if MIN_SCORE_THRESHOLD > 0:
            before_count = len(results)
            results = [r for r in results if r['score'] >= MIN_SCORE_THRESHOLD]
            filtered = before_count - len(results)
        else:
            filtered = 0
        
        if results:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]}{'...' if len(keywords)>8 else ''} → 命中 {len(results)} 条" + (f"（过滤 {filtered} 条低分）" if filtered else ""))
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (hits={r['hit_count']}, imp={r['importance']}) {r['content'][:60]}...")
            
            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]} → 无结果" + (f"（{filtered} 条被分数阈值过滤）" if filtered else ""))
        
        return results


def _effective_days_ago(event_date, created_at, now_utc):
    """时效天数：事件记忆按本地日历日差（event_date 是本地日期，不做 UTC 换算），
    普通碎片按 created_at 精确时长"""
    if event_date:
        local_today = (now_utc + timedelta(hours=TIMEZONE_HOURS)).date()
        return max(0.0, float((local_today - event_date).days))
    return max(0.0, (now_utc - created_at).total_seconds() / 86400.0)


async def search_memories_hybrid(query: str, limit: int = 10):
    """
    记忆混合搜索：关键词 + 向量，归一化后四维加权
    
    权重：MEMORY_HW_KEYWORD + MEMORY_HW_SEMANTIC + MEMORY_HW_IMPORTANCE + MEMORY_HW_RECENCY
    """
    from datetime import datetime, timezone
    
    keywords = extract_search_keywords(query)
    query_embedding = await get_query_embedding(query) if EMBEDDING_API_KEY else []
    
    if not keywords and not query_embedding:
        return []
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        candidates = {}  # id -> {content, importance, created_at, kw_score, similarity}
        
        # ---- 关键词路 ----
        if keywords:
            case_parts = []
            params = []
            for i, kw in enumerate(keywords):
                case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
                params.append(kw)
            
            hit_count_expr = " + ".join(case_parts)
            max_hits = len(keywords)
            where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
            where_clause = f"is_active = TRUE AND ({' OR '.join(where_parts)})"
            
            limit_idx = len(keywords) + 1
            params.append(limit * 3)
            
            kw_sql = f"""
                SELECT id, content, importance, created_at, event_date,
                       ({hit_count_expr}) AS hit_count,
                       ({hit_count_expr})::float / {max_hits}.0 AS kw_score
                FROM memories
                WHERE {where_clause}
                ORDER BY kw_score DESC
                LIMIT ${limit_idx}
            """
            kw_rows = await conn.fetch(kw_sql, *params)
            
            for r in kw_rows:
                candidates[r['id']] = {
                    'content': r['content'],
                    'importance': r['importance'],
                    'created_at': r['created_at'],
                    'event_date': r['event_date'],
                    'hit_count': r['hit_count'],
                    'kw_score': float(r['kw_score']),
                    'similarity': 0.0,
                }
        
        # ---- 向量路 ----
        if query_embedding:
            if HAS_PGVECTOR:
                vec_str = '[' + ','.join(str(f) for f in query_embedding) + ']'
                sem_rows = await conn.fetch("""
                    SELECT id, content, importance, created_at, event_date,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM memories
                    WHERE embedding IS NOT NULL AND is_active = TRUE
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """, vec_str, limit * 3)
            else:
                # Python端计算cosine
                import json
                all_mem = await conn.fetch("""
                    SELECT id, content, importance, created_at, event_date, embedding_json
                    FROM memories WHERE embedding_json IS NOT NULL AND is_active = TRUE
                """)
                
                scored = []
                for row in all_mem:
                    try:
                        emb = json.loads(row['embedding_json'])
                        sim = _cosine_sim(query_embedding, emb)
                        scored.append({**dict(row), 'similarity': sim})
                    except Exception:
                        continue
                scored.sort(key=lambda x: -x['similarity'])
                sem_rows = scored[:limit * 3]
            
            for r in sem_rows:
                sim = float(r['similarity'])
                if sim < MEMORY_SEMANTIC_THRESHOLD:
                    continue
                mid = r['id']
                if mid in candidates:
                    candidates[mid]['similarity'] = sim
                else:
                    candidates[mid] = {
                        'content': r['content'],
                        'importance': r['importance'],
                        'created_at': r['created_at'],
                        'event_date': r['event_date'],
                        'hit_count': 0,
                        'kw_score': 0.0,
                        'similarity': sim,
                    }
            
            # debug：向量路统计
            sem_total = len(sem_rows)
            sem_passed = sum(1 for r in sem_rows if float(r['similarity']) >= MEMORY_SEMANTIC_THRESHOLD)
            sem_max = max((float(r['similarity']) for r in sem_rows), default=0)
            if sem_total > 0 and sem_passed == 0:
                print(f"   🔢 向量路: {sem_total}条候选全被阈值过滤（最高sim={sem_max:.3f}, 阈值={MEMORY_SEMANTIC_THRESHOLD}）")
            elif sem_total > 0:
                print(f"   🔢 向量路: {sem_passed}/{sem_total}条通过阈值（最高sim={sem_max:.3f}）")
        
        if not candidates:
            print(f"🔍 混合搜索 '{query}' → 两路均无结果")
            return []
        
        # ---- 归一化 + 加权 ----
        kw_norm = _min_max_normalize({mid: v['kw_score'] for mid, v in candidates.items()})
        sem_norm = _min_max_normalize({mid: v['similarity'] for mid, v in candidates.items()})
        
        now = datetime.now(timezone.utc)
        final = []
        for mid, info in candidates.items():
            kw = kw_norm.get(mid, 0.0)
            sem = sem_norm.get(mid, 0.0)
            imp = info['importance'] / 10.0
            days = _effective_days_ago(info.get('event_date'), info['created_at'], now)
            rec = 1.0 / (1.0 + days)
            
            score = (MEMORY_HW_KEYWORD * kw +
                     MEMORY_HW_SEMANTIC * sem +
                     MEMORY_HW_IMPORTANCE * imp +
                     MEMORY_HW_RECENCY * rec)
            
            final.append({
                'id': mid,
                'content': info['content'],
                'importance': info['importance'],
                'created_at': info['created_at'],
                'event_date': info.get('event_date'),
                'hit_count': info['hit_count'],
                'similarity': info['similarity'],
                'score': score,
            })
        
        final.sort(key=lambda x: (-x['score'], -x['importance']))
        
        # 过滤低分
        if MIN_SCORE_THRESHOLD > 0:
            before_count = len(final)
            final = [r for r in final if r['score'] >= MIN_SCORE_THRESHOLD]
            filtered = before_count - len(final)
        else:
            filtered = 0
        
        results = final[:limit]
        
        if results:
            mode_tag = "混合" if query_embedding else "关键词"
            kw_tag = f"关键词 {keywords[:6]}" if keywords else "无关键词"
            print(f"🔍 {mode_tag}搜索 '{query}' → {kw_tag} → 命中 {len(results)} 条" + (f"（过滤 {filtered} 条低分）" if filtered else ""))
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (kw={r['hit_count']}, sim={r['similarity']:.2f}, imp={r['importance']}) {r['content'][:60]}...")
            
            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 混合搜索 '{query}' → 无结果" + (f"（{filtered} 条被过滤）" if filtered else ""))
        
        return [dict(r) for r in results]


async def get_pending_memory_embedding_count():
    """查询还没有embedding的记忆数量"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if HAS_PGVECTOR:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NULL AND content IS NOT NULL"
            )
        else:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE embedding_json IS NULL AND content IS NOT NULL"
            )


async def backfill_memory_embeddings(batch_size: int = 20):
    """给已有记忆补算embedding（没有embedding的记忆）"""
    if not EMBEDDING_API_KEY:
        print("⚠️ EMBEDDING_API_KEY 未设置，无法补算embedding")
        return 0
    
    pool = await get_pool()
    total_updated = 0
    
    async with pool.acquire() as conn:
        if HAS_PGVECTOR:
            rows = await conn.fetch("""
                SELECT id, content FROM memories 
                WHERE embedding IS NULL AND content IS NOT NULL
                ORDER BY id
                LIMIT $1
            """, batch_size)
        else:
            rows = await conn.fetch("""
                SELECT id, content FROM memories 
                WHERE embedding_json IS NULL AND content IS NOT NULL
                ORDER BY id
                LIMIT $1
            """, batch_size)
    
    if not rows:
        print("✅ 所有记忆已有embedding，无需补算")
        return 0
    
    print(f"🔄 开始补算记忆embedding... 本批 {len(rows)} 条")
    
    async with pool.acquire() as conn:
        for row in rows:
            try:
                embedding = await compute_embedding(row['content'] or '')
                if embedding:
                    await save_memory_embedding(conn, row['id'], embedding)
                    total_updated += 1
            except Exception as e:
                print(f"⚠️ 记忆 {row['id']} embedding计算失败: {e}")
    
    # 检查剩余
    async with pool.acquire() as conn:
        if HAS_PGVECTOR:
            remaining = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE embedding IS NULL AND content IS NOT NULL")
        else:
            remaining = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE embedding_json IS NULL AND content IS NOT NULL")
    
    print(f"✅ 本批补算完成：{total_updated}/{len(rows)} 条成功" + (f"，剩余 {remaining} 条待处理" if remaining > 0 else ""))
    return total_updated


# ============================================================
# 对话检索索引与向量持续补算
# ============================================================

async def rebuild_content_tsv(batch_size: int = 200):
    """以 content_tsv IS NULL 为持久账本，分批补齐关键词索引。"""
    if not CONVERSATION_RECALL_ENABLED:
        return 0
    pool = await get_pool()
    total_updated = 0
    last_id = 0
    while True:
        if not CONVERSATION_RECALL_ENABLED:
            break
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                r"""SELECT id, content FROM conversations
                    WHERE content_tsv IS NULL AND id > $2
                      AND content IS NOT NULL AND content !~ '^\s*$'
                    ORDER BY id
                    LIMIT $1""",
                batch_size, last_id,
            )
        if not rows:
            break
        last_id = rows[-1]["id"]
        row_ids = [row["id"] for row in rows]
        tsv_texts = [jieba_tokenize_for_tsv(row["content"] or "") for row in rows]
        if not CONVERSATION_RECALL_ENABLED:
            break
        async with pool.acquire() as conn:
            await conn.execute(
                """UPDATE conversations AS c
                   SET content_tsv = array_to_tsvector(
                       string_to_array(batch.tsv_text, ' ')
                   )
                   FROM UNNEST($1::int[], $2::text[]) AS batch(id, tsv_text)
                   WHERE c.id = batch.id AND c.content_tsv IS NULL""",
                row_ids, tsv_texts,
            )
        total_updated += len(rows)
    return total_updated


EMBED_BACKFILL_SLEEP = float(os.getenv("EMBED_BACKFILL_SLEEP", "0.7"))
EMBED_BACKFILL_FAIL_LIMIT = int(os.getenv("EMBED_BACKFILL_FAIL_LIMIT", "10"))
EMBED_BACKFILL_BATCH = int(os.getenv("EMBED_BACKFILL_BATCH", "50"))

_embed_backfill_task = None
_embed_backfill_rerun = False
_embed_backfill_state = {
    "running": False,
    "done_count": 0,
    "fail_count": 0,
    "last_error": None,
    "stopped_reason": None,
    "last_run_at": None,
}


def _conversation_embedding_pending_condition() -> str:
    column = "embedding" if HAS_PGVECTOR else "embedding_json"
    return rf"{column} IS NULL AND content IS NOT NULL AND content !~ '^\s*$'"


async def backfill_conversation_embeddings_once(
    sleep_seconds: float | None = None,
    fail_limit: int | None = None,
):
    """补算非空 NULL 向量；失败项保持 NULL，下一次可续跑。"""
    import asyncio as _asyncio

    state = _embed_backfill_state
    state.update({
        "running": True,
        "done_count": 0,
        "fail_count": 0,
        "last_error": None,
        "stopped_reason": None,
    })
    if sleep_seconds is None:
        sleep_seconds = EMBED_BACKFILL_SLEEP
    if fail_limit is None:
        fail_limit = EMBED_BACKFILL_FAIL_LIMIT

    try:
        if not CONVERSATION_RECALL_ENABLED:
            state["stopped_reason"] = "recall_disabled"
            return 0, 0, None
        if not EMBEDDING_API_KEY:
            state["last_error"] = "EMBEDDING_API_KEY未设置"
            state["stopped_reason"] = "no_api_key"
            return 0, 0, state["last_error"]

        pool = await get_pool()
        condition = _conversation_embedding_pending_condition()
        consecutive_failures = 0
        last_id = 0
        while True:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT id, content FROM conversations
                        WHERE {condition} AND id > $2
                        ORDER BY id
                        LIMIT $1""",
                    EMBED_BACKFILL_BATCH, last_id,
                )
            if not rows:
                break
            last_id = rows[-1]["id"]

            for row in rows:
                if not CONVERSATION_RECALL_ENABLED:
                    state["stopped_reason"] = "recall_disabled"
                    return state["done_count"], state["fail_count"], state["last_error"]
                try:
                    vector = await compute_embedding(row["content"] or "")
                except Exception as exc:
                    vector = []
                    state["last_error"] = str(exc)

                if vector:
                    try:
                        async with pool.acquire() as conn:
                            await save_conversation_embedding(conn, row["id"], vector)
                        state["done_count"] += 1
                        consecutive_failures = 0
                    except Exception as exc:
                        state["fail_count"] += 1
                        consecutive_failures += 1
                        state["last_error"] = f"写回失败 id={row['id']}: {exc}"
                else:
                    state["fail_count"] += 1
                    consecutive_failures += 1
                    state["last_error"] = f"embedding计算返回空 id={row['id']}"

                if consecutive_failures >= fail_limit:
                    state["stopped_reason"] = f"连续失败{consecutive_failures}条，本轮停止"
                    return state["done_count"], state["fail_count"], state["last_error"]
                if sleep_seconds > 0:
                    await _asyncio.sleep(sleep_seconds)

        return state["done_count"], state["fail_count"], state["last_error"]
    finally:
        state["running"] = False
        state["last_run_at"] = datetime.now(dt_timezone.utc).isoformat()


async def _conversation_embedding_backfill_runner():
    global _embed_backfill_rerun

    try:
        while True:
            _embed_backfill_rerun = False
            await backfill_conversation_embeddings_once()
            if not _embed_backfill_rerun or _embed_backfill_state["stopped_reason"]:
                break
    except Exception as exc:
        _embed_backfill_state["last_error"] = str(exc)
        _embed_backfill_state["running"] = False


def kick_embedding_backfill() -> bool:
    """单实例唤醒补算器；运行中只登记再跑一轮。"""
    global _embed_backfill_task, _embed_backfill_rerun
    import asyncio as _asyncio

    if not CONVERSATION_RECALL_ENABLED or not EMBEDDING_API_KEY:
        return False
    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        return False
    if _embed_backfill_task is not None and not _embed_backfill_task.done():
        _embed_backfill_rerun = True
        return False
    _embed_backfill_task = loop.create_task(_conversation_embedding_backfill_runner())
    return True


async def get_embedding_backfill_status():
    remaining = None
    cumulative_embedded = None
    content_tsv_remaining = None
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                f"SELECT COUNT(*) FROM conversations WHERE {_conversation_embedding_pending_condition()}"
            )
            embedding_column = "embedding" if HAS_PGVECTOR else "embedding_json"
            cumulative_embedded = await conn.fetchval(
                f"SELECT COUNT(*) FROM conversations WHERE {embedding_column} IS NOT NULL"
            )
            content_tsv_remaining = await conn.fetchval(
                r"""SELECT COUNT(*) FROM conversations
                    WHERE content_tsv IS NULL
                      AND content IS NOT NULL AND content !~ '^\s*$'"""
            )
    except Exception as exc:
        if not _embed_backfill_state["last_error"]:
            _embed_backfill_state["last_error"] = f"查询补算状态失败: {exc}"

    return {
        "enabled": CONVERSATION_RECALL_ENABLED,
        "running": _embed_backfill_state["running"],
        "last_run_done_count": _embed_backfill_state["done_count"],
        "fail_count": _embed_backfill_state["fail_count"],
        "last_error": _embed_backfill_state["last_error"],
        "stopped_reason": _embed_backfill_state["stopped_reason"],
        "last_run_at": _embed_backfill_state["last_run_at"],
        "remaining": remaining,
        "cumulative_embedded": cumulative_embedded,
        "content_tsv_remaining": content_tsv_remaining,
    }


async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """SELECT id, content, importance, created_at
               FROM memories
               WHERE is_active = TRUE
               ORDER BY created_at DESC
               LIMIT $1""",
            limit,
        )


async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]


async def get_all_memories():
    """导出所有记忆（用于备份，含归档记录与三层结构字段）

    embedding/embedding_json 和 last_accessed 是可重算的派生数据与访问状态，不进备份。
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance, source_session, created_at,
                   layer, title, is_active, merged_from, event_date
            FROM memories ORDER BY id
        """)
        return [dict(r) for r in rows]


def _parse_backup_datetime(value):
    """解析备份里的时间字符串；解析不了返回 None（落库走默认 NOW()）"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt_timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=dt_timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt_timezone.utc)
        except ValueError:
            return None


def _parse_backup_date(value):
    """解析备份里的日期字符串（event_date 用），解析不了返回 None"""
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


async def import_memories_v2(memories: list):
    """恢复 v2 版本化备份：单事务两遍，先插入建映射，再回填合并关系。

    - 同内容且库中唯一 → 跳过并映射到已有行（同一份备份重复导入幂等）
    - 同内容但库中多行 → 不猜挂哪条，计入 conflicts 回执，引用它的合并链降级
    - merged_from 引用了备份中不存在的 backup_id → 备份损坏，抛错整批回滚
    - embedding 不随导入计算，恢复后由 backfill 重算
    """
    # ---- 事务外纯格式校验：先收集全部 backup_id，再验证引用封闭性 ----
    backup_ids = set()
    for mem in memories:
        if not isinstance(mem, dict):
            raise ValueError("记忆条目必须是 JSON 对象")
        bid = mem.get("backup_id")
        if isinstance(bid, bool) or not isinstance(bid, int):
            raise ValueError(f"backup_id 缺失或非法: {bid!r}")
        if bid in backup_ids:
            raise ValueError(f"backup_id 重复: {bid}")
        backup_ids.add(bid)
        if not isinstance(mem.get("content"), str) or not mem["content"].strip():
            raise ValueError(f"记忆 {bid} 缺少 content")
        if mem.get("layer", 1) not in (1, 2, 3):
            raise ValueError(f"记忆 {bid} 层级非法: {mem.get('layer')!r}")
    for mem in memories:
        for ref in (mem.get("merged_from") or []):
            if isinstance(ref, bool) or not isinstance(ref, int):
                raise ValueError(f"记忆 {mem['backup_id']} 的 merged_from 含非法引用: {ref!r}")
            if ref not in backup_ids:
                raise ValueError(
                    f"记忆 {mem['backup_id']} 的 merged_from 引用了备份中不存在的 {ref}，备份不完整"
                )

    pool = await get_pool()
    imported = 0
    skipped = 0
    conflicts = []
    degraded = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            # ---- 第一遍：插入并建立 旧 backup_id → 新库 id 映射 ----
            id_map = {}
            for mem in memories:
                bid = mem["backup_id"]
                content = mem["content"]
                rows = await conn.fetch(
                    "SELECT id FROM memories WHERE content = $1", content
                )
                if len(rows) == 1:
                    id_map[bid] = int(rows[0]["id"])
                    skipped += 1
                    continue
                if len(rows) > 1:
                    conflicts.append({
                        "backup_id": bid,
                        "matched_ids": sorted(int(r["id"]) for r in rows),
                    })
                    skipped += 1
                    continue
                row = await conn.fetchrow("""
                    INSERT INTO memories (content, importance, source_session, created_at,
                                          layer, title, is_active, event_date)
                    VALUES ($1, $2, $3, COALESCE($4, NOW()), $5, $6, $7, $8)
                    RETURNING id
                """,
                    content,
                    mem.get("importance", 5),
                    mem.get("source_session") or "json-import",
                    _parse_backup_datetime(mem.get("created_at")),
                    mem.get("layer", 1),
                    mem.get("title") or "",
                    bool(mem.get("is_active", True)),
                    _parse_backup_date(mem.get("event_date")),
                )
                id_map[bid] = int(row["id"])
                imported += 1

            # ---- 第二遍：用映射回填 merged_from ----
            for mem in memories:
                refs = mem.get("merged_from") or []
                if not refs:
                    continue
                bid = mem["backup_id"]
                new_id = id_map.get(bid)
                if new_id is None:
                    # 父条本身因内容冲突被跳过，没有落库行可回填
                    continue
                unresolved = [ref for ref in refs if ref not in id_map]
                if unresolved:
                    # 来源条目因冲突未建立映射：不猜关系，保持 NULL 并回执降级
                    degraded.append({"backup_id": bid, "unresolved": unresolved})
                    continue
                await conn.execute(
                    "UPDATE memories SET merged_from = $1 WHERE id = $2",
                    [id_map[ref] for ref in refs], new_id,
                )

    total = await get_all_memories_count()
    result = {
        "status": "done",
        "schema_version": 2,
        "imported": imported,
        "skipped": skipped,
        "conflicts": conflicts,
        "degraded": degraded,
        "total": total,
    }
    if MEMORY_VECTOR_ENABLED:
        try:
            result["pending_embeddings"] = await get_pending_memory_embedding_count()
        except Exception:
            pass
    return result


async def get_all_memories_detail(limit: int = None, layer: int = None, active_only: bool = None):
    """获取所有记忆（含 id，用于管理页面）
    
    Args:
        limit: 可选，限制返回数量
        layer: 可选，筛选指定层级（1=原始碎片, 2=事件记忆, 3=核心记忆）
        active_only: 可选，是否只返回 is_active=true 的记忆
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        conditions = []
        params = []
        param_idx = 1
        
        if layer is not None:
            conditions.append(f"layer = ${param_idx}")
            params.append(layer)
            param_idx += 1
        
        if active_only is not None:
            conditions.append(f"is_active = ${param_idx}")
            params.append(active_only)
            param_idx += 1
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        if limit is not None:
            limit_clause = f"LIMIT ${param_idx}"
            params.append(limit)
        else:
            limit_clause = ""
        
        rows = await conn.fetch(f"""
            SELECT id, content, importance, source_session, created_at,
                   layer, title, is_active, merged_from, event_date
            FROM memories
            {where_clause}
            ORDER BY id
            {limit_clause}
        """, *params)
        return [dict(r) for r in rows]


async def update_memory(memory_id: int, content: str = None, importance: int = None):
    """更新单条记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if content is not None and importance is not None:
            await conn.execute(
                "UPDATE memories SET content = $1, importance = $2 WHERE id = $3",
                content, importance, memory_id
            )
        elif content is not None:
            await conn.execute(
                "UPDATE memories SET content = $1 WHERE id = $2",
                content, memory_id
            )
        elif importance is not None:
            await conn.execute(
                "UPDATE memories SET importance = $1 WHERE id = $2",
                importance, memory_id
            )


async def delete_memory(memory_id: int):
    """删除单条记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)


async def delete_memories_batch(memory_ids: list):
    """批量删除记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memories WHERE id = ANY($1::int[])", memory_ids
        )


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


# ============================================================
# 对话历史读取（分区缓存用）
# ============================================================

async def get_conversation_messages(session_id: str, limit: int = 100):
    """按时间正序读取session的消息"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT role, content, metadata, created_at
            FROM conversations
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """, session_id, limit)
        return [dict(r) for r in rows]


# ============================================================
# 分区缓存状态管理
# ============================================================

def _active_seen_fragment_ids(seen_fragment_times, ttl_hours: float, now=None) -> list:
    """Return fragment IDs whose individual seen timestamps are still inside TTL."""
    if ttl_hours <= 0:
        return []
    if isinstance(seen_fragment_times, str):
        try:
            seen_fragment_times = json.loads(seen_fragment_times)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(seen_fragment_times, dict):
        return []

    now = now or datetime.now(dt_timezone.utc)
    cutoff = now - timedelta(hours=ttl_hours)
    active = []
    for fragment_id, raw_seen_at in seen_fragment_times.items():
        try:
            if isinstance(raw_seen_at, datetime):
                seen_at = raw_seen_at
            else:
                seen_at = datetime.fromisoformat(str(raw_seen_at).replace("Z", "+00:00"))
            if seen_at.tzinfo is None:
                seen_at = seen_at.replace(tzinfo=dt_timezone.utc)
            if seen_at >= cutoff:
                active.append(str(fragment_id))
        except (TypeError, ValueError):
            continue
    return sorted(set(active))


async def get_session_cache_state(session_id: str, seen_ttl_hours: float = None) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT summary, a_start_round, seen_fragment_ids,
                      seen_fragment_times, updated_at
               FROM session_cache_state WHERE session_id = $1""",
            session_id
        )
        if row:
            raw_summary = row['summary'] or ''
            summary_parts = []
            if raw_summary:
                try:
                    parsed = json.loads(raw_summary)
                    if isinstance(parsed, list):
                        summary_parts = parsed
                    else:
                        summary_parts = [raw_summary]
                except (json.JSONDecodeError, ValueError):
                    summary_parts = [raw_summary]
            raw_seen_times = row.get('seen_fragment_times') or {}
            if isinstance(raw_seen_times, str):
                try:
                    raw_seen_times = json.loads(raw_seen_times)
                except (json.JSONDecodeError, ValueError):
                    raw_seen_times = {}
            if not raw_seen_times and row['seen_fragment_ids']:
                legacy_seen_at = row['updated_at'] or datetime.now(dt_timezone.utc)
                raw_seen_times = {
                    str(fragment_id): legacy_seen_at.isoformat()
                    for fragment_id in row['seen_fragment_ids']
                }
            seen_fragment_ids = (
                _active_seen_fragment_ids(raw_seen_times, seen_ttl_hours)
                if seen_ttl_hours is not None
                else sorted(str(fragment_id) for fragment_id in raw_seen_times)
            )
            return {
                'summary_parts': summary_parts,
                'a_start_round': row['a_start_round'] or 0,
                'seen_fragment_ids': seen_fragment_ids,
                'updated_at': row['updated_at'],
            }
        return {
            'summary_parts': [],
            'a_start_round': 0,
            'seen_fragment_ids': [],
            'updated_at': None,
        }


async def save_session_cache_state(session_id: str, summary_parts: list, a_start_round: int):
    import json
    summary_json = json.dumps(summary_parts, ensure_ascii=False)
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO session_cache_state (session_id, summary, a_start_round, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (session_id) 
            DO UPDATE SET summary = $2, a_start_round = $3, updated_at = NOW()
        """, session_id, summary_json, a_start_round)


async def mark_fragments_seen(session_id: str, fragment_ids: list, ttl_hours: float = 6):
    """成功请求结束后原子合并已注入 fragment_id，不覆盖分区摘要状态。"""
    ids = sorted({str(value) for value in fragment_ids if value})
    ttl_hours = float(ttl_hours)
    if not session_id or not ids or ttl_hours <= 0:
        return 0
    seen_at = datetime.now(dt_timezone.utc).isoformat()
    fresh_seen = json.dumps(
        {fragment_id: seen_at for fragment_id in ids},
        ensure_ascii=False,
    )
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO session_cache_state (
                   session_id, seen_fragment_times, updated_at
               ) VALUES ($1, $2::jsonb, NOW())
               ON CONFLICT (session_id) DO UPDATE
               SET seen_fragment_times = (
                       SELECT COALESCE(
                           jsonb_object_agg(active.key, active.value),
                           '{}'::jsonb
                       )
                       FROM jsonb_each(
                           COALESCE(
                               session_cache_state.seen_fragment_times,
                               '{}'::jsonb
                           )
                       ) AS active(key, value)
                       WHERE (active.value #>> '{}')::timestamptz >=
                             NOW() - ($3::double precision * INTERVAL '1 hour')
                   ) || EXCLUDED.seen_fragment_times,
                   updated_at = NOW()""",
            session_id, fresh_seen, ttl_hours,
        )
    return len(ids)


# ============================================================
# Token 使用记录
# ============================================================

async def ensure_token_usage_table():
    """确保token_usage表存在（在init_tables里调用）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT,
                model           TEXT,
                prompt_tokens   INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                usage_type      TEXT DEFAULT 'chat',
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_created ON token_usage (created_at DESC);
        """)


async def save_token_usage(session_id: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, usage_type: str = "chat"):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO token_usage (session_id, model, prompt_tokens, completion_tokens, total_tokens, usage_type)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, session_id, model, prompt_tokens, completion_tokens, total_tokens, usage_type)


# ============================================================
# 对话记录管理
# ============================================================

async def get_conversations_paginated(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    pool = await get_pool()
    async with pool.acquire() as conn:
        total_row = await conn.fetchrow(
            "SELECT COUNT(DISTINCT session_id) as total FROM conversations"
        )
        total = total_row['total'] if total_row else 0

        rows = await conn.fetch("""
            WITH session_info AS (
                SELECT session_id, MIN(created_at) as first_time, MAX(created_at) as last_time, COUNT(*) as message_count
                FROM conversations GROUP BY session_id ORDER BY last_time DESC LIMIT $1 OFFSET $2
            )
            SELECT si.*,
                   COALESCE(tu.total_all, 0) as total_tokens
            FROM session_info si
            LEFT JOIN (
                SELECT session_id, SUM(total_tokens) as total_all FROM token_usage WHERE usage_type = 'chat' GROUP BY session_id
            ) tu ON si.session_id = tu.session_id
            ORDER BY si.last_time DESC
        """, per_page, offset)
        
        results = []
        for r in rows:
            preview_row = await conn.fetchrow(
                "SELECT content FROM conversations WHERE session_id = $1 AND role = 'user' ORDER BY created_at LIMIT 1",
                r['session_id']
            )
            preview = preview_row['content'][:80] if preview_row else ''
            title = (preview[:30] + '...' if len(preview) > 30 else preview) or r['session_id']
            results.append({
                'session_id': r['session_id'],
                'title': title,
                'first_time': r['first_time'].isoformat() if r['first_time'] else None,
                'last_time': r['last_time'].isoformat() if r['last_time'] else None,
                'message_count': r['message_count'],
                'preview': preview,
                'total_tokens': r['total_tokens'],
            })
        return results, total


async def delete_conversation(session_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = $1", session_id)


async def batch_delete_conversations(session_ids: list):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE session_id = ANY($1)", session_ids)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = ANY($1)", session_ids)


async def merge_sessions_to_target(source_ids: list, target_id: str) -> dict:
    if not source_ids:
        return {'merged_sessions': 0, 'merged_messages': 0, 'merged_token_records': 0}
    pool = await get_pool()
    async with pool.acquire() as conn:
        msg_count = await conn.fetchval("SELECT COUNT(*) FROM conversations WHERE session_id = ANY($1)", source_ids)
        await conn.execute("UPDATE conversations SET session_id = $1 WHERE session_id = ANY($2)", target_id, source_ids)
        token_count = await conn.fetchval("SELECT COUNT(*) FROM token_usage WHERE session_id = ANY($1)", source_ids)
        await conn.execute("UPDATE token_usage SET session_id = $1 WHERE session_id = ANY($2)", target_id, source_ids)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = ANY($1)", source_ids)
        return {'merged_sessions': len(source_ids), 'merged_messages': msg_count or 0, 'merged_token_records': token_count or 0}


async def list_all_session_cache_states() -> list:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT scs.session_id, scs.summary, scs.a_start_round, scs.updated_at,
                   COALESCE(c.message_count, 0) as message_count,
                   COALESCE(tu.chat_tokens, 0) as chat_tokens
            FROM session_cache_state scs
            LEFT JOIN (SELECT session_id, COUNT(*) as message_count FROM conversations GROUP BY session_id) c ON scs.session_id = c.session_id
            LEFT JOIN (SELECT session_id, SUM(total_tokens) as chat_tokens FROM token_usage WHERE usage_type = 'chat' GROUP BY session_id) tu ON scs.session_id = tu.session_id
            ORDER BY scs.updated_at DESC
        """)
        results = []
        for r in rows:
            raw_summary = r['summary'] or ''
            try:
                import json
                parsed = json.loads(raw_summary)
                if isinstance(parsed, list):
                    summary_parts = parsed
                else:
                    summary_parts = [raw_summary] if raw_summary else []
            except (json.JSONDecodeError, ValueError):
                summary_parts = [raw_summary] if raw_summary else []
            results.append({
                'session_id': r['session_id'],
                'summary': '\n\n'.join(summary_parts),
                'summary_length': sum(len(p) for p in summary_parts),
                'summary_count': len(summary_parts),
                'a_start_round': r['a_start_round'],
                'updated_at': r['updated_at'].isoformat() if r['updated_at'] else None,
                'message_count': r['message_count'],
                'chat_tokens': r['chat_tokens'],
            })
        return results


async def delete_session_cache_state(session_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = $1", session_id)


async def rename_session_id(old_id: str, new_id: str) -> bool:
    """重命名对话线ID（事务内同时修改三个表）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 检查新ID是否已存在
            exists = await conn.fetchval(
                "SELECT 1 FROM session_cache_state WHERE session_id = $1", new_id
            )
            if exists:
                return False
            # session_cache_state
            await conn.execute(
                "UPDATE session_cache_state SET session_id = $1 WHERE session_id = $2",
                new_id, old_id
            )
            # conversations
            await conn.execute(
                "UPDATE conversations SET session_id = $1 WHERE session_id = $2",
                new_id, old_id
            )
            # token_usage
            await conn.execute(
                "UPDATE token_usage SET session_id = $1 WHERE session_id = $2",
                new_id, old_id
            )
            return True


def db_row_to_message(row: dict) -> dict:
    """
    把DB记录还原成API消息格式。
    
    普通消息: {"role": "user", "content": "你好"} 
    工具调用: {"role": "assistant", "content": null, "tool_calls": [...]}
    工具结果: {"role": "tool", "content": "结果", "tool_call_id": "call_xxx"}
    思维链:   {"role": "assistant", "content": "回答", "reasoning_content": "思维链"}
    """
    import json as _json
    msg = {"role": row["role"], "content": row.get("content") or ""}
    
    meta_str = row.get("metadata")
    if meta_str:
        try:
            meta = _json.loads(meta_str)
            # assistant 带 tool_calls
            if "tool_calls" in meta:
                msg["tool_calls"] = meta["tool_calls"]
                if not row.get("content"):
                    msg["content"] = None
            # assistant 带 reasoning_content（deepseek thinking mode）
            if "reasoning_content" in meta:
                msg["reasoning_content"] = meta["reasoning_content"]
            # tool 消息带 tool_call_id
            if "tool_call_id" in meta:
                msg["tool_call_id"] = meta["tool_call_id"]
            # 其他可能的字段（name 等）
            if "name" in meta:
                msg["name"] = meta["name"]
        except Exception:
            pass
    
    return msg


async def export_all_conversations():
    """导出所有对话记录（用于备份）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT session_id, role, content, model, created_at
            FROM conversations
            ORDER BY session_id, created_at
        """)
        return [
            {
                'session_id': r['session_id'],
                'role': r['role'],
                'content': r['content'],
                'model': r['model'] or '',
                'created_at': r['created_at'].isoformat() if r['created_at'] else None,
            }
            for r in rows
        ]


async def import_conversations(records: list):
    """
    导入对话记录（自动去重）
    
    records: [{ session_id, role, content, model?, created_at? }, ...]
    按 session_id + role + created_at 三元组去重，已存在的跳过。
    返回 (导入数量, 跳过数量)
    """
    if not records:
        return 0, 0
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        imported = 0
        skipped = 0
        for r in records:
            session_id = r.get('session_id')
            role = r.get('role')
            content = r.get('content')
            
            if not all([session_id, role, content]):
                continue
            
            model = r.get('model', '')
            created_at = r.get('created_at')
            tsv_text = (
                jieba_tokenize_for_tsv(content)
                if CONVERSATION_RECALL_ENABLED
                else None
            )
            
            # 解析时间
            from datetime import datetime
            if created_at and isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    created_at = None
            
            # 去重检查
            if created_at:
                existing = await conn.fetchrow("""
                    SELECT id FROM conversations
                    WHERE session_id = $1 AND role = $2 AND created_at = $3
                    LIMIT 1
                """, session_id, role, created_at)
                
                if existing:
                    skipped += 1
                    continue
                
                await conn.execute("""
                    INSERT INTO conversations (
                        session_id, role, content, model, created_at, content_tsv
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        array_to_tsvector(string_to_array($6, ' '))
                    )
                """, session_id, role, content, model, created_at, tsv_text)
            else:
                await conn.execute("""
                    INSERT INTO conversations (
                        session_id, role, content, model, content_tsv
                    ) VALUES (
                        $1, $2, $3, $4,
                        array_to_tsvector(string_to_array($5, ' '))
                    )
                """, session_id, role, content, model, tsv_text)
            
            imported += 1
        
        if skipped:
            print(f"📥 导入对话: {imported} 条新增, {skipped} 条已存在跳过")
        else:
            print(f"📥 导入对话: {imported} 条新增")
        
        if (
            imported
            and CONVERSATION_RECALL_ENABLED
            and EMBEDDING_API_KEY
        ):
            kick_embedding_backfill()
        return imported, skipped


# ============================================================
# 三层记忆架构（碎片/事件/核心）
# ============================================================

async def get_fragments_by_date(event_date):
    """获取指定日期的原始碎片（用于每日整理）"""
    # 把本地日期转成UTC时间范围，避免DATE()用UTC截断导致日期偏移
    local_tz = dt_timezone(timedelta(hours=TIMEZONE_HOURS))
    start_utc = datetime(event_date.year, event_date.month, event_date.day, tzinfo=local_tz).astimezone(dt_timezone.utc)
    end_utc = start_utc + timedelta(days=1)
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance, created_at
            FROM memories
            WHERE layer = 1 AND is_active = TRUE
            AND created_at >= $1 AND created_at < $2
            ORDER BY created_at
        """, start_utc, end_utc)
        return [dict(r) for r in rows]


async def get_fragments_by_date_range(start_date, end_date):
    """获取指定时间段的原始碎片（用于跨天整理）"""
    # 把本地日期转成UTC时间范围，避免DATE()用UTC截断导致日期偏移
    local_tz = dt_timezone(timedelta(hours=TIMEZONE_HOURS))
    start_utc = datetime(start_date.year, start_date.month, start_date.day, tzinfo=local_tz).astimezone(dt_timezone.utc)
    # end_date 当天结束 = end_date 下一天的 00:00
    end_utc = datetime(end_date.year, end_date.month, end_date.day, tzinfo=local_tz).astimezone(dt_timezone.utc) + timedelta(days=1)
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance, created_at
            FROM memories
            WHERE layer = 1 AND is_active = TRUE
            AND created_at >= $1 AND created_at < $2
            ORDER BY created_at
        """, start_utc, end_utc)
        return [dict(r) for r in rows]


async def create_event_memory(title: str, content: str, importance: int, 
                               event_date, merged_from: list):
    """创建事件记忆（从碎片合并而来）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO memories (content, importance, layer, title, is_active, merged_from, event_date)
            VALUES ($1, $2, 2, $3, TRUE, $4, $5)
            RETURNING id
        """, content, importance, title, merged_from, event_date)
        
        new_id = row['id'] if row else None
        
        # 向量搜索：计算并保存 embedding
        if MEMORY_VECTOR_ENABLED and new_id:
            try:
                embedding = await compute_embedding(content)
                if embedding:
                    await save_memory_embedding(conn, new_id, embedding)
            except Exception as e:
                print(f"⚠️ 事件记忆embedding计算失败（id={new_id}）: {e}")
        
        return new_id


async def deactivate_memories(memory_ids: list):
    """将记忆标记为不活跃（合并后的碎片）"""
    if not memory_ids:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE memories SET is_active = FALSE
            WHERE id = ANY($1::int[])
        """, memory_ids)


async def create_consolidated_events(events: list, expected_fragment_ids: list):
    """原子地创建整理事件并归档被完整覆盖的来源碎片。

    模型结果必须完整且唯一地覆盖 expected_fragment_ids。事务开始后再次锁定并
    验证所有来源仍是活跃碎片，避免并发整理或手动操作造成部分提交。
    """
    expected_ids = [int(memory_id) for memory_id in expected_fragment_ids]
    expected_set = set(expected_ids)
    if not expected_ids:
        return []
    if len(expected_ids) != len(expected_set):
        raise ValueError("expected_fragment_ids 存在重复")

    merged_ids = []
    seen_ids = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("整理事件必须是JSON对象")
        if not isinstance(event.get("content"), str) or not event["content"].strip():
            raise ValueError("整理事件缺少 content")
        event_ids = event.get("merged_ids", [])
        if not event_ids:
            raise ValueError("整理事件缺少 merged_ids")
        for memory_id in event_ids:
            if isinstance(memory_id, bool) or not isinstance(memory_id, int):
                raise ValueError(f"整理事件包含非法碎片ID: {memory_id}")
            if memory_id not in expected_set:
                raise ValueError(f"整理事件引用了范围外碎片: {memory_id}")
            if memory_id in seen_ids:
                raise ValueError(f"碎片被多个事件重复引用: {memory_id}")
            seen_ids.add(memory_id)
            merged_ids.append(memory_id)

    missing_ids = expected_set - seen_ids
    if missing_ids:
        raise ValueError(f"整理事件未覆盖全部碎片: {sorted(missing_ids)}")

    pool = await get_pool()
    created = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch("""
                SELECT id
                FROM memories
                WHERE id = ANY($1::int[])
                  AND layer = 1
                  AND is_active = TRUE
                FOR UPDATE
            """, expected_ids)
            active_ids = {int(row["id"]) for row in rows}
            if active_ids != expected_set:
                unavailable = sorted(expected_set - active_ids)
                raise RuntimeError(f"部分来源碎片已被其他操作修改: {unavailable}")

            for event in events:
                row = await conn.fetchrow("""
                    INSERT INTO memories (
                        content, importance, layer, title,
                        is_active, merged_from, event_date
                    )
                    VALUES ($1, $2, 2, $3, TRUE, $4, $5)
                    RETURNING id
                """,
                    event.get("content", ""),
                    event.get("importance", 5),
                    event.get("title", ""),
                    event["merged_ids"],
                    event.get("event_date"),
                )
                if not row:
                    raise RuntimeError("创建事件记忆失败")
                created.append({
                    "id": int(row["id"]),
                    "content": event.get("content", ""),
                })

            result = await conn.execute("""
                UPDATE memories
                SET is_active = FALSE
                WHERE id = ANY($1::int[])
                  AND layer = 1
                  AND is_active = TRUE
            """, merged_ids)
            updated = int(result.split()[-1]) if result else 0
            if updated != len(expected_ids):
                raise RuntimeError(
                    f"归档来源碎片数量不符: expected={len(expected_ids)}, updated={updated}"
                )

    # embedding 失败不影响事件与来源碎片的原子提交，和旧逻辑保持一致。
    if MEMORY_VECTOR_ENABLED and created:
        async with pool.acquire() as conn:
            for event in created:
                try:
                    embedding = await compute_embedding(event["content"])
                    if embedding:
                        await save_memory_embedding(conn, event["id"], embedding)
                except Exception as exc:
                    print(f"⚠️ 事件记忆embedding计算失败（id={event['id']}）: {exc}")

    return [event["id"] for event in created]


async def promote_to_core(memory_id: int, title: str = None):
    """将记忆升级为核心记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if title:
            await conn.execute("""
                UPDATE memories SET layer = 3, title = $2
                WHERE id = $1
            """, memory_id, title)
        else:
            await conn.execute("""
                UPDATE memories SET layer = 3
                WHERE id = $1
            """, memory_id)


async def merge_memories(memory_ids: list, new_title: str, new_content: str, 
                         importance: int, layer: int = 2):
    """合并多条记忆为一条新记忆"""
    if not memory_ids:
        return None
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 获取原记忆的日期（取最早的；来源含事件记忆时优先其真实发生日，而非整理日）
        rows = await conn.fetch("""
            SELECT MIN(COALESCE(event_date, DATE(created_at))) as event_date
            FROM memories WHERE id = ANY($1::int[])
        """, memory_ids)
        event_date = rows[0]['event_date'] if rows else None
        
        # 创建新记忆
        row = await conn.fetchrow("""
            INSERT INTO memories (content, importance, layer, title, is_active, merged_from, event_date)
            VALUES ($1, $2, $3, $4, TRUE, $5, $6)
            RETURNING id
        """, new_content, importance, layer, new_title, memory_ids, event_date)
        
        new_id = row['id'] if row else None
        
        # 向量搜索：计算并保存 embedding
        if MEMORY_VECTOR_ENABLED and new_id:
            try:
                embedding = await compute_embedding(new_content)
                if embedding:
                    await save_memory_embedding(conn, new_id, embedding)
            except Exception as e:
                print(f"⚠️ 合并记忆embedding计算失败（id={new_id}）: {e}")
        
        # 将原记忆标记为不活跃
        if new_id:
            await deactivate_memories(memory_ids)
        
        return new_id


async def check_duplicate_memory(new_content: str, threshold: float = 0.7) -> dict:
    """检查新记忆是否与现有记忆重复
    
    三层去重策略：
    1. 精确匹配：内容完全相同
    2. 包含关系：新内容包含旧内容，或旧内容包含新内容
    3. 关键词重叠度：Jaccard 相似度 > threshold
    
    Returns:
        {
            "is_duplicate": bool,
            "reason": str,  # "exact" / "containment" / "similarity"
            "matched_id": int or None,
            "similarity": float or None
        }
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 获取所有活跃记忆
        rows = await conn.fetch("""
            SELECT id, content FROM memories 
            WHERE is_active = TRUE
        """)
        
        new_content_lower = new_content.strip().lower()
        new_keywords = set(extract_search_keywords(new_content))
        
        for row in rows:
            old_content = row['content']
            old_content_lower = old_content.strip().lower()
            
            # 第一层：精确匹配
            if new_content_lower == old_content_lower:
                return {
                    "is_duplicate": True,
                    "reason": "exact",
                    "matched_id": row['id'],
                    "similarity": 1.0
                }
            
            # 第二层：包含关系
            if new_content_lower in old_content_lower:
                return {
                    "is_duplicate": True,
                    "reason": "containment",
                    "matched_id": row['id'],
                    "similarity": len(new_content) / len(old_content)
                }
            if old_content_lower in new_content_lower:
                return {
                    "is_duplicate": True,
                    "reason": "containment_update",
                    "matched_id": row['id'],
                    "similarity": len(old_content) / len(new_content)
                }
            
            # 第三层：关键词重叠度（Jaccard 相似度）
            old_keywords = set(extract_search_keywords(old_content))
            if new_keywords and old_keywords:
                intersection = new_keywords & old_keywords
                union = new_keywords | old_keywords
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity > threshold:
                    return {
                        "is_duplicate": True,
                        "reason": "similarity",
                        "matched_id": row['id'],
                        "similarity": similarity
                    }
        
        return {
            "is_duplicate": False,
            "reason": None,
            "matched_id": None,
            "similarity": None
        }


async def update_memory_with_layer(memory_id: int, content: str = None, 
                                    importance: int = None, title: str = None,
                                    layer: int = None, is_active: bool = None):
    """更新记忆（支持三层架构新字段）"""
    updates = []
    params = []
    param_idx = 2  # $1 给 memory_id
    
    if content is not None:
        updates.append(f"content = ${param_idx}")
        params.append(content)
        param_idx += 1
    
    if importance is not None:
        updates.append(f"importance = ${param_idx}")
        params.append(importance)
        param_idx += 1
    
    if title is not None:
        updates.append(f"title = ${param_idx}")
        params.append(title)
        param_idx += 1
    
    if layer is not None:
        updates.append(f"layer = ${param_idx}")
        params.append(layer)
        param_idx += 1
    
    if is_active is not None:
        updates.append(f"is_active = ${param_idx}")
        params.append(is_active)
        param_idx += 1
    
    if not updates:
        return
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = $1",
            memory_id, *params
        )


async def get_layer_statistics():
    """获取各层记忆的统计数据"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                layer,
                COUNT(*) as count,
                COUNT(*) FILTER (WHERE is_active = TRUE) as active_count
            FROM memories
            GROUP BY layer
            ORDER BY layer
        """)
        
        stats = {
            "layer_1": {"total": 0, "active": 0},  # 原始碎片
            "layer_2": {"total": 0, "active": 0},  # 事件记忆
            "layer_3": {"total": 0, "active": 0},  # 核心记忆
        }
        
        for row in rows:
            layer = row['layer'] or 1  # 默认为层级1
            key = f"layer_{layer}"
            if key in stats:
                stats[key] = {
                    "total": row['count'],
                    "active": row['active_count']
                }
        
        return stats


async def cleanup_old_fragments(days: int = 30):
    """清理指定天数前的归档碎片
    
    只清理满足以下条件的记忆：
    - layer = 1（原始碎片）
    - is_active = FALSE（已归档）
    - created_at 在 days 天之前
    
    Returns:
        删除的记忆数量
    """
    from datetime import datetime, timedelta
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        result = await conn.execute("""
            DELETE FROM memories
            WHERE layer = 1
            AND is_active = FALSE
            AND created_at < $1
        """, cutoff_date)
        
        # 解析删除数量，格式如 "DELETE 5"
        deleted = int(result.split()[-1]) if result else 0
        return deleted


async def revert_merge(memory_id: int):
    """撤回合并操作
    
    恢复原始碎片（is_active = TRUE），删除合并后的事件记忆
    
    Args:
        memory_id: 要撤回的事件记忆ID
        
    Returns:
        {"status": "ok", "restored": 恢复的碎片数量}
        或 {"error": "错误信息"}
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 获取事件记忆信息
        row = await conn.fetchrow("""
            SELECT id, layer, merged_from FROM memories WHERE id = $1
        """, memory_id)
        
        if not row:
            return {"error": "记忆不存在"}
        
        if row['layer'] != 2:
            return {"error": "只能撤回事件记忆的合并"}
        
        merged_from = row['merged_from']
        if not merged_from or len(merged_from) == 0:
            return {"error": "没有合并来源，无法撤回"}
        
        # 恢复原始碎片
        result = await conn.execute("""
            UPDATE memories SET is_active = TRUE
            WHERE id = ANY($1::int[])
        """, merged_from)
        restored = int(result.split()[-1]) if result else 0
        
        # 删除事件记忆
        await conn.execute("""
            DELETE FROM memories WHERE id = $1
        """, memory_id)
        
        return {"status": "ok", "restored": restored}
