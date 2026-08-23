"""Tokenization, embeddings, recall search, and search backfill."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import List

from db import core as db_core

logger = logging.getLogger(__name__)

# ============================================================
# 中文分词工具（基于 jieba）
# ============================================================

import jieba
import jieba.analyse

import shared

# 静默加载词典
jieba.setLogLevel(jieba.logging.INFO)

EN_WORD_PATTERN = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')
NUM_PATTERN = re.compile(r'\d{2,}')
# 清理查询开头的时间戳（如 "2026-05-02 20:26"）
TIMESTAMP_PATTERN = re.compile(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*\d{1,2}:\d{1,2}\s*')
_CONVERSATION_CANDIDATE_POOL = 20

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
    if not shared.EMBEDDING_API_KEY:
        return []

    try:
        import httpx

        if len(text) > 4000:
            text = text[:4000]

        body = {
            "model": shared.EMBEDDING_MODEL,
            "input": text,
        }
        if shared.EMBEDDING_DIM > 0:
            body["dimensions"] = shared.EMBEDDING_DIM

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{shared.EMBEDDING_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {shared.EMBEDDING_API_KEY}",
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

    if not shared.EMBEDDING_API_KEY:
        return []
    normalized_query = query.strip()
    if not normalized_query:
        return []

    key = (
        normalized_query,
        shared.EMBEDDING_BASE_URL.rstrip("/"),
        shared.EMBEDDING_MODEL,
        shared.EMBEDDING_DIM,
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

    if db_core.HAS_PGVECTOR:
        vec_str = '[' + ','.join(str(f) for f in embedding) + ']'
        await conn.execute(
            "UPDATE memories SET embedding = $1::vector WHERE id = $2",
            vec_str, memory_id
        )
    else:
        await conn.execute(
            "UPDATE memories SET embedding_json = $1 WHERE id = $2",
            json.dumps(embedding), memory_id
        )


async def save_conversation_embedding(conn, message_id: int, embedding: list):
    """保存单条原始对话向量。"""
    if not embedding:
        return
    if db_core.HAS_PGVECTOR:
        vector_text = "[" + ",".join(str(value) for value in embedding) + "]"
        await conn.execute(
            "UPDATE conversations SET embedding = $1::vector WHERE id = $2",
            vector_text,
            message_id,
        )
    else:
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

    if db_core.HAS_PGVECTOR:
        vector_text = "[" + ",".join(str(value) for value in query_embedding) + "]"
        params = [vector_text]
        exclusion_sql, exclusion_params = _session_exclusion_sql(
            exclude_session_ids, len(params) + 1
        )
        params.extend(exclusion_params)
        ranked_limit = max(100, pool_size * 10)
        params.extend([ranked_limit, shared.CONVERSATION_MIN_SCORE_THRESHOLD, pool_size])
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
        if similarity < shared.CONVERSATION_MIN_SCORE_THRESHOLD:
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

    if not shared.CONVERSATION_RECALL_ENABLED:
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
        if mode == "hybrid" and shared.EMBEDDING_API_KEY
        else []
    )
    pool_size = max(_CONVERSATION_CANDIDATE_POOL, max_sessions * 3)
    pool = await db_core.get_pool()
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
                shared.CONVERSATION_HW_KEYWORD * keyword_normalized.get(session_id, 0.0)
                + shared.CONVERSATION_HW_SEMANTIC * semantic_normalized.get(session_id, 0.0)
                + shared.CONVERSATION_HW_RECENCY * recency_normalized.get(session_id, 0.0)
            )
            for session_id in session_ids
        }
    ranked = sorted(final_scores.items(), key=lambda item: -item[1])

    vector_text = None
    if db_core.HAS_PGVECTOR and query_embedding:
        vector_text = "[" + ",".join(str(value) for value in query_embedding) + "]"
    results = []
    async with pool.acquire() as conn:
        for session_id, final_score in ranked:
            if len(results) >= max_sessions:
                break
            if db_core.HAS_PGVECTOR and vector_text:
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
                embedding_column = "embedding_json" if not db_core.HAS_PGVECTOR else "NULL::text"
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
                if db_core.HAS_PGVECTOR and vector_text:
                    semantic_similarity = float(message["sem_sim"] or 0)
                elif query_embedding and message["embedding_json"]:
                    try:
                        semantic_similarity = _cosine_sim(
                            query_embedding, json.loads(message["embedding_json"])
                        )
                    except Exception:
                        semantic_similarity = 0.0
                semantic_match = (
                    mode == "hybrid"
                    and semantic_similarity >= shared.CONVERSATION_MIN_SCORE_THRESHOLD
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
# 对话检索索引与向量持续补算
# ============================================================

async def rebuild_content_tsv(batch_size: int = 200):
    """以 content_tsv IS NULL 为持久账本，分批补齐关键词索引。"""
    if not shared.CONVERSATION_RECALL_ENABLED:
        return 0
    pool = await db_core.get_pool()
    total_updated = 0
    last_id = 0
    while True:
        if not shared.CONVERSATION_RECALL_ENABLED:
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
        if not shared.CONVERSATION_RECALL_ENABLED:
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
    column = "embedding" if db_core.HAS_PGVECTOR else "embedding_json"
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
        if not shared.CONVERSATION_RECALL_ENABLED:
            state["stopped_reason"] = "recall_disabled"
            return 0, 0, None
        if not shared.EMBEDDING_API_KEY:
            state["last_error"] = "EMBEDDING_API_KEY未设置"
            state["stopped_reason"] = "no_api_key"
            return 0, 0, state["last_error"]

        pool = await db_core.get_pool()
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
                if not shared.CONVERSATION_RECALL_ENABLED:
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

    if not shared.CONVERSATION_RECALL_ENABLED or not shared.EMBEDDING_API_KEY:
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
        pool = await db_core.get_pool()
        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                f"SELECT COUNT(*) FROM conversations WHERE {_conversation_embedding_pending_condition()}"
            )
            embedding_column = "embedding" if db_core.HAS_PGVECTOR else "embedding_json"
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
        "enabled": shared.CONVERSATION_RECALL_ENABLED,
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
