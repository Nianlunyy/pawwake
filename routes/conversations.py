"""Conversation management and recall routes."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import shared
from db import core as db_core
from db import search as db_search
from db import conversations as db_conversations

router = APIRouter()

# ============================================================
# 对话记录管理 API
# ============================================================

@router.get("/api/conversations")
async def api_conversations(page: int = 1, per_page: int = 20):
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        results, total = await db_conversations.get_conversations_paginated(page, per_page)
        total_pages = max(1, -(-total // per_page))  # 向上取整
        return {"conversations": results, "total": total, "page": page, "per_page": per_page, "total_pages": total_pages}
    except Exception:
        return shared._api_failure("加载对话失败")


@router.get("/api/conversations/{session_id}/messages")
async def api_conversation_messages(session_id: str, limit: int = 50, offset: int = 0):
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        pool = await db_core.get_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM conversations WHERE session_id = $1", session_id
            )
            rows = await conn.fetch("""
                SELECT id, role, content, created_at
                FROM conversations WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """, session_id, limit, offset)
        msgs = [{"id": r["id"], "role": r["role"], "content": r["content"],
                 "created_at": r["created_at"].isoformat() if r.get("created_at") else None} for r in rows]
        return {"messages": msgs, "total": total}
    except Exception:
        return shared._api_failure("加载消息失败")


@router.delete("/api/conversations/{session_id}")
async def api_delete_conversation(session_id: str):
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        await db_conversations.delete_conversation(session_id)
        return {"status": "ok"}
    except Exception:
        return shared._api_failure("删除对话失败")


@router.post("/api/conversations/batch-delete")
async def api_batch_delete_conversations(request: Request):
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        body = await request.json()
        ids = body.get("session_ids", [])
        if ids:
            await db_conversations.batch_delete_conversations(ids)
        return {"status": "ok", "deleted": len(ids)}
    except Exception:
        return shared._api_failure("批量删除对话失败")


@router.post("/api/admin/merge-sessions")
async def api_merge_sessions(request: Request):
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        body = await request.json()
        source_ids = [s for s in body.get("source_ids", []) if s != body.get("target_id", "")]
        target_id = body.get("target_id", "")
        if not source_ids or not target_id:
            return {"error": "source_ids 和 target_id 不能为空"}
        result = await db_conversations.merge_sessions_to_target(source_ids, target_id)
        return {"status": "ok", **result}
    except Exception:
        return shared._api_failure("合并对话失败")


@router.get("/api/chat/search")
async def api_search_conversations(q: str = "", limit: int = 20, offset: int = 0):
    """搜索对话内容"""
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    if not q.strip():
        return {"error": "搜索关键词不能为空", "results": [], "total": 0}
    try:
        results, total = await db_conversations.search_conversations(q.strip(), limit, offset)
        return {"results": results, "total": total}
    except Exception:
        return shared._api_failure("搜索对话失败", results=[], total=0)


def _bounded_int(value, default: int, lower: int, upper: int) -> int:
    try:
        return min(upper, max(lower, int(value)))
    except (TypeError, ValueError):
        return default


async def _run_fragment_search(
    query: str,
    max_sessions,
    max_matches,
    context,
    mode: str,
    exclude_session_ids: list,
    exclude_fragment_ids: list,
):
    query = query.strip()
    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": "搜索关键词不能为空", "results": [], "total_sessions": 0},
        )
    if not shared.CONVERSATION_RECALL_ENABLED:
        return JSONResponse(
            status_code=409,
            content={"error": "对话召回未启用", "results": [], "total_sessions": 0},
        )
    mode = mode if mode in {"keyword", "hybrid"} else "hybrid"
    default_limit = max(1, shared.MAX_CONVERSATIONS_INJECT)
    max_sessions = _bounded_int(max_sessions, default_limit, 1, 50)
    max_matches = _bounded_int(max_matches, 1, 1, 5)
    context = _bounded_int(context, 1, 0, 5)
    try:
        results, total_sessions = await db_search.search_chat_fragments(
            query,
            max_sessions=max_sessions,
            max_matches_per_session=max_matches,
            context=context,
            mode=mode,
            exclude_session_ids=exclude_session_ids,
            exclude_fragment_ids=exclude_fragment_ids,
        )
        return {
            "results": results,
            "total_sessions": total_sessions,
            "query": query,
            "mode": mode,
        }
    except Exception:
        return JSONResponse(
            status_code=500,
            content=shared._api_failure("搜索对话片段失败", results=[], total_sessions=0),
        )


@router.get("/api/chat/search-fragments")
async def api_chat_search_fragments(
    q: str = "",
    max_sessions: int = None,
    max_matches: int = 1,
    context: int = 1,
    mode: str = "hybrid",
    exclude_session_ids: str = "",
    exclude_fragment_ids: str = "",
):
    """无状态 raw 召回；排除项用逗号分隔，敏感查询优先使用 POST。"""
    return await _run_fragment_search(
        q,
        max_sessions,
        max_matches,
        context,
        mode,
        [value for value in exclude_session_ids.split(",") if value],
        [value for value in exclude_fragment_ids.split(",") if value],
    )


@router.post("/api/chat/search-fragments")
async def api_chat_search_fragments_post(request: Request):
    """POST 变体避免查询原文进入 URL 与浏览器历史。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON"})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "请求体必须是 JSON 对象"})

    exclude_session_ids = body.get("exclude_session_ids", [])
    exclude_fragment_ids = body.get("exclude_fragment_ids", [])
    if not isinstance(exclude_session_ids, list) or not isinstance(exclude_fragment_ids, list):
        return JSONResponse(
            status_code=400,
            content={"error": "exclude_session_ids 和 exclude_fragment_ids 必须是数组"},
        )
    return await _run_fragment_search(
        body.get("q", "") if isinstance(body.get("q", ""), str) else "",
        body.get("max_sessions"),
        body.get("max_matches", 1),
        body.get("context", 1),
        body.get("mode", "hybrid") if isinstance(body.get("mode", ""), str) else "hybrid",
        [str(value) for value in exclude_session_ids if value],
        [str(value) for value in exclude_fragment_ids if value],
    )


@router.post("/api/admin/rebuild-conversation-search")
async def api_rebuild_conversation_search():
    if not shared.CONVERSATION_RECALL_ENABLED:
        return JSONResponse(status_code=409, content={"error": "对话召回未启用"})
    try:
        updated_tsv = await db_search.rebuild_content_tsv()
        started = db_search.kick_embedding_backfill()
        return {
            "status": "started" if started else "already_running_or_not_configured",
            "content_tsv_updated": updated_tsv,
            "backfill": await db_search.get_embedding_backfill_status(),
        }
    except Exception:
        return JSONResponse(status_code=500, content=shared._api_failure("重建对话搜索失败"))


@router.get("/api/admin/conversation-embedding-status")
async def api_conversation_embedding_status():
    return await db_search.get_embedding_backfill_status()


@router.patch("/api/chat/messages/{message_id}")
async def api_update_message(message_id: int, request: Request):
    """编辑单条消息内容"""
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return {"error": "内容不能为空"}
        updated = await db_conversations.update_message_content(message_id, content)
        if updated == 0:
            return {"error": "消息不存在"}
        return {"status": "ok"}
    except Exception:
        return shared._api_failure("保存消息失败")


@router.delete("/api/chat/messages/{message_id}")
async def api_delete_message(message_id: int):
    """删除单条消息"""
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        deleted = await db_conversations.delete_single_message(message_id)
        if deleted == 0:
            return {"error": "消息不存在"}
        return {"status": "ok"}
    except Exception:
        return shared._api_failure("删除消息失败")


@router.get("/api/conversations/export")
async def api_export_conversations():
    """导出所有对话记录"""
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        data = await db_conversations.export_all_conversations()
        return JSONResponse(content=data)
    except Exception:
        return shared._api_failure("导出对话失败")


@router.post("/api/conversations/import")
async def api_import_conversations(request: Request):
    """导入对话记录（JSON格式，自动去重）"""
    if not shared.conversation_persistence_enabled():
        return {"error": "对话持久化未启用"}
    try:
        records = await request.json()
        if not isinstance(records, list):
            return {"error": "格式错误：需要 JSON 数组"}
        imported, skipped = await db_conversations.import_conversations(records)
        return {"status": "ok", "imported": imported, "skipped": skipped, "total": imported + skipped}
    except Exception:
        return shared._api_failure("导入对话失败")
