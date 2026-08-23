"""Partition-cache state and thread routes."""

from fastapi import APIRouter, Request

import shared
from db import core as db_core
from db import conversations as db_conversations

router = APIRouter()

# ============================================================
# 对话线管理 API（分区缓存）
# ============================================================

@router.get("/api/partition/status")
async def api_partition_status():
    active_sid = shared.get_active_session_id()
    state = await db_conversations.get_session_cache_state(active_sid) if active_sid else {}
    return {
        "enabled": shared.CACHE_PARTITION_ENABLED,
        "active_session_id": active_sid,
        "partition_x": shared.CACHE_PARTITION_X,
        "summary_model": shared.CACHE_SUMMARY_MODEL,
        "summary": '\n\n'.join(state.get('summary_parts', [])),
        "summary_parts": state.get('summary_parts', []),
        "summary_count": len(state.get('summary_parts', [])),
        "summary_length": sum(len(p) for p in state.get('summary_parts', [])),
        "a_start_round": state.get('a_start_round', 0),
        "updated_at": state.get('updated_at').isoformat() if state.get('updated_at') else None,
    }


@router.get("/api/partition/threads")
async def api_partition_threads():
    threads = await db_conversations.list_all_session_cache_states()
    active_sid = shared.get_active_session_id()
    for t in threads:
        t['is_active'] = (t['session_id'] == active_sid)
    if active_sid and not any(t['session_id'] == active_sid for t in threads):
        threads.insert(0, {'session_id': active_sid, 'summary': '', 'summary_length': 0, 'summary_count': 0, 'a_start_round': 0, 'updated_at': None, 'message_count': 0, 'chat_tokens': 0, 'is_active': True})
    return {"threads": threads, "active_session_id": active_sid}


@router.put("/api/partition/summary")
async def api_update_summary(request: Request):
    try:
        body = await request.json()
        sid = body.get("session_id", "")
        summary = body.get("summary", "")
        if not sid:
            return {"error": "session_id 不能为空"}
        state = await db_conversations.get_session_cache_state(sid)
        summary_parts = [summary] if isinstance(summary, str) and summary else summary if isinstance(summary, list) else []
        # 摘要清空时 a_start_round 也归零，否则历史会被跳过
        a_start = state.get('a_start_round', 0) if summary_parts else 0
        await db_conversations.save_session_cache_state(sid, summary_parts, a_start)
        total_len = sum(len(p) for p in summary_parts)
        return {"status": "ok", "summary_parts": len(summary_parts), "summary_length": total_len}
    except Exception:
        return shared._api_failure("保存摘要失败")


@router.delete("/api/partition/summary")
async def api_clear_summary(request: Request):
    try:
        body = await request.json()
        sid = body.get("session_id", "")
        if not sid:
            return {"error": "session_id 不能为空"}
        # 摘要和 a_start_round 一起归零
        await db_conversations.save_session_cache_state(sid, [], 0)
        return {"status": "ok"}
    except Exception:
        return shared._api_failure("清空摘要失败")


@router.post("/api/partition/thread")
async def api_create_thread(request: Request):
    try:
        body = await request.json()
        new_id = body.get("session_id", "").strip()
        copy_from = body.get("copy_summary_from", "")
        if not new_id:
            return {"error": "session_id 不能为空"}
        existing = await db_conversations.get_session_cache_state(new_id)
        if existing.get('updated_at'):
            return {"error": f"对话线 '{new_id}' 已存在"}
        summary_parts = []
        if copy_from:
            source = await db_conversations.get_session_cache_state(copy_from)
            summary_parts = source.get('summary_parts', [])
        await db_conversations.save_session_cache_state(new_id, summary_parts, 0)
        total_len = sum(len(p) for p in summary_parts)
        return {"status": "ok", "session_id": new_id, "summary_length": total_len}
    except Exception:
        return shared._api_failure("创建对话线失败")


@router.post("/api/partition/switch")
async def api_switch_thread(request: Request):
    try:
        body = await request.json()
        new_id = body.get("session_id", "").strip()
        if not new_id:
            return {"error": "session_id 不能为空"}
        old_id = shared.PARTITION_SESSION_ID
        shared.PARTITION_SESSION_ID = new_id
        await db_core.set_gateway_config("partition_session_id", new_id)
        return {"status": "ok", "old_session_id": old_id, "new_session_id": new_id}
    except Exception:
        return shared._api_failure("切换对话线失败")


@router.put("/api/partition/thread/rename")
async def api_rename_thread(request: Request):
    try:
        body = await request.json()
        old_id = body.get("old_id", "").strip()
        new_id = body.get("new_id", "").strip()
        if not old_id or not new_id:
            return {"error": "old_id 和 new_id 不能为空"}
        if old_id == new_id:
            return {"error": "新旧ID相同"}
        success = await db_conversations.rename_session_id(old_id, new_id)
        if not success:
            return {"error": f"对话线 '{new_id}' 已存在"}
        # 如果重命名的是活跃线，同步更新
        if shared.PARTITION_SESSION_ID == old_id:
            shared.PARTITION_SESSION_ID = new_id
            await db_core.set_gateway_config("partition_session_id", new_id)
        return {"status": "ok", "old_id": old_id, "new_id": new_id}
    except Exception:
        return shared._api_failure("重命名对话线失败")


@router.delete("/api/partition/thread/{session_id:path}")
async def api_delete_thread(session_id: str):
    """删除对话线（不允许删除当前活跃线）"""
    try:
        active_sid = shared.get_active_session_id()
        if session_id == active_sid:
            return {"error": "不能删除当前活跃的对话线"}
        await db_conversations.delete_session_cache_state(session_id)
        print(f"🗑️ 删除对话线: {session_id}")
        return {"status": "ok", "session_id": session_id}
    except Exception:
        return shared._api_failure("删除对话线失败")
