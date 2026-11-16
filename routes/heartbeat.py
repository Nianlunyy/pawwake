"""Autonomous Heartbeat routes: Cron tick & incremental proactive message sync."""

from fastapi import APIRouter, Request

import heartbeat_service
import shared
from db import conversations as db_conversations

router = APIRouter()


@router.post("/api/heartbeat/tick")
@router.get("/api/heartbeat/tick")
async def api_heartbeat_tick(request: Request = None, background: bool = False):
    """外部 12 分钟 Cron 保活兼自主唤醒统一入口；background=1 时过门卫后即刻响应 200 并转后台生成。"""
    try:
        if request is not None:
            cuplivo_origin = str(request.headers.get("X-Cuplivo-Origin") or "").strip().rstrip("/")
            if cuplivo_origin:
                shared.CUPLIVO_ORIGIN = cuplivo_origin
        result = await heartbeat_service.dispatch_heartbeat_tick(background=background)
        return {"ok": True, **result}
    except Exception:
        return shared._api_failure("自主唤醒执行失败")


@router.get("/api/proactive/messages")
async def api_proactive_messages(session_id: str = "", after_id: int = 0, limit: int = 50):
    """按自增 ID 增量拉取自主唤醒消息，供 Cuplivo 无弹窗同步。"""
    try:
        target_sid = (session_id or "").strip() or await heartbeat_service.resolve_target_session_id()
        if not target_sid:
            return {"status": "ok", "session_id": "", "messages": [], "latest_id": after_id}
        msgs = await db_conversations.get_proactive_messages_since(
            target_sid,
            after_id=after_id,
            limit=limit,
        )
        latest_id = msgs[-1]["id"] if msgs else after_id
        return {
            "status": "ok",
            "session_id": target_sid,
            "messages": msgs,
            "latest_id": latest_id,
        }
    except Exception:
        return shared._api_failure("拉取增量唤醒消息失败", messages=[])


@router.get("/api/heartbeat/status")
async def api_heartbeat_status():
    """查看自主唤醒开关、锁定主会话与当前门卫状态。"""
    try:
        target_sid = await heartbeat_service.resolve_target_session_id()
        gate = (
            await heartbeat_service.evaluate_heartbeat_gate(target_sid)
            if target_sid
            else {"allowed": False, "status": "skipped_no_session"}
        )
        stats = gate.get("stats") or {}
        pending_alarm = stats.get("pending_alarm")
        alarm_info = None
        if isinstance(pending_alarm, dict):
            due_at = pending_alarm.get("due_at")
            alarm_info = {
                "id": pending_alarm.get("id"),
                "due_at": due_at.isoformat() if hasattr(due_at, "isoformat") else str(due_at or ""),
                "reason": pending_alarm.get("reason", ""),
                "status": pending_alarm.get("status", "pending"),
            }
        return {
            "status": "ok",
            "enabled": shared.HEARTBEAT_ENABLED,
            "session_id": target_sid,
            "gate_status": gate.get("status"),
            "unread_proactive_chat": stats.get("unread_proactive_chat", 0),
            "unread_proactive_event": stats.get("unread_proactive_event", 0),
            "max_unread": shared.HEARTBEAT_MAX_UNREAD,
            "base_cooldown_minutes": shared.HEARTBEAT_BASE_COOLDOWN_MINUTES,
            "quiet_hours": f"{shared.HEARTBEAT_QUIET_START}-{shared.HEARTBEAT_QUIET_END}",
            "pending_alarm": alarm_info,
        }
    except Exception:
        return shared._api_failure("获取自主唤醒状态失败")
