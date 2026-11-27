"""Autonomous Heartbeat routes: Cron tick & incremental proactive message sync."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import heartbeat_service
import shared
from db import conversations as db_conversations
from routes import chat as chat_routes

router = APIRouter()


@router.post("/api/heartbeat/tool-config")
async def api_heartbeat_tool_config(request: Request):
    """供 Cuplivo Worker 在设置保存后即时同步 Tavily/MCP 工具直连配置（不触发唤醒）。"""
    try:
        cuplivo_origin = str(request.headers.get("X-Cuplivo-Origin") or "").strip().rstrip("/")
        if cuplivo_origin:
            shared.CUPLIVO_ORIGIN = cuplivo_origin

        raw_body = await request.body()
        if not raw_body:
            return JSONResponse(status_code=400, content={"error": "请求体不能为空"})
        try:
            body = json.loads(raw_body)
        except (ValueError, TypeError):
            return JSONResponse(status_code=400, content={"error": "请求体必须为合法 JSON 对象"})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": "请求体必须为 JSON 对象"})

        if isinstance(body.get("tool_config"), dict):
            cfg = body["tool_config"]
        elif any(
            k in body
            for k in (
                "tavily_api_key",
                "clear_tavily_api_key",
                "tavily_max_results",
                "mcp_tool_routes",
                "clear_mcp_tool_routes",
                "mcp_servers",
                "clear_mcp_servers",
            )
        ):
            cfg = body
        else:
            cfg = {}

        await heartbeat_service.save_heartbeat_tool_config(cfg)
        effective = await heartbeat_service.get_effective_tool_config()
        tavily_ready, tavily_source = heartbeat_service.get_tavily_key_status(effective)
        return {
            "ok": True,
            "tavily_ready": tavily_ready,
            "tavily_key_source": tavily_source,
        }
    except Exception as exc:
        return shared._api_failure("同步自主唤醒工具配置失败", error_type=type(exc).__name__)


@router.post("/api/heartbeat/tick")
@router.get("/api/heartbeat/tick")
async def api_heartbeat_tick(request: Request = None, background: bool = False):
    """外部 12 分钟 Cron 保活兼自主唤醒统一入口；background=1 时过门卫后即刻响应 200 并转后台生成。"""
    try:
        if request is not None:
            cuplivo_origin = str(request.headers.get("X-Cuplivo-Origin") or "").strip().rstrip("/")
            if cuplivo_origin:
                shared.CUPLIVO_ORIGIN = cuplivo_origin
            if request.method == "POST":
                try:
                    raw_body = await request.body()
                    if raw_body:
                        import json as _json
                        body = _json.loads(raw_body)
                        if isinstance(body, dict) and isinstance(body.get("tool_config"), dict):
                            await heartbeat_service.save_heartbeat_tool_config(body["tool_config"])
                except Exception:
                    pass
        result = await heartbeat_service.dispatch_heartbeat_tick(background=background)
        return {"ok": True, **result}
    except Exception as exc:
        return shared._api_failure("自主唤醒执行失败", error_type=type(exc).__name__)


@router.get("/api/proactive/messages")
async def api_proactive_messages(
    session_id: str = "",
    after_id: int = 0,
    limit: int = 50,
    recover_since_ms: int = 0,
):
    """按自增 ID 增量拉取自主唤醒消息，供 Cuplivo 无弹窗同步（断流恢复时带 recover_since_ms）。"""
    try:
        target_sid = (session_id or "").strip() or await heartbeat_service.resolve_target_session_id()
        if not target_sid:
            return {"status": "ok", "session_id": "", "messages": [], "latest_id": after_id}
        extra_kwargs = {"recover_since_ms": recover_since_ms} if recover_since_ms and recover_since_ms > 0 else {}
        msgs = await db_conversations.get_proactive_messages_since(
            target_sid,
            after_id=after_id,
            limit=limit,
            **extra_kwargs,
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
    """查看自主唤醒开关、锁定主会话、配额水位、待触发闹钟与 96h 冷却话题。"""
    try:
        target_sid = await heartbeat_service.resolve_target_session_id()
        gate = (
            await heartbeat_service.evaluate_heartbeat_gate(target_sid)
            if target_sid
            else {"allowed": False, "status": "skipped_no_session"}
        )
        stats = gate.get("stats") or {}
        if not stats and target_sid and shared.DATABASE_ENABLED:
            try:
                stats = await db_conversations.get_heartbeat_session_stats(target_sid)
            except Exception:
                stats = {}

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

        now_local = heartbeat_service._to_local_dt()
        if "daily_surf_count" in gate:
            daily_surf_count = int(gate.get("daily_surf_count") or 0)
        else:
            raw_surf_date = stats.get("daily_surf_date")
            if hasattr(raw_surf_date, "date") and callable(raw_surf_date.date):
                raw_surf_date = raw_surf_date.date()
            daily_surf_count = (
                int(stats.get("daily_surf_count") or 0)
                if (raw_surf_date is None or raw_surf_date == now_local.date())
                else 0
            )
        surf_daily_limit = max(0, min(10, int(shared.HEARTBEAT_SURF_DAILY_LIMIT)))
        surf_max_steps = max(2, min(10, int(shared.HEARTBEAT_SURF_MAX_STEPS)))
        surf_quota_remaining = max(0, surf_daily_limit - daily_surf_count)

        cooling_topics = []
        if target_sid and shared.DATABASE_ENABLED:
            try:
                raw_topics = await db_conversations.get_active_proactive_topics(target_sid)
                for item in raw_topics:
                    cooldown_dt = item.get("cooldown_until")
                    last_seen_dt = item.get("last_seen_at")
                    cooling_topics.append({
                        "topic_key": item.get("topic_key", ""),
                        "topic_summary": item.get("topic_summary", ""),
                        "hit_count": int(item.get("hit_count") or 1),
                        "is_cooling": bool(item.get("is_cooling")),
                        "last_seen_at": last_seen_dt.isoformat() if hasattr(last_seen_dt, "isoformat") else str(last_seen_dt or ""),
                        "cooldown_until": cooldown_dt.isoformat() if hasattr(cooldown_dt, "isoformat") else (str(cooldown_dt) if cooldown_dt else None),
                    })
            except Exception:
                cooling_topics = []

        tavily_ready = False
        tavily_key_source = "none"
        try:
            tool_config = await heartbeat_service.get_effective_tool_config()
            tavily_ready, tavily_key_source = heartbeat_service.get_tavily_key_status(tool_config)
        except Exception:
            tavily_ready, tavily_key_source = heartbeat_service.get_tavily_key_status(None)

        snapshot_tool_names: list[str] = []
        if target_sid and shared.DATABASE_ENABLED:
            try:
                envelope = await db_conversations.get_session_envelope_snapshot(target_sid) or {}
                if isinstance(envelope, dict):
                    raw_snap_tools = chat_routes._strip_secret_cards_tool(envelope.get("tools") or [])
                    if isinstance(raw_snap_tools, list):
                        for tool in raw_snap_tools:
                            if isinstance(tool, dict):
                                fn = tool.get("function")
                                name = (fn.get("name") if isinstance(fn, dict) else None) or tool.get("name")
                                if name:
                                    snapshot_tool_names.append(str(name))
            except Exception:
                snapshot_tool_names = []

        has_search_tool = any(n in ("search_web", "tavily_search") for n in snapshot_tool_names)
        fetch_tool_ready = "fetch_url" in snapshot_tool_names
        search_tool_ready = bool(tavily_ready and has_search_tool)

        return {
            "status": "ok",
            "enabled": shared.HEARTBEAT_ENABLED,
            "session_id": target_sid,
            "gate_status": gate.get("status"),
            "elapsed_minutes": gate.get("elapsed_minutes"),
            "required_minutes": gate.get("required_minutes"),
            "cooldown_context": gate.get("context_kind"),
            "unread_proactive_chat": stats.get("unread_proactive_chat", 0),
            "unread_proactive_event": stats.get("unread_proactive_event", 0),
            "max_unread": shared.HEARTBEAT_MAX_UNREAD,
            "base_cooldown_minutes": shared.HEARTBEAT_BASE_COOLDOWN_MINUTES,
            "quiet_hours": f"{shared.HEARTBEAT_QUIET_START}-{shared.HEARTBEAT_QUIET_END}",
            "pending_alarm": alarm_info,
            "daily_surf_count": daily_surf_count,
            "surf_daily_limit": surf_daily_limit,
            "surf_max_steps": surf_max_steps,
            "surf_quota_remaining": surf_quota_remaining,
            "cooling_topics": cooling_topics,
            "tavily_ready": tavily_ready,
            "tavily_key_source": tavily_key_source,
            "snapshot_tool_names": snapshot_tool_names,
            "search_tool_ready": search_tool_ready,
            "fetch_tool_ready": fetch_tool_ready,
        }
    except Exception:
        return shared._api_failure("获取自主唤醒状态失败")
