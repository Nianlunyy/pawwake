"""Autonomous Heartbeat service for Pawwake (Phase 1).

Provides:
- Non-reentrant locking & single active main session targeting
- Zero-token local gatekeeper (quiet hours, unread cap, context-aware stepped cooldown)
- Prefix-cache-preserving LLM invocation using main chat envelope snapshot
- Anti-robotic tail note & Open-Meteo weather context
- Optimistic lock & busy-yield protection before persisting proactive messages
"""

import asyncio
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone

import httpx

import partition_engine
import shared
from db import conversations as db_conversations
from routes import chat as chat_routes

logger = logging.getLogger(__name__)

_active_heartbeat_lock = asyncio.Lock()

_WEATHER_CACHE: dict = {"text": "", "fetched_at": 0.0, "key": ""}
_WEATHER_TTL_SECONDS = 30 * 60

_WMO_WEATHER_MAP = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "零星小雨",
    53: "毛毛雨",
    55: "细雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "中阵雨",
    82: "强阵雨",
    95: "雷阵雨",
}

_CLOSING_KEYWORDS = (
    "晚安",
    "睡了",
    "睡觉",
    "午睡",
    "睡午觉",
    "去开会",
    "开会去",
    "先忙",
    "去忙",
    "忙去了",
    "去洗澡",
    "开车",
    "休息了",
    "回头聊",
    "拜拜",
    "晚点聊",
    "明天见",
)

_INTERRUPTED_ENDINGS = ("?", "？", "吗", "呢", "啊", "呀", "…", "...")
_SILENT_TOKEN_RE = re.compile(r"^\s*\[?\s*SILENT\s*\]?\s*$", re.IGNORECASE)
_THOUGHT_TAG_RE = re.compile(r"<(thought|think)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _to_local_dt(dt: datetime | None = None) -> datetime:
    base = dt or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.astimezone(timezone(timedelta(hours=shared.TIMEZONE_HOURS)))


def _parse_hhmm(text: str, fallback: tuple[int, int]) -> int:
    try:
        parts = str(text or "").strip().split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
    except (ValueError, IndexError, TypeError):
        pass
    return fallback[0] * 60 + fallback[1]


def is_quiet_hours(now_local: datetime, start_str: str, end_str: str) -> bool:
    """判断当前本地时间是否处于夜间静默时段（支持跨午夜）。"""
    start_min = _parse_hhmm(start_str, (23, 30))
    end_min = _parse_hhmm(end_str, (8, 0))
    if start_min == end_min:
        return False
    cur_min = now_local.hour * 60 + now_local.minute
    if start_min < end_min:
        return start_min <= cur_min < end_min
    return cur_min >= start_min or cur_min < end_min


def classify_user_context_multiplier(last_user_content: str) -> tuple[float, str]:
    """根据用户最后一句收尾语境返回动态冷却倍率。"""
    clean = partition_engine.strip_ui_state_suffix(last_user_content or "").strip()
    if not clean:
        return 1.0, "default"
    if any(kw in clean for kw in _CLOSING_KEYWORDS):
        return 2.2, "closing"
    if clean.endswith(_INTERRUPTED_ENDINGS) or len(clean) <= 25:
        return 0.75, "mid_chat"
    return 1.0, "normal"


def unread_backoff_multiplier(unread_chat_count: int) -> float:
    """按未读主动短句条数阶梯拉长空闲探头冷却时间。"""
    n = max(0, int(unread_chat_count or 0))
    if n == 0:
        return 1.0
    if n == 1:
        return 1.5
    if n == 2:
        return 2.5
    if n == 3:
        return 4.0
    return 6.0


async def resolve_target_session_id() -> str:
    """严格锁定唯一的活跃主会话，绝不遍历历史旧会话。"""
    sid = shared.resolve_heartbeat_session_id()
    if sid:
        return sid
    return await db_conversations.get_most_recent_active_session_id()


async def fetch_local_weather() -> str:
    """通过 Open-Meteo 免 Key 获取本地天气描述（带 30 分钟缓存，失败静默返回空串）。"""
    city = (shared.HEARTBEAT_WEATHER_CITY or "南京").strip()
    lat = shared.HEARTBEAT_WEATHER_LAT
    lon = shared.HEARTBEAT_WEATHER_LON
    cache_key = f"{city}:{lat:.4f},{lon:.4f}"
    now_mono = time.monotonic()
    if (
        _WEATHER_CACHE.get("key") == cache_key
        and _WEATHER_CACHE.get("text")
        and (now_mono - _WEATHER_CACHE.get("fetched_at", 0.0)) < _WEATHER_TTL_SECONDS
    ):
        return _WEATHER_CACHE["text"]

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weather_code&timezone=Asia%2FShanghai"
    )
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return _WEATHER_CACHE.get("text", "") if _WEATHER_CACHE.get("key") == cache_key else ""
        data = resp.json()
        current = data.get("current") or {}
        temp = current.get("temperature_2m")
        code = current.get("weather_code")
        desc = _WMO_WEATHER_MAP.get(int(code), "") if code is not None else ""
        parts = [city]
        if desc:
            parts.append(desc)
        if temp is not None:
            parts.append(f"{round(float(temp))}°C")
        text = " ".join(parts).strip()
        if text:
            _WEATHER_CACHE.update({"text": text, "fetched_at": now_mono, "key": cache_key})
        return text
    except Exception as e:
        logger.debug("Open-Meteo weather fetch skipped: %s", e)
        return _WEATHER_CACHE.get("text", "") if _WEATHER_CACHE.get("key") == cache_key else ""


def _describe_time_of_day(now_local: datetime) -> str:
    weekday = _WEEKDAYS_ZH[now_local.weekday()]
    h = now_local.hour
    if h < 6:
        period = "凌晨"
    elif h < 9:
        period = "早上"
    elif h < 12:
        period = "上午"
    elif h < 14:
        period = "中午"
    elif h < 18:
        period = "下午"
    elif h < 22:
        period = "晚上"
    else:
        period = "夜里"
    return f"{weekday}{period} {now_local.strftime('%H:%M')}"


def _describe_gap(elapsed_minutes: float) -> str:
    if elapsed_minutes < 45:
        return "刚分开没多久"
    if elapsed_minutes < 120:
        return "过去一个多钟头了"
    if elapsed_minutes < 300:
        return f"过去了约莫 {max(2, round(elapsed_minutes / 60))} 个钟头"
    if elapsed_minutes < 24 * 60:
        return "大半天没怎么说话了"
    days = max(1, round(elapsed_minutes / (24 * 60)))
    return f"隔了快 {days} 天"


def build_heartbeat_tail_note(
    now_local: datetime,
    elapsed_minutes: float,
    weather_text: str = "",
    unread_count: int = 0,
) -> str:
    """构建贴合生活质感的内心旁白纸条（不暴露系统模板与 JSON 格式）。"""
    time_desc = _describe_time_of_day(now_local)
    gap_desc = _describe_gap(elapsed_minutes)
    weather_clause = (
        f"，外头{weather_text}（只作你对窗外天色的隐约体感，台词里切勿生硬报出温度数字或天气播报词）"
        if weather_text
        else ""
    )
    unread_clause = (
        f"你前头已经顺手发过 {unread_count} 条她还没来得及回的微信，这会儿更得掂量分寸，别显得烦。"
        if unread_count > 0
        else "你刚把手头的事稍稍停了一会。"
    )
    return (
        f"（现在是{time_desc}{weather_clause}。距离阿狸上次和你说话已经{gap_desc}，她这会儿大概在忙自己的事。"
        f"{unread_clause}\n"
        f"如果觉得此刻没什么非说不可的、或者脑子里冒出的只是“忙完了吗 / 喝水没 / 注意休息 / 别太累”这种泛泛寒暄，"
        f"请直接只回复 [SILENT] 继续忙你自己的，不要硬找话；"
        f"只有当你确实想起前头聊过的某件具体小事、或手头有一句实在想顺嘴告诉她的话时，"
        f"才像平时发微信那样发 1~2 句简短自然的口语（隔着屏幕别写面对面肢体动作），末尾照常保留你的后台独白代码块。"
        f"本次只是随手发个消息或继续休息，不要调用搜索或外部工具。）"
    )


async def evaluate_heartbeat_gate(
    session_id: str,
    *,
    now_utc: datetime | None = None,
    has_due_alarm: bool = False,
    jitter_fn=None,
) -> dict:
    """纯本地零 Token 门卫，决定本次 Cron 唤醒是放行还是秒回跳过。"""
    if not shared.DATABASE_ENABLED:
        return {"allowed": False, "status": "skipped_db_disabled"}
    if not shared.HEARTBEAT_ENABLED:
        return {"allowed": False, "status": "skipped_disabled"}
    if not shared.API_KEY:
        return {"allowed": False, "status": "skipped_no_api_key"}
    if not session_id:
        return {"allowed": False, "status": "skipped_no_session"}
    if chat_routes.is_session_chat_busy(session_id):
        return {"allowed": False, "status": "skipped_busy"}

    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    now_local = _to_local_dt(now_utc)

    if is_quiet_hours(now_local, shared.HEARTBEAT_QUIET_START, shared.HEARTBEAT_QUIET_END):
        return {"allowed": False, "status": "skipped_quiet"}

    stats = await db_conversations.get_heartbeat_session_stats(session_id)
    if not stats.get("last_normal_at"):
        return {"allowed": False, "status": "skipped_empty_history", "stats": stats}

    unread_chat = int(stats.get("unread_proactive_chat", 0))
    max_unread = max(1, int(shared.HEARTBEAT_MAX_UNREAD))
    if not has_due_alarm and unread_chat >= max_unread:
        return {
            "allowed": False,
            "status": "skipped_unread_cap",
            "unread_proactive_chat": unread_chat,
            "stats": stats,
        }
    if has_due_alarm and unread_chat >= max_unread + 1:
        return {
            "allowed": False,
            "status": "skipped_hard_cap",
            "unread_proactive_chat": unread_chat,
            "stats": stats,
        }

    ref_ts = (
        stats.get("last_proactive_chat_at")
        if unread_chat > 0 and stats.get("last_proactive_chat_at")
        else stats.get("last_normal_at")
    )
    if ref_ts.tzinfo is None:
        ref_ts = ref_ts.replace(tzinfo=timezone.utc)
    elapsed_minutes = max(0.0, (now_utc - ref_ts).total_seconds() / 60.0)

    normal_anchor = stats.get("last_normal_at")
    if normal_anchor.tzinfo is None:
        normal_anchor = normal_anchor.replace(tzinfo=timezone.utc)
    interaction_gap_minutes = max(0.0, (now_utc - normal_anchor).total_seconds() / 60.0)

    if not has_due_alarm:
        ctx_mult, ctx_kind = classify_user_context_multiplier(stats.get("last_user_content", ""))
        step_mult = unread_backoff_multiplier(unread_chat)
        jitter = float(jitter_fn()) if jitter_fn else random.uniform(0.85, 1.15)
        base_min = max(1, int(shared.HEARTBEAT_BASE_COOLDOWN_MINUTES))
        required_minutes = max(10.0, base_min * ctx_mult * step_mult * jitter)
        if elapsed_minutes < required_minutes:
            return {
                "allowed": False,
                "status": "skipped_cooldown",
                "elapsed_minutes": round(elapsed_minutes, 1),
                "required_minutes": round(required_minutes, 1),
                "context_kind": ctx_kind,
                "unread_proactive_chat": unread_chat,
                "stats": stats,
            }
    else:
        ctx_kind = "due_alarm"
        required_minutes = 0.0

    return {
        "allowed": True,
        "status": "allowed",
        "now_utc": now_utc,
        "now_local": now_local,
        "elapsed_minutes": elapsed_minutes,
        "interaction_gap_minutes": interaction_gap_minutes,
        "required_minutes": required_minutes,
        "context_kind": ctx_kind,
        "unread_proactive_chat": unread_chat,
        "stats": stats,
    }


def _strip_thoughts_and_check_silent(raw_text: str) -> tuple[str, bool]:
    """剥离思考标签并判定是否为静默信号或空白正文。"""
    cleaned = _THOUGHT_TAG_RE.sub("", raw_text or "").strip()
    if not cleaned or _SILENT_TOKEN_RE.match(cleaned):
        return "", True
    if cleaned.upper().startswith("[SILENT]"):
        return "", True
    return cleaned, False


async def _load_partition_history_for_heartbeat(session_id: str) -> list[dict]:
    db_history, cutoff_row, a_start_round, is_sliced = (
        await db_conversations.get_partition_conversation_messages(session_id, limit=10000)
    )
    db_msgs = []
    if is_sliced and a_start_round > 0:
        cutoff_id = cutoff_row.get("id") if cutoff_row else None
        cutoff_at = cutoff_row.get("created_at") if cutoff_row else None
        db_msgs.extend([{"role": "user", "content": "", "_virtual": True}] * (a_start_round - 1))
        db_msgs.append({
            "role": "user",
            "content": "",
            "id": cutoff_id,
            "created_at": cutoff_at,
            "_virtual": True,
        })
    for m in (db_history or []):
        msg = db_conversations.db_row_to_message(m)
        msg["created_at"] = m.get("created_at")
        if m.get("message_type") and m.get("message_type") != "normal":
            msg["message_type"] = m.get("message_type")
        if m.get("proactive_meta") is not None:
            msg["proactive_meta"] = m.get("proactive_meta")
        if m.get("skip_memory_extract"):
            msg["skip_memory_extract"] = True
        db_msgs.append(msg)
    return db_msgs


async def run_heartbeat_tick(
    *,
    now_utc: datetime | None = None,
    jitter_fn=None,
) -> dict:
    """执行一次完整的自主唤醒周期（含防重入锁、零成本门卫、缓存复用、乐观锁落库）。"""
    if _active_heartbeat_lock.locked():
        return {"status": "skipped_reentrant"}

    async with _active_heartbeat_lock:
        if not shared.DATABASE_ENABLED:
            return {"status": "skipped_db_disabled"}
        if not shared.HEARTBEAT_ENABLED:
            return {"status": "skipped_disabled"}

        session_id = await resolve_target_session_id()
        gate = await evaluate_heartbeat_gate(session_id, now_utc=now_utc, jitter_fn=jitter_fn)
        if not gate.get("allowed"):
            return {"status": gate["status"], "session_id": session_id}

        stats = gate["stats"]
        baseline_msg_id = stats.get("latest_id")
        now_local = gate["now_local"]
        weather_text = await fetch_local_weather()
        tail_note = build_heartbeat_tail_note(
            now_local,
            gate.get("interaction_gap_minutes", gate["elapsed_minutes"]),
            weather_text=weather_text,
            unread_count=gate["unread_proactive_chat"],
        )

        envelope = await db_conversations.get_session_envelope_snapshot(session_id) or {}
        model = envelope.get("model") or shared.DEFAULT_MODEL
        if not model:
            return {"status": "skipped_no_model", "session_id": session_id}

        resolved_system_prompt = await shared.get_system_prompt()
        partition_prompt = resolved_system_prompt
        if shared.memory_injection_enabled():
            partition_prompt = (resolved_system_prompt or "") + partition_engine.MEMORY_USAGE_GUIDE
        client_system_text = (envelope.get("client_system_text") or "").strip()
        if client_system_text:
            partition_prompt = ((partition_prompt or "") + "\n\n" + client_system_text).strip()

        db_msgs = await _load_partition_history_for_heartbeat(session_id)
        if shared.CACHE_PARTITION_ENABLED:
            built_messages = await partition_engine.build_partitioned_messages(
                session_id,
                db_msgs,
                partition_prompt,
                "",
                allow_rotation=False,
                temporary_tail_message={"role": "user", "content": tail_note},
            )
        else:
            built_messages = []
            if partition_prompt:
                built_messages.append({"role": "system", "content": partition_prompt})
            built_messages.extend(
                partition_engine._prepare_zone_messages(db_msgs, drop_tool_calls=False, drop_empty_assistant=False)
            )
            partition_engine._append_tail_user_message(built_messages, {"role": "user", "content": tail_note})
            built_messages = partition_engine.fold_historical_interactive_ui_messages(built_messages)

        if shared.CACHE_PARTITION_ENABLED and not partition_engine._is_anthropic_model(model):
            partition_engine._strip_cache_control(built_messages)

        req_body: dict = {
            "model": model,
            "messages": built_messages,
            "stream": False,
        }
        if envelope.get("tools"):
            req_body["tools"] = envelope["tools"]

        effective_effort = envelope.get("thinking_level") or shared.REASONING_EFFORT
        if effective_effort:
            if shared.is_vertex_endpoint():
                req_body["google"] = {
                    "thinking_config": {
                        "thinking_level": effective_effort,
                        "include_thoughts": True,
                    }
                }
            else:
                req_body["reasoning_effort"] = effective_effort

        req_id = f"hb{int(time.time()) % 10000:04d}"
        try:
            chat_routes.diagnose_cache_prefix(
                session_id,
                req_body,
                req_id=req_id,
                db_history=len(db_msgs),
            )
        except Exception:
            pass

        async with httpx.AsyncClient(timeout=180) as client:
            response = await chat_routes._post_chat_completion_with_429_retry(
                client,
                shared.API_BASE_URL,
                shared.API_KEY,
                req_body,
                tag="自主唤醒",
                max_retries=1 if shared.is_vertex_endpoint() else 0,
                req_id=req_id,
            )

        if response.status_code != 200:
            logger.warning("Heartbeat upstream HTTP %s: %s", response.status_code, response.text[:300])
            return {"status": "upstream_error", "code": response.status_code, "session_id": session_id}

        resp_data = response.json()
        usage = resp_data.get("usage") or {}
        try:
            chat_routes.log_cache_usage(usage, req_id=req_id)
        except Exception:
            pass
        tt = usage.get("total_tokens", 0)
        if tt > 0:
            asyncio.create_task(
                db_conversations.save_token_usage(
                    session_id,
                    model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    tt,
                    usage_type="heartbeat",
                )
            )

        msg_obj = (resp_data.get("choices") or [{}])[0].get("message") or {}
        # 禁工具期误调 tools 或输出空正文：一律按 [SILENT] 静默丢弃，绝不写空消息或报错
        if msg_obj.get("tool_calls"):
            print(f"🤫 req={req_id} 自主唤醒意外触发 tool_calls，熔断静默丢弃")
            return {"status": "silent", "reason": "unexpected_tool_calls", "session_id": session_id}

        cleaned_content, is_silent = _strip_thoughts_and_check_silent(msg_obj.get("content") or "")
        if is_silent:
            print(f"💤 req={req_id} 自主唤醒选择静默继续睡（零落库）")
            return {"status": "silent", "reason": "model_silent", "session_id": session_id}

        # 落库前二次校验并发占线与最新 message_id 乐观锁
        if chat_routes.is_session_chat_busy(session_id):
            print(f"🛑 req={req_id} 唤醒落库前检测到主聊天占线，静默让路")
            return {"status": "aborted_conflict", "reason": "chat_busy", "session_id": session_id}

        latest_meta = await db_conversations.get_latest_message_meta(session_id)
        latest_id = latest_meta.get("id") if latest_meta else None
        if latest_id != baseline_msg_id:
            print(f"🛑 req={req_id} 唤醒期间会话出现新消息 ({baseline_msg_id} -> {latest_id})，静默丢弃结果")
            return {"status": "aborted_conflict", "reason": "baseline_changed", "session_id": session_id}

        reasoning_content = msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or None
        meta_str = json.dumps({"reasoning_content": reasoning_content}, ensure_ascii=False) if reasoning_content else None
        proactive_meta = {
            "trigger_source": "cron",
            "weather": weather_text,
            "context_kind": gate.get("context_kind", "normal"),
        }
        msg_id = await db_conversations.save_message(
            session_id,
            "assistant",
            cleaned_content,
            model=model,
            metadata=meta_str,
            message_type="proactive_chat",
            proactive_meta=proactive_meta,
            skip_memory_extract=False,
            memory_extracted=False,
        )
        print(f"💌 req={req_id} 自主唤醒落库成功: id={msg_id}, session={session_id}, len={len(cleaned_content)}")
        return {
            "status": "delivered",
            "session_id": session_id,
            "message_id": msg_id,
            "message_type": "proactive_chat",
        }
