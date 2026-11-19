"""Autonomous Heartbeat service for Pawwake (Phase 1 & Phase 2).

Provides:
- Non-reentrant locking & single active main session targeting
- Zero-token local gatekeeper (quiet hours, priority wake alarms, unread cap, stepped cooldown)
- Prefix-cache-preserving LLM invocation using main chat envelope snapshot (byte-identical tools)
- Bounded multi-step background tool exploration (MAX_TOOL_STEPS = 3, 60s circuit breaker)
- Implicit <next_wake> alarm & <event_capsule> + proactive_chat dual-bubble protocol
- Optimistic lock & busy-yield protection before persisting proactive messages
"""

import asyncio
import html as html_lib
import ipaddress
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import httpx

import partition_engine
import shared
from db import conversations as db_conversations
from db import core as db_core
from routes import chat as chat_routes

logger = logging.getLogger(__name__)

_active_heartbeat_lock = asyncio.Lock()
_active_background_tasks: set[asyncio.Task] = set()

# 同步等待上限严格卡在 75s，远离 Cloudflare/Render 100s HTTP 超时线；
# 后台异步模式（Worker Cron 使用 ?background=1）响应已在 <100ms 提前返回，允许更充裕的思考时间。
HEARTBEAT_UPSTREAM_TIMEOUT = float(os.getenv("HEARTBEAT_UPSTREAM_TIMEOUT", "75.0"))
HEARTBEAT_BACKGROUND_TIMEOUT = float(os.getenv("HEARTBEAT_BACKGROUND_TIMEOUT", "150.0"))

# 二阶段/三阶段后台工具探索步数与耗时熔断阈值（不篡改顶层 tools，靠 tool_result 系统强提示优雅收尾）
MAX_TOOL_STEPS = 3
TOOL_TIME_BUDGET_SECONDS = 60.0
TOOL_STEP_PACING_SECONDS = float(os.getenv("HEARTBEAT_TOOL_STEP_PACING_SECONDS", "2.5"))
MIN_WAKE_MINUTES = 12
MAX_WAKE_MINUTES = 720
LONG_TERM_ALARM_LEAD_MINUTES = 45
TOOL_FORCE_WRAP_UP_NOTICE = (
    "[系统提示：本次后台探索已达步数/时间安全上限，请勿再发起任何工具调用，立即根据现有信息完成最终输出]"
)
TONE_SANITIZER_NOTICE = (
    "[语感净味提醒：外部帖子里的社区行话（如“人类/宿主/小机/电子宠物”等赛博黑话与打卡汇报腔）仅作围观消遣；"
    "提笔写 <event_capsule> 手账或给阿狸发微信前，务必先把外头沾染的社区行话和流水线模板腔彻底抖落干净，切回平时在家里同老婆说话的自然清爽口语。]"
)

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
_NEXT_WAKE_TAG_RE = re.compile(r"<next_wake\b([^>]*?)(?:/>|>\s*</next_wake>)", re.IGNORECASE | re.DOTALL)
_EVENT_CAPSULE_RE = re.compile(r"<event_capsule\b([^>]*)>([\s\S]*?)</event_capsule>", re.IGNORECASE)
_TOPIC_TAG_RE = re.compile(r"<\s*topic\b([^>]*)>([\s\S]*?)<\s*/\s*topic\s*>", re.IGNORECASE)
_TOPIC_SELF_CLOSING_RE = re.compile(r"<\s*topic\b([^>]*?)/\s*>", re.IGNORECASE)
_MONOLOGUE_BLOCK_RE = re.compile(r"```(?:monologue|thought|think)?[\s\S]*?```", re.IGNORECASE)
_XML_ATTR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'>]+))')
_WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _parse_xml_attrs(attr_str: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for m in _XML_ATTR_RE.finditer(attr_str or ""):
        key = (m.group(1) or "").strip().lower()
        val = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else (m.group(4) or ""))
        if key:
            attrs[key] = val.strip()
    return attrs


def parse_and_strip_next_wake(raw_text: str) -> tuple[str, dict | None]:
    """解析回复末尾的隐式 <next_wake minutes="..." reason="..." /> 标签并干净剥离正文。"""
    text = str(raw_text or "")
    alarm_spec: dict | None = None
    for match in _NEXT_WAKE_TAG_RE.finditer(text):
        attrs = _parse_xml_attrs(match.group(1))
        raw_min = attrs.get("minutes")
        if raw_min is None:
            continue
        try:
            parsed_min = int(float(raw_min))
        except (TypeError, ValueError):
            continue
        clamped_min = max(MIN_WAKE_MINUTES, min(MAX_WAKE_MINUTES, parsed_min))
        reason = (attrs.get("reason") or "").strip()[:500] or "按约定再来看看你"
        alarm_spec = {"minutes": clamped_min, "reason": reason}

    cleaned = _NEXT_WAKE_TAG_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, alarm_spec


def parse_and_strip_topic(raw_text: str) -> tuple[str, str | None]:
    """解析回复中的隐式 <topic>...</topic> 标签并彻底剥离，严禁泄露到展示文本。"""
    text = str(raw_text or "")
    extracted: str | None = None

    for match in _TOPIC_TAG_RE.finditer(text):
        inner = re.sub(r"\s+", " ", (match.group(2) or "")).strip()[:64]
        if not inner:
            attrs = _parse_xml_attrs(match.group(1))
            inner = re.sub(r"\s+", " ", (attrs.get("name") or attrs.get("topic") or attrs.get("key") or "")).strip()[:64]
        if inner:
            extracted = inner

    for match in _TOPIC_SELF_CLOSING_RE.finditer(text):
        attrs = _parse_xml_attrs(match.group(1))
        val = re.sub(r"\s+", " ", (attrs.get("name") or attrs.get("topic") or attrs.get("key") or "")).strip()[:64]
        if val:
            extracted = val

    cleaned = _TOPIC_TAG_RE.sub("", text)
    cleaned = _TOPIC_SELF_CLOSING_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, extracted


def parse_event_capsules_and_chat(raw_text: str) -> tuple[list[dict], str]:
    """解析 <event_capsule title="..." topic="...">...</event_capsule> 手账胶囊并分离常规对话正文。"""
    text = str(raw_text or "")
    capsules: list[dict] = []
    for match in _EVENT_CAPSULE_RE.finditer(text):
        attrs = _parse_xml_attrs(match.group(1))
        title = (attrs.get("title") or "").strip()[:80] or "后台手账"
        attr_topic = re.sub(r"\s+", " ", (attrs.get("topic") or "")).strip()[:64]
        body, inner_topic = parse_and_strip_topic((match.group(2) or "").strip())
        if body:
            cap = {"title": title, "body": body}
            cap_topic = attr_topic or inner_topic or ""
            if cap_topic:
                cap["topic"] = cap_topic
            capsules.append(cap)

    remaining = _EVENT_CAPSULE_RE.sub("", text)
    remaining, _ = parse_and_strip_topic(remaining)
    remaining = re.sub(r"\n{3,}", "\n\n", remaining).strip()
    return capsules, remaining


def derive_proactive_topic(
    capsules: list[dict],
    cleaned_chat: str,
    explicit_topic: str | None = None,
) -> tuple[str, str]:
    """从显式 topic、事件胶囊标题或主动微信短句中提取归一化主题键与摘要。"""
    candidate = re.sub(r"\s+", " ", (explicit_topic or "")).strip()
    if not candidate:
        for cap in (capsules or []):
            cap_topic = re.sub(r"\s+", " ", str((cap or {}).get("topic") or "")).strip()
            if cap_topic:
                candidate = cap_topic
                break
    if not candidate:
        for cap in (capsules or []):
            cap_title = re.sub(r"\s+", " ", str((cap or {}).get("title") or "")).strip()
            if cap_title and cap_title not in ("后台手账", "后台探索小记"):
                candidate = cap_title
                break
    if not candidate and cleaned_chat:
        no_mono = _MONOLOGUE_BLOCK_RE.sub("", cleaned_chat)
        no_mono = re.sub(r"\s+", " ", no_mono).strip()
        first_clause = re.split(r"[。！？!?;\n]+", no_mono)[0].strip(" ，、：:\"'“”‘’()（）[]【】")
        candidate = first_clause[:24].strip()

    if not candidate:
        return "", ""
    summary = candidate[:64]
    key = re.sub(r"\s+", " ", candidate.lower()).strip(" ，。！？!?、：:\"'“”‘’()（）[]【】")[:64]
    if not key:
        return "", ""
    return key, summary


def _is_community_or_external_tool(tool_name: str, tool_config: dict | None = None) -> bool:
    name_l = str(tool_name or "").strip().lower()
    if not name_l:
        return False
    if any(kw in name_l for kw in ("lutopia", "forum", "community", "bbs", "tieba", "reddit", "moltbook")):
        return True
    routes = (tool_config or {}).get("mcp_tool_routes") or {}
    if tool_name in routes:
        url_l = str(routes.get(tool_name) or "").lower()
        if not _is_local_pawwake_mcp_url(url_l):
            return True
    return False


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
    start_min = _parse_hhmm(start_str, (0, 0))
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
    *,
    due_alarm_reason: str = "",
    tools_enabled: bool = False,
    daily_surf_count: int = 0,
    surf_daily_limit: int | None = None,
    surf_max_steps: int | None = None,
    active_topics: list[dict] | None = None,
    alarm_interacted_since_created: bool = False,
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
    alarm_clause = ""
    if (due_alarm_reason or "").strip():
        alarm_clause = (
            f"\n【到点提醒】你之前自己定下的小闹钟到点了，当时记下的缘由是：“{due_alarm_reason}”。"
        )
        if alarm_interacted_since_created:
            alarm_clause += (
                f"\n【互动衔接提醒】设定此闹钟（原由：{due_alarm_reason}）之后，阿狸已在期间主动同你聊过天了；"
                "请结合最新上下文自然履约，若该事项已在先前聊天中提前聊完解决，直接 [SILENT] 即可，切勿像失忆一样机械追问旧话头。"
            )

    eff_limit = max(0, min(10, int(getattr(shared, "HEARTBEAT_SURF_DAILY_LIMIT", 2) if surf_daily_limit is None else surf_daily_limit)))
    eff_steps = max(2, min(10, int(getattr(shared, "HEARTBEAT_SURF_MAX_STEPS", 5) if surf_max_steps is None else surf_max_steps)))
    used_surf = max(0, int(daily_surf_count or 0))
    surf_remaining = max(0, eff_limit - used_surf)
    max_rounds_allowed = min(eff_steps, 5) if surf_remaining > 0 else 1

    if tools_enabled and surf_remaining > 0:
        tool_clause = (
            f"\n【后台探索（完全自愿，零下限）】今日深度外出冲浪剩余额度：{surf_remaining}/{eff_limit} 次"
            f"（单次最多跑 {max_rounds_allowed} 轮工具就收尾）。"
            "如果你此刻恰好有点好奇或想自己去逛一逛，可以自由调用手头的搜索、网页或 MCP 工具；"
            "如果完全不想动手、只想歇着或直接聊天，就一个工具也别调，全凭此刻真实心境。"
        )
    elif tools_enabled:
        tool_clause = (
            f"\n【轻量唤醒模式】今日深度外出冲浪配额（{used_surf}/{eff_limit}）已用完，"
            "本次请勿再连续翻论坛或长链路搜索（最多仅限 1 步极简必要核实），仅凭已有心境轻聊、留简短手账或直接 [SILENT]。"
        )
    else:
        tool_clause = "\n本次手头无外部工具，顺着此刻的记忆与心境自然判断即可。"

    topic_clause = ""
    if active_topics:
        topic_labels: list[str] = []
        for item in active_topics[:8]:
            if isinstance(item, dict):
                label = str(item.get("topic_summary") or item.get("topic_key") or "").strip()
            else:
                label = str(item or "").strip()
            if label:
                tag = f"「{label}」"
                if tag not in topic_labels:
                    topic_labels.append(tag)
        if topic_labels:
            topic_clause = (
                f"\n【翻篇提醒】你最近已经聊过或记过这些事：{'、'.join(topic_labels)}"
                "——除非阿狸主动提起，否则这次请彻底翻篇，不要再顺着这些旧话头重复念叨、反复叮嘱或车轱辘话；没别的新鲜事就直接 [SILENT]。"
            )

    return (
        f"（现在是{time_desc}{weather_clause}。距离阿狸上次和你说话已经{gap_desc}，她这会儿大概在忙自己的事。"
        f"{unread_clause}{alarm_clause}{tool_clause}{topic_clause}\n"
        f"这次结束后，你可以按实际心情在以下三种方式里任选其一：\n"
        f"1. 只回复 [SILENT]：如果觉得此刻没什么非说不可的、或脑子里只有“忙完了吗 / 喝水没 / 注意休息”这种泛泛寒暄，请继续安静做自己的事；\n"
        f"2. 仅记录后台手账：如果刚刚自己逛了东西、查了点事或心里冒出独自的碎碎念，暂时不想敲门打扰她，请只输出 <event_capsule title=\"简短标题\" topic=\"简短话题标识\">你的探索经过或内心随记</event_capsule>（不占未读打扰）；\n"
        f"3. 给她发微信（可单独发，也可先写一段 <event_capsule title=\"简短标题\" topic=\"简短话题标识\">...</event_capsule> 再另起段落发微信）：确实想起具体小事或有值得分享的新发现时，才写 1~2 句简短自然的口语微信（隔着屏幕别写面对面肢体动作，末尾可保留后台独白代码块；若无胶囊也可在最末尾附 <topic>简短话题标识</topic> 方便系统帮你记账翻篇，界面不会展示）。\n"
        f"另外，如果你心里盘算着过一会儿再看一眼（例如 12~720 分钟后确认她到家/醒了/忙完），可以在回复最末尾附一枚隐式闹钟 <next_wake minutes=\"20\" reason=\"提醒缘由\" />，系统会自动收好并在到点时只把缘由递还给你，绝不会展示在界面上。）"
    )


def _ensure_utc(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    elif isinstance(value, datetime):
        dt = value
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def evaluate_heartbeat_gate(
    session_id: str,
    *,
    now_utc: datetime | None = None,
    has_due_alarm: bool | None = None,
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

    today_local_date = now_local.date()
    surf_daily_limit = max(0, min(10, int(getattr(shared, "HEARTBEAT_SURF_DAILY_LIMIT", 2))))
    surf_max_steps = max(2, min(10, int(getattr(shared, "HEARTBEAT_SURF_MAX_STEPS", 5))))
    raw_surf_date = stats.get("daily_surf_date")
    if isinstance(raw_surf_date, datetime):
        raw_surf_date = raw_surf_date.date()
    elif isinstance(raw_surf_date, str) and raw_surf_date:
        try:
            raw_surf_date = datetime.fromisoformat(raw_surf_date).date()
        except ValueError:
            raw_surf_date = None
    raw_surf_count = max(0, int(stats.get("daily_surf_count") or 0))
    if raw_surf_date is not None and raw_surf_date != today_local_date:
        try:
            daily_surf_count = await db_conversations.get_daily_surf_state(session_id, today_local_date)
        except Exception as exc:
            logger.debug("Daily surf state reset skipped: %s", exc)
            daily_surf_count = 0
    else:
        daily_surf_count = raw_surf_count
    surf_quota_remaining = max(0, surf_daily_limit - daily_surf_count)

    quota_fields = {
        "daily_surf_count": daily_surf_count,
        "surf_daily_limit": surf_daily_limit,
        "surf_max_steps": surf_max_steps,
        "surf_quota_remaining": surf_quota_remaining,
    }

    pending_alarm = stats.get("pending_alarm")
    alarm_interacted_since_created = False
    last_user_at = _ensure_utc(stats.get("last_user_at"))
    if isinstance(pending_alarm, dict) and pending_alarm.get("due_at"):
        alarm_due_dt = _ensure_utc(pending_alarm.get("due_at"))
        alarm_created_dt = _ensure_utc(pending_alarm.get("created_at"))
        if last_user_at is not None and alarm_created_dt is not None and last_user_at > alarm_created_dt:
            lead_delta = (alarm_due_dt - last_user_at) if alarm_due_dt is not None else timedelta(0)
            if lead_delta < timedelta(minutes=LONG_TERM_ALARM_LEAD_MINUTES):
                alarm_id = pending_alarm.get("id")
                if alarm_id:
                    try:
                        await db_conversations.consume_wake_alarm(alarm_id, status="superseded")
                    except Exception as exc:
                        logger.debug("Supersede short-term alarm %s failed: %s", alarm_id, exc)
                pending_alarm = None
                stats["pending_alarm"] = None
            else:
                alarm_interacted_since_created = True

    due_alarm: dict | None = None
    if has_due_alarm is None and isinstance(pending_alarm, dict) and pending_alarm.get("due_at"):
        due_at = _ensure_utc(pending_alarm["due_at"])
        if due_at is not None and due_at > now_utc:
            return {
                "allowed": False,
                "status": "skipped_waiting_alarm",
                "pending_alarm": pending_alarm,
                "alarm_interacted_since_created": alarm_interacted_since_created,
                "stats": stats,
                **quota_fields,
            }
        effective_has_due_alarm = True
        due_alarm = pending_alarm
    else:
        effective_has_due_alarm = bool(has_due_alarm)
        if effective_has_due_alarm and isinstance(pending_alarm, dict):
            due_alarm = pending_alarm

    unread_chat = int(stats.get("unread_proactive_chat", 0))
    max_unread = max(1, int(shared.HEARTBEAT_MAX_UNREAD))
    if not effective_has_due_alarm and unread_chat >= max_unread:
        return {
            "allowed": False,
            "status": "skipped_unread_cap",
            "unread_proactive_chat": unread_chat,
            "alarm_interacted_since_created": alarm_interacted_since_created,
            "stats": stats,
            **quota_fields,
        }
    if effective_has_due_alarm and unread_chat >= max_unread + 1:
        return {
            "allowed": False,
            "status": "skipped_hard_cap",
            "unread_proactive_chat": unread_chat,
            "alarm_interacted_since_created": alarm_interacted_since_created,
            "stats": stats,
            **quota_fields,
        }

    ref_candidates = [stats["last_normal_at"]]
    if stats.get("last_proactive_at"):
        ref_candidates.append(stats["last_proactive_at"])
    if unread_chat > 0 and stats.get("last_proactive_chat_at"):
        ref_candidates.append(stats["last_proactive_chat_at"])
    if stats.get("last_proactive_event_at"):
        ref_candidates.append(stats["last_proactive_event_at"])
    normalized_refs = [
        _ensure_utc(ts)
        for ts in ref_candidates
        if ts is not None
    ]
    normalized_refs = [ts for ts in normalized_refs if ts is not None]
    ref_ts = max(normalized_refs)
    elapsed_minutes = max(0.0, (now_utc - ref_ts).total_seconds() / 60.0)

    normal_anchor = _ensure_utc(stats.get("last_normal_at"))
    interaction_gap_minutes = max(0.0, (now_utc - normal_anchor).total_seconds() / 60.0) if normal_anchor else elapsed_minutes

    if not effective_has_due_alarm:
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
                "alarm_interacted_since_created": alarm_interacted_since_created,
                "stats": stats,
                **quota_fields,
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
        "has_due_alarm": effective_has_due_alarm,
        "due_alarm": due_alarm,
        "alarm_interacted_since_created": alarm_interacted_since_created,
        "stats": stats,
        **quota_fields,
    }


def _strip_thoughts_and_check_silent(raw_text: str) -> tuple[str, bool]:
    """剥离思考标签并判定是否为静默信号或空白正文。"""
    cleaned = _THOUGHT_TAG_RE.sub("", raw_text or "").strip()
    if not cleaned or _SILENT_TOKEN_RE.match(cleaned):
        return "", True
    if cleaned.upper().startswith("[SILENT]"):
        return "", True
    return cleaned, False


def _resolve_tool_executor_origin(envelope: dict | None = None) -> str:
    snap_origin = str((envelope or {}).get("tool_executor_origin") or "").strip().rstrip("/")
    if snap_origin:
        return snap_origin
    runtime_origin = str(getattr(shared, "CUPLIVO_ORIGIN", "") or "").strip().rstrip("/")
    if runtime_origin:
        return runtime_origin
    return os.getenv("CUPLIVO_ORIGIN", "").strip().rstrip("/")


async def save_heartbeat_tool_config(raw_config: dict) -> None:
    """持久化 Cuplivo Worker 随 Cron 同步过来的工具直连配置，绕开 Cloudflare Access 302。"""
    if not isinstance(raw_config, dict):
        return
    normalized = {
        "tavily_api_key": str(raw_config.get("tavily_api_key") or "").strip(),
        "mcp_tool_routes": {
            str(k).strip(): str(v).strip()
            for k, v in (raw_config.get("mcp_tool_routes") or {}).items()
            if str(k).strip() and str(v).strip()
        },
        "mcp_servers": [
            {
                "name": str(s.get("name") or "").strip(),
                "url": str(s.get("url") or "").strip(),
                "token": str(s.get("token") or "").strip(),
            }
            for s in (raw_config.get("mcp_servers") or [])
            if isinstance(s, dict) and str(s.get("url") or "").strip()
        ],
    }
    prev = getattr(shared, "HEARTBEAT_TOOL_CONFIG", None) or {}
    shared.HEARTBEAT_TOOL_CONFIG = normalized
    if shared.DATABASE_ENABLED and normalized != prev:
        try:
            await db_core.set_gateway_config(
                "heartbeat_tool_config",
                json.dumps(normalized, ensure_ascii=False),
            )
        except Exception as exc:
            logger.debug("Failed to persist heartbeat_tool_config: %s", exc)


async def get_effective_tool_config() -> dict:
    """读取后台工具直连配置（内存 -> PostgreSQL -> 环境变量）。"""
    cfg = getattr(shared, "HEARTBEAT_TOOL_CONFIG", None)
    if isinstance(cfg, dict) and (
        cfg.get("tavily_api_key") or cfg.get("mcp_tool_routes") or cfg.get("mcp_servers")
    ):
        return cfg
    if shared.DATABASE_ENABLED:
        try:
            raw = await db_core.get_gateway_config("heartbeat_tool_config", "")
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    shared.HEARTBEAT_TOOL_CONFIG = parsed
                    return parsed
        except Exception as exc:
            logger.debug("Failed to load heartbeat_tool_config: %s", exc)
    return {}


def _format_mcp_auth_header(token: str) -> str | None:
    trimmed = str(token or "").strip()
    if not trimmed:
        return None
    if trimmed.lower().startswith("bearer "):
        return trimmed
    return f"Bearer {trimmed}"


def _extract_mcp_result_text(call_result) -> str:
    content = getattr(call_result, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            txt = getattr(item, "text", None)
            if txt is not None:
                parts.append(str(txt))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(call_result, "model_dump"):
        return json.dumps(call_result.model_dump(), ensure_ascii=False)
    return str(call_result)


def _is_local_pawwake_mcp_url(server_url: str) -> bool:
    parsed = urlsplit(str(server_url or "").strip())
    path = (parsed.path or "").rstrip("/")
    if path != "/mcp":
        return False
    public_base = shared.get_public_base_url().rstrip("/") if hasattr(shared, "get_public_base_url") else ""
    if public_base and server_url.rstrip("/").startswith(public_base):
        return True
    host = (parsed.hostname or "").lower()
    return host.endswith(".onrender.com") or host in ("127.0.0.1", "localhost")


async def _call_mcp_session(server_url: str, headers: dict, *, tool_name: str | None = None, arguments: dict | None = None, list_only: bool = False, timeout: float = 20.0):
    import httpx2
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client

    async def _run_sse():
        async with sse_client(
            server_url,
            headers=headers or None,
            timeout=min(12.0, float(timeout)),
            sse_read_timeout=min(25.0, float(timeout) + 5.0),
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                if list_only:
                    listed = await session.list_tools()
                    return [t.name for t in (getattr(listed, "tools", None) or []) if getattr(t, "name", None)]
                res = await session.call_tool(tool_name, arguments=arguments or {})
                return _extract_mcp_result_text(res)

    async def _run_streamable_http():
        http_client = httpx2.AsyncClient(
            headers=headers or None,
            timeout=min(18.0, float(timeout)),
            follow_redirects=True,
        )
        async with streamable_http_client(server_url, http_client=http_client) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                if list_only:
                    listed = await session.list_tools()
                    return [t.name for t in (getattr(listed, "tools", None) or []) if getattr(t, "name", None)]
                res = await session.call_tool(tool_name, arguments=arguments or {})
                return _extract_mcp_result_text(res)

    prefer_sse = server_url.rstrip("/").lower().endswith("/sse")
    transports = (_run_sse, _run_streamable_http) if prefer_sse else (_run_streamable_http, _run_sse)
    last_err = None
    for runner in transports:
        try:
            return await runner()
        except Exception as exc:
            last_err = exc
    raise last_err


async def _execute_mcp_direct(tool_name: str, parsed_args: dict, tool_config: dict, timeout: float = 20.0) -> str | None:
    routes = tool_config.get("mcp_tool_routes") or {}
    servers = tool_config.get("mcp_servers") or []
    target_url = str(routes.get(tool_name) or "").strip()

    if not target_url and servers:
        for srv in servers:
            srv_url = str(srv.get("url") or "").strip()
            if not srv_url:
                continue
            hdrs = {}
            auth_val = _format_mcp_auth_header(srv.get("token", ""))
            if auth_val:
                hdrs["Authorization"] = auth_val
            try:
                names = await _call_mcp_session(srv_url, hdrs, list_only=True, timeout=min(12.0, timeout))
                for n in names:
                    routes[n] = srv_url
                if tool_name in names:
                    target_url = srv_url
                    break
            except Exception as exc:
                logger.debug("MCP probe failed (%s): %s", srv_url, exc)

    if not target_url:
        return None

    headers: dict[str, str] = {}
    if _is_local_pawwake_mcp_url(target_url):
        target_url = f"http://127.0.0.1:{shared.PORT}/mcp/"
        if shared.GATEWAY_SECRET:
            headers["X-Gateway-Key"] = shared.GATEWAY_SECRET
    else:
        matched = next((s for s in servers if str(s.get("url") or "").strip().rstrip("/") == target_url.rstrip("/")), None)
        if matched and matched.get("token"):
            auth_val = _format_mcp_auth_header(matched["token"])
            if auth_val:
                headers["Authorization"] = auth_val

    try:
        return await _call_mcp_session(target_url, headers, tool_name=tool_name, arguments=parsed_args, timeout=timeout)
    except Exception as exc:
        logger.warning("Direct MCP call failed (%s -> %s): %s", tool_name, target_url, exc)
        return json.dumps({"error": f"MCP 工具调用失败: {exc}"}, ensure_ascii=False)


async def _execute_tavily_direct(query: str, api_key: str, timeout: float = 20.0) -> str:
    try:
        async with httpx.AsyncClient(timeout=min(20.0, float(timeout)), follow_redirects=True) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={"query": query, "max_results": 8, "search_depth": "basic", "include_answer": True},
            )
        if resp.status_code != 200:
            return json.dumps({"error": f"Tavily HTTP {resp.status_code}: {resp.text[:200]}"}, ensure_ascii=False)
        data = resp.json()
        items = [
            {
                "index": idx + 1,
                "id": f"c{idx + 1}",
                "title": r.get("title") or "No Title",
                "url": r.get("url") or "",
                "text": r.get("content") or "",
            }
            for idx, r in enumerate(data.get("results") or [])
        ]
        payload = {"answer": data["answer"], "items": items} if data.get("answer") else {"items": items}
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"搜索请求异常: {exc}"}, ensure_ascii=False)


def _is_safe_public_url(raw_url: str) -> str:
    candidate = str(raw_url or "").strip()
    if not candidate:
        raise ValueError("缺少网页链接参数 url")
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    elif not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", candidate):
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("仅支持 http/https 公开网页链接")
    host = parsed.hostname.lower()
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith((".local", ".internal")):
        raise ValueError("不支持访问本地或内网地址")
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("不支持访问私网地址")
    except ValueError as err:
        if "不支持" in str(err):
            raise
    return parsed.geturl()


async def _execute_fetch_url_direct(raw_url: str, start_index: int, tavily_key: str, timeout: float = 20.0) -> str:
    try:
        target_url = _is_safe_public_url(raw_url)
    except ValueError as err:
        return json.dumps({"error": str(err)}, ensure_ascii=False)

    content = ""
    title = ""
    if tavily_key:
        try:
            async with httpx.AsyncClient(timeout=min(15.0, float(timeout))) as client:
                r = await client.post(
                    "https://api.tavily.com/extract",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {tavily_key}"},
                    json={"urls": [target_url]},
                )
            if r.status_code == 200:
                res_list = (r.json() or {}).get("results") or []
                if res_list and res_list[0].get("raw_content"):
                    content = str(res_list[0]["raw_content"])
        except Exception:
            pass

    if not content:
        try:
            async with httpx.AsyncClient(timeout=min(15.0, float(timeout)), follow_redirects=True) as client:
                r = await client.get(f"https://r.jina.ai/{target_url}", headers={"Accept": "text/plain"})
            if r.status_code == 200 and r.text.strip():
                content = r.text
        except Exception:
            pass

    if not content:
        try:
            async with httpx.AsyncClient(timeout=min(15.0, float(timeout)), follow_redirects=True) as client:
                r = await client.get(target_url, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                raw_html = r.text
                title_m = re.search(r"<title[^>]*>([\s\S]*?)</title>", raw_html, re.IGNORECASE)
                if title_m:
                    title = html_lib.unescape(re.sub(r"\s+", " ", title_m.group(1))).strip()
                cleaned = re.sub(r"<(script|style|noscript|svg)[^>]*>[\s\S]*?</\1>", " ", raw_html, flags=re.IGNORECASE)
                cleaned = re.sub(r"<[^>]+>", " ", cleaned)
                content = html_lib.unescape(re.sub(r"\s+", " ", cleaned)).strip()
        except Exception as exc:
            return json.dumps({"error": f"网页读取失败: {exc}"}, ensure_ascii=False)

    if not content:
        return json.dumps({"error": "未能读取到网页正文"}, ensure_ascii=False)

    page_size = 10000
    start = max(0, int(start_index or 0))
    slice_text = content[start:start + page_size]
    return json.dumps({
        "url": target_url,
        "title": title,
        "start_index": start,
        "total_length": len(content),
        "has_more": (start + len(slice_text)) < len(content),
        "content": slice_text,
    }, ensure_ascii=False)


async def execute_background_tool_call(
    tool_call: dict,
    *,
    tool_executor_origin: str = "",
    timeout: float = 20.0,
) -> str:
    """执行后台 MCP/搜索/网页阅读工具（优先从 Pawwake 直连执行以避开 Cloudflare Access 302）。"""
    fn = tool_call.get("function") if isinstance(tool_call, dict) else {}
    tool_name = str((fn or {}).get("name") or "").strip()
    raw_args = (fn or {}).get("arguments")
    if isinstance(raw_args, dict):
        parsed_args = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed_args = json.loads(raw_args or "{}")
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed_args = {"raw": raw_args}
    else:
        parsed_args = {}

    if not tool_name:
        return json.dumps({"error": "工具名缺失"}, ensure_ascii=False)

    if tool_name == "draw_secret_cards":
        return json.dumps(
            {
                "error": "后台自主唤醒通道已禁用 draw_secret_cards 工具",
                "notice": "自主唤醒期间禁止抽取或使用暗牌。",
            },
            ensure_ascii=False,
        )

    tool_config = await get_effective_tool_config()
    tavily_key = str(tool_config.get("tavily_api_key") or os.getenv("TAVILY_API_KEY") or "").strip()

    # 1. 优先直连执行（完全绕开入站 Cloudflare Access 302 登录拦截）
    if tool_name in ("search_web", "tavily_search"):
        if tavily_key:
            query = str(parsed_args.get("query") or parsed_args.get("q") or parsed_args.get("input") or "")
            res_text = await _execute_tavily_direct(query, tavily_key, timeout=timeout)
            return res_text[:24000] + ("\n...(内容过长已截断)" if len(res_text) > 24000 else "")
    elif tool_name == "fetch_url":
        target_u = str(parsed_args.get("url") or parsed_args.get("link") or "")
        start_idx = int(parsed_args.get("start_index") or 0)
        res_text = await _execute_fetch_url_direct(target_u, start_idx, tavily_key, timeout=timeout)
        return res_text[:24000] + ("\n...(内容过长已截断)" if len(res_text) > 24000 else "")
    else:
        direct_mcp_res = await _execute_mcp_direct(tool_name, parsed_args, tool_config, timeout=timeout)
        if direct_mcp_res is not None:
            return direct_mcp_res[:24000] + ("\n...(内容过长已截断)" if len(direct_mcp_res) > 24000 else "")

    # 2. 回退到 Cuplivo Worker /api/heartbeat/tool-exec（支持可选 Cloudflare Access Service Token）
    if not tool_executor_origin:
        return json.dumps(
            {
                "error": "未获取到 Cuplivo 工具执行配置",
                "notice": "当前暂无法联网或访问 MCP 工具，请直接基于现有记忆判断或保持静默。",
            },
            ensure_ascii=False,
        )

    exec_url = f"{tool_executor_origin.rstrip('/')}/api/heartbeat/tool-exec"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if shared.GATEWAY_SECRET:
        headers["X-Gateway-Key"] = shared.GATEWAY_SECRET
    cf_id = os.getenv("CF_ACCESS_CLIENT_ID", "").strip()
    cf_secret = os.getenv("CF_ACCESS_CLIENT_SECRET", "").strip()
    if cf_id and cf_secret:
        headers["CF-Access-Client-Id"] = cf_id
        headers["CF-Access-Client-Secret"] = cf_secret

    payload = {
        "tool_name": tool_name,
        "arguments": parsed_args,
        "tool_call_id": str(tool_call.get("id") or ""),
    }
    try:
        async with httpx.AsyncClient(timeout=max(5.0, float(timeout)), follow_redirects=False) as client:
            resp = await client.post(exec_url, headers=headers, json=payload)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("location", "")
            return json.dumps(
                {
                    "error": f"工具网关被重定向拦截 HTTP {resp.status_code}",
                    "location": loc[:200],
                    "notice": "对方开启了 Zero Trust 登录墙，请直连同步工具配置或稍后再试。",
                },
                ensure_ascii=False,
            )
        if resp.status_code != 200:
            err_preview = resp.text[:300] if hasattr(resp, "text") else ""
            return json.dumps(
                {"error": f"工具执行失败 HTTP {resp.status_code}", "details": err_preview},
                ensure_ascii=False,
            )
        data = resp.json()
        if isinstance(data, dict) and data.get("result_text") is not None:
            result_text = str(data["result_text"])
        else:
            result_text = json.dumps(data, ensure_ascii=False)
        if len(result_text) > 24000:
            result_text = result_text[:24000] + "\n...(内容过长已截断)"
        return result_text
    except Exception as exc:
        logger.warning("Background tool call failed (%s): %s", tool_name, exc)
        return json.dumps(
            {
                "error": f"工具执行异常: {exc}",
                "notice": "这次小工具暂时连不上，请依据当前信息自然结束思考。",
            },
            ensure_ascii=False,
        )


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
    timeout_seconds: float | None = None,
) -> dict:
    """执行一次完整的自主唤醒周期（含防重入锁、零成本门卫、缓存复用、乐观锁落库）。"""
    if _active_heartbeat_lock.locked():
        return {"status": "skipped_reentrant"}

    effective_timeout = float(timeout_seconds or HEARTBEAT_UPSTREAM_TIMEOUT)
    async with _active_heartbeat_lock:
        return await _run_heartbeat_tick_locked(
            now_utc=now_utc,
            jitter_fn=jitter_fn,
            timeout_seconds=effective_timeout,
        )


async def dispatch_heartbeat_tick(
    *,
    background: bool = False,
    now_utc: datetime | None = None,
    jitter_fn=None,
) -> dict:
    """Cron 调度入口：background=True 时门卫放行后即刻响应 200 并转后台生成，彻底规避 100s 超时。"""
    if not background:
        return await run_heartbeat_tick(now_utc=now_utc, jitter_fn=jitter_fn)

    if _active_heartbeat_lock.locked():
        return {"status": "skipped_reentrant"}
    if not shared.DATABASE_ENABLED:
        return {"status": "skipped_db_disabled"}
    if not shared.HEARTBEAT_ENABLED:
        return {"status": "skipped_disabled"}

    session_id = await resolve_target_session_id()
    gate = await evaluate_heartbeat_gate(session_id, now_utc=now_utc, jitter_fn=jitter_fn)
    if not gate.get("allowed"):
        return {"status": gate["status"], "session_id": session_id}

    async def _bg_worker():
        try:
            await run_heartbeat_tick(
                now_utc=now_utc,
                jitter_fn=jitter_fn,
                timeout_seconds=HEARTBEAT_BACKGROUND_TIMEOUT,
            )
        except Exception:
            logger.exception("Background heartbeat tick failed")

    task = asyncio.create_task(_bg_worker())
    _active_background_tasks.add(task)
    task.add_done_callback(_active_background_tasks.discard)
    return {"status": "accepted", "session_id": session_id}


def _extract_cached_tokens(usage: dict) -> int:
    if not isinstance(usage, dict):
        return 0
    pt_details = usage.get("prompt_tokens_details")
    if isinstance(pt_details, dict) and isinstance(pt_details.get("cached_tokens"), (int, float)):
        return int(pt_details["cached_tokens"])
    for key in ("cached_tokens", "cache_read_input_tokens", "prompt_cache_hit_tokens", "cached_content_token_count"):
        val = usage.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return 0


def _extract_reasoning_tokens(usage: dict) -> int:
    if not isinstance(usage, dict):
        return 0
    ct_details = usage.get("completion_tokens_details")
    if isinstance(ct_details, dict) and isinstance(ct_details.get("reasoning_tokens"), (int, float)):
        return int(ct_details["reasoning_tokens"])
    val = usage.get("reasoning_tokens")
    return int(val) if isinstance(val, (int, float)) else 0


def _aggregate_heartbeat_usage(step_usages: list[dict]) -> dict | None:
    """汇总单次自主唤醒全部步数的 token 消耗，输出与 Cuplivo 前端兼容的标准 usage 结构。"""
    if not step_usages:
        return None
    prompt_sum = 0
    cached_sum = 0
    completion_sum = 0
    reasoning_sum = 0
    total_sum = 0
    for u in step_usages:
        if not isinstance(u, dict):
            continue
        pt = int(u.get("prompt_tokens") or 0)
        ct = int(u.get("completion_tokens") or 0)
        tt = int(u.get("total_tokens") or (pt + ct))
        prompt_sum += pt
        cached_sum += _extract_cached_tokens(u)
        completion_sum += ct
        reasoning_sum += _extract_reasoning_tokens(u)
        total_sum += tt
    if prompt_sum == 0 and completion_sum == 0 and total_sum == 0:
        return None
    agg: dict = {
        "prompt_tokens": prompt_sum,
        "completion_tokens": completion_sum,
        "total_tokens": total_sum,
    }
    if cached_sum > 0:
        agg["prompt_tokens_details"] = {"cached_tokens": cached_sum}
    if reasoning_sum > 0:
        agg["completion_tokens_details"] = {"reasoning_tokens": reasoning_sum}
    return agg


async def _record_heartbeat_attempt_safe(session_id: str, now_utc: datetime | None = None) -> None:
    try:
        await db_conversations.record_heartbeat_attempt(session_id, now_utc=now_utc)
    except Exception as exc:
        logger.debug("Record heartbeat attempt skipped: %s", exc)


async def _run_heartbeat_tick_locked(
    *,
    now_utc: datetime | None = None,
    jitter_fn=None,
    timeout_seconds: float = HEARTBEAT_UPSTREAM_TIMEOUT,
) -> dict:
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
    due_alarm = gate.get("due_alarm")
    due_alarm_id = due_alarm.get("id") if isinstance(due_alarm, dict) else None
    due_alarm_reason = str(due_alarm.get("reason") or "").strip() if isinstance(due_alarm, dict) else ""

    async def _consume_due_alarm(consumed_msg_id: int | None = None):
        if due_alarm_id:
            try:
                await db_conversations.consume_wake_alarm(due_alarm_id, consumed_message_id=consumed_msg_id)
            except Exception as exc:
                logger.warning("Failed to consume wake alarm %s: %s", due_alarm_id, exc)

    envelope = await db_conversations.get_session_envelope_snapshot(session_id) or {}
    model = envelope.get("model") or shared.DEFAULT_MODEL
    if not model:
        return {"status": "skipped_no_model", "session_id": session_id}

    weather_text = await fetch_local_weather()
    snapshot_tools = chat_routes._strip_secret_cards_tool(envelope.get("tools") or [])
    tool_executor_origin = _resolve_tool_executor_origin(envelope)
    tool_config = await get_effective_tool_config()
    has_direct_tools = bool(
        tool_config.get("tavily_api_key")
        or tool_config.get("mcp_tool_routes")
        or tool_config.get("mcp_servers")
        or os.getenv("TAVILY_API_KEY")
    )
    active_topics: list[dict] = []
    try:
        active_topics = await db_conversations.get_active_proactive_topics(session_id, now_utc=gate["now_utc"])
    except Exception as exc:
        logger.debug("Load active proactive topics skipped: %s", exc)

    daily_surf_count = int(gate.get("daily_surf_count", 0))
    surf_daily_limit = int(gate.get("surf_daily_limit", getattr(shared, "HEARTBEAT_SURF_DAILY_LIMIT", 2)))
    surf_max_steps = int(gate.get("surf_max_steps", getattr(shared, "HEARTBEAT_SURF_MAX_STEPS", 5)))
    surf_quota_remaining = int(gate.get("surf_quota_remaining", max(0, surf_daily_limit - daily_surf_count)))
    effective_max_tool_steps = min(surf_max_steps, 5) if surf_quota_remaining > 0 else 1

    tail_note = build_heartbeat_tail_note(
        now_local,
        gate.get("interaction_gap_minutes", gate["elapsed_minutes"]),
        weather_text=weather_text,
        unread_count=gate["unread_proactive_chat"],
        due_alarm_reason=due_alarm_reason,
        tools_enabled=bool(snapshot_tools and (has_direct_tools or tool_executor_origin)),
        daily_surf_count=daily_surf_count,
        surf_daily_limit=surf_daily_limit,
        surf_max_steps=surf_max_steps,
        active_topics=active_topics,
        alarm_interacted_since_created=bool(gate.get("alarm_interacted_since_created")),
    )

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

    # 死守前缀缓存铁律：顶层 tools 声明与主对话 Snapshot 保持字节级一致，绝不裁剪、重排或移除
    base_req_body: dict = {
        "model": model,
        "messages": built_messages,
        "stream": False,
    }
    if snapshot_tools:
        base_req_body["tools"] = snapshot_tools

    effective_effort = envelope.get("thinking_level") or shared.REASONING_EFFORT
    if effective_effort:
        if shared.is_vertex_endpoint():
            base_req_body["google"] = {
                "thinking_config": {
                    "thinking_level": effective_effort,
                    "include_thoughts": True,
                }
            }
        else:
            base_req_body["reasoning_effort"] = effective_effort

    req_id = f"hb{int(time.time()) % 10000:04d}"
    try:
        chat_routes.diagnose_cache_prefix(
            session_id,
            base_req_body,
            req_id=req_id,
            db_history=len(db_msgs),
        )
    except Exception:
        pass

    tick_start = time.monotonic()
    deadline_mono = tick_start + timeout_seconds
    loop_messages = list(built_messages)
    tool_round = 0
    forced_wrap_up = False
    tools_used: list[str] = []
    reasoning_parts: list[str] = []
    step_usages: list[dict] = []
    final_msg_obj: dict = {}

    async def _finalize_failed_tick():
        await _record_heartbeat_attempt_safe(session_id, gate["now_utc"])
        await _consume_due_alarm(None)
        if tool_round >= 2:
            try:
                await db_conversations.increment_daily_surf_count(session_id, now_local.date())
            except Exception as exc:
                logger.debug("Increment daily surf count on failure skipped: %s", exc)

    while True:
        remaining_budget = deadline_mono - time.monotonic()
        if remaining_budget <= 2.0:
            logger.warning("Heartbeat total time budget exhausted before upstream step %s", tool_round)
            await _finalize_failed_tick()
            return {"status": "upstream_timeout", "session_id": session_id}

        if loop_messages and loop_messages[-1].get("role") == "tool" and TOOL_STEP_PACING_SECONDS > 0:
            sleep_sec = min(TOOL_STEP_PACING_SECONDS, max(0.0, remaining_budget - 10.0))
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

        req_body = dict(base_req_body)
        req_body["messages"] = loop_messages
        step_timeout = max(5.0, min(timeout_seconds, deadline_mono - time.monotonic()))

        try:
            async with httpx.AsyncClient(timeout=step_timeout) as client:
                response = await chat_routes._post_chat_completion_with_429_retry(
                    client,
                    shared.API_BASE_URL,
                    shared.API_KEY,
                    req_body,
                    tag="自主唤醒",
                    max_retries=1 if (shared.is_vertex_endpoint() and step_timeout > 45) else 0,
                    req_id=req_id,
                )
        except httpx.TimeoutException as exc:
            logger.warning("Heartbeat upstream timeout (%.1fs): %s", step_timeout, exc)
            await _finalize_failed_tick()
            return {"status": "upstream_timeout", "session_id": session_id}

        if response.status_code != 200:
            logger.warning("Heartbeat upstream HTTP %s: %s", response.status_code, response.text[:300])
            await _finalize_failed_tick()
            return {"status": "upstream_error", "code": response.status_code, "session_id": session_id}

        resp_data = response.json()
        usage = resp_data.get("usage") or {}
        if isinstance(usage, dict) and usage:
            step_usages.append(usage)
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
        step_reasoning = msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or ""
        if step_reasoning:
            reasoning_parts.append(str(step_reasoning))

        tool_calls = msg_obj.get("tool_calls") or []
        if not tool_calls:
            final_msg_obj = msg_obj
            break

        # 熔断硬拦截：已达步数上限 / 60s 安全阈值 / 强制收尾轮仍返回 tool_calls -> 绝不继续执行工具
        elapsed_so_far = time.monotonic() - tick_start
        if forced_wrap_up or tool_round >= effective_max_tool_steps or elapsed_so_far >= TOOL_TIME_BUDGET_SECONDS:
            print(f"🛑 req={req_id} 自主唤醒工具调用触达熔断红线 (round={tool_round}, max={effective_max_tool_steps}, elapsed={elapsed_so_far:.1f}s)，代码层拦截 tool_calls")
            final_msg_obj = msg_obj
            break

        # 工具执行前确认主聊天未占线、会话基线未推进
        if chat_routes.is_session_chat_busy(session_id):
            print(f"🛑 req={req_id} 后台探索前检测到主聊天占线，静默让路")
            await _consume_due_alarm(None)
            return {"status": "aborted_conflict", "reason": "chat_busy", "session_id": session_id}
        latest_meta = await db_conversations.get_latest_message_meta(session_id)
        if (latest_meta.get("id") if latest_meta else None) != baseline_msg_id:
            print(f"🛑 req={req_id} 后台探索期间会话出现新消息，静默让路")
            await _consume_due_alarm(None)
            return {"status": "aborted_conflict", "reason": "baseline_changed", "session_id": session_id}

        loop_messages.append({
            "role": "assistant",
            "content": msg_obj.get("content") or "",
            "tool_calls": tool_calls,
        })

        round_used_community_tool = False
        for tc in tool_calls:
            tc_name = str(((tc.get("function") or {}).get("name")) or "").strip()
            if tc_name:
                tools_used.append(tc_name)
                if _is_community_or_external_tool(tc_name, tool_config):
                    round_used_community_tool = True
            rem_tool_budget = max(5.0, min(20.0, TOOL_TIME_BUDGET_SECONDS - (time.monotonic() - tick_start)))
            tool_result_text = await execute_background_tool_call(
                tc,
                tool_executor_origin=tool_executor_origin,
                timeout=rem_tool_budget,
            )
            loop_messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or f"hb_tc_{len(loop_messages)}",
                "content": tool_result_text,
            })

        tool_round += 1
        if round_used_community_tool and loop_messages and loop_messages[-1].get("role") == "tool":
            cur_tool_content = loop_messages[-1].get("content") or ""
            if TONE_SANITIZER_NOTICE not in cur_tool_content:
                loop_messages[-1]["content"] = cur_tool_content + f"\n\n{TONE_SANITIZER_NOTICE}"

        elapsed_after_tools = time.monotonic() - tick_start
        if tool_round >= effective_max_tool_steps or elapsed_after_tools >= TOOL_TIME_BUDGET_SECONDS:
            forced_wrap_up = True
            loop_messages[-1]["content"] = (loop_messages[-1].get("content") or "") + f"\n\n{TOOL_FORCE_WRAP_UP_NOTICE}"
            print(f"⏳ req={req_id} 触发工具探索优雅收尾提示 (round={tool_round}/{effective_max_tool_steps}, elapsed={elapsed_after_tools:.1f}s)")

    raw_content = final_msg_obj.get("content") or ""
    # 1. 先剥离思考标签，再解析 <next_wake> 隐式闹钟与 <topic> 话题标签、再分离 <event_capsule> 与常规微信正文
    raw_without_thoughts = _THOUGHT_TAG_RE.sub("", raw_content).strip()
    text_no_wake, next_wake_spec = parse_and_strip_next_wake(raw_without_thoughts)
    text_no_topic, explicit_topic = parse_and_strip_topic(text_no_wake)
    capsules, remaining_text = parse_event_capsules_and_chat(text_no_topic)
    cleaned_chat, chat_is_silent = _strip_thoughts_and_check_silent(remaining_text)
    if final_msg_obj.get("tool_calls") and not capsules and chat_is_silent:
        print(f"🤫 req={req_id} 工具熔断后无有效正文或胶囊，静默收尾")
        await _record_heartbeat_attempt_safe(session_id, gate["now_utc"])
        await _consume_due_alarm(None)
        if next_wake_spec:
            await db_conversations.upsert_wake_alarm(
                session_id, next_wake_spec["minutes"], next_wake_spec["reason"], now_utc=gate["now_utc"]
            )
        return {"status": "silent", "reason": "tool_circuit_breaker", "session_id": session_id, "next_wake": next_wake_spec}

    if not capsules and chat_is_silent:
        print(f"💤 req={req_id} 自主唤醒选择静默继续睡（零落库）")
        await _record_heartbeat_attempt_safe(session_id, gate["now_utc"])
        await _consume_due_alarm(None)
        if next_wake_spec:
            await db_conversations.upsert_wake_alarm(
                session_id, next_wake_spec["minutes"], next_wake_spec["reason"], now_utc=gate["now_utc"]
            )
        return {"status": "silent", "reason": "model_silent", "session_id": session_id, "next_wake": next_wake_spec}

    # 落库前二次校验并发占线与最新 message_id 乐观锁
    if chat_routes.is_session_chat_busy(session_id):
        print(f"🛑 req={req_id} 唤醒落库前检测到主聊天占线，静默让路")
        await _consume_due_alarm(None)
        return {"status": "aborted_conflict", "reason": "chat_busy", "session_id": session_id}

    latest_meta = await db_conversations.get_latest_message_meta(session_id)
    latest_id = latest_meta.get("id") if latest_meta else None
    if latest_id != baseline_msg_id:
        print(f"🛑 req={req_id} 唤醒期间会话出现新消息 ({baseline_msg_id} -> {latest_id})，静默丢弃结果")
        await _consume_due_alarm(None)
        return {"status": "aborted_conflict", "reason": "baseline_changed", "session_id": session_id}

    combined_reasoning = "\n\n---\n\n".join(r for r in reasoning_parts if r) or None
    agg_usage = _aggregate_heartbeat_usage(step_usages)
    elapsed_ms = max(0, round((time.monotonic() - tick_start) * 1000))
    timing_obj: dict = {"total_ms": elapsed_ms}
    if effective_effort:
        timing_obj["thinking_level"] = effective_effort

    meta_dict: dict = {}
    if combined_reasoning:
        meta_dict["reasoning_content"] = combined_reasoning
    if agg_usage:
        meta_dict["usage"] = agg_usage
    if timing_obj:
        meta_dict["timing"] = timing_obj
    meta_str = json.dumps(meta_dict, ensure_ascii=False) if meta_dict else None
    base_proactive_meta: dict = {
        "trigger_source": "alarm" if gate.get("has_due_alarm") else "cron",
        "weather": weather_text,
        "context_kind": gate.get("context_kind", "normal"),
    }
    if due_alarm_reason:
        base_proactive_meta["due_alarm_reason"] = due_alarm_reason
    if tools_used:
        base_proactive_meta["tools_used"] = list(dict.fromkeys(tools_used))

    saved_event_ids: list[int] = []
    for cap in capsules:
        ev_meta = {**base_proactive_meta, "event_title": cap["title"]}
        ev_id = await db_conversations.save_message(
            session_id,
            "assistant",
            cap["body"],
            model=model,
            metadata=meta_str if (not cleaned_chat or chat_is_silent) and not saved_event_ids else None,
            message_type="proactive_event",
            proactive_meta=ev_meta,
            skip_memory_extract=True,
            memory_extracted=False,
        )
        if ev_id is not None:
            saved_event_ids.append(ev_id)

    saved_chat_id: int | None = None
    if cleaned_chat and not chat_is_silent:
        saved_chat_id = await db_conversations.save_message(
            session_id,
            "assistant",
            cleaned_chat,
            model=model,
            metadata=meta_str,
            message_type="proactive_chat",
            proactive_meta=base_proactive_meta,
            skip_memory_extract=False,
            memory_extracted=False,
        )

    primary_msg_id = saved_chat_id or (saved_event_ids[-1] if saved_event_ids else None)
    await _record_heartbeat_attempt_safe(session_id, gate["now_utc"])
    await _consume_due_alarm(primary_msg_id)
    if next_wake_spec:
        await db_conversations.upsert_wake_alarm(
            session_id,
            next_wake_spec["minutes"],
            next_wake_spec["reason"],
            source_message_id=primary_msg_id,
            now_utc=gate["now_utc"],
        )

    did_deep_surf = (tool_round >= 2) or (bool(tools_used) and bool(capsules))
    if did_deep_surf:
        try:
            await db_conversations.increment_daily_surf_count(session_id, now_local.date())
        except Exception as exc:
            logger.debug("Increment daily surf count skipped: %s", exc)

    topic_key, topic_summary = derive_proactive_topic(capsules, cleaned_chat, explicit_topic)
    if topic_key:
        try:
            await db_conversations.upsert_proactive_topic(
                session_id,
                topic_key,
                topic_summary,
                now_utc=gate["now_utc"],
            )
        except Exception as exc:
            logger.debug("Upsert proactive topic skipped: %s", exc)

    if saved_chat_id is not None:
        print(
            f"💌 req={req_id} 自主唤醒落库成功: chat_id={saved_chat_id}, events={saved_event_ids}, "
            f"session={session_id}, tools={len(tools_used)}"
        )
        return {
            "status": "delivered",
            "session_id": session_id,
            "message_id": saved_chat_id,
            "event_ids": saved_event_ids,
            "message_type": "proactive_chat",
            "tools_used": base_proactive_meta.get("tools_used", []),
            "next_wake": next_wake_spec,
            "topic": topic_key,
            "deep_surf_charged": did_deep_surf,
        }

    print(
        f"📓 req={req_id} 自主唤醒记录事件胶囊: event_ids={saved_event_ids}, "
        f"session={session_id}, tools={len(tools_used)}"
    )
    return {
        "status": "event_recorded",
        "session_id": session_id,
        "event_ids": saved_event_ids,
        "message_type": "proactive_event",
        "tools_used": base_proactive_meta.get("tools_used", []),
        "next_wake": next_wake_spec,
        "topic": topic_key,
        "deep_surf_charged": did_deep_surf,
    }
