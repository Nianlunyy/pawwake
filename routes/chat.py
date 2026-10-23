"""Core OpenAI-compatible chat routes."""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import uuid

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

import shared
import partition_engine
import memory_pipeline
from db import core as db_core
from db import search as db_search
from db import conversations as db_conversations
from db import memories as db_memories

router = APIRouter()
logger = logging.getLogger(__name__)

FALLBACK_PHASE1_TIMEOUT = 60
FALLBACK_PHASE2_TIMEOUT = 90
UPSTREAM_KEEPALIVE_INTERVAL = float(os.getenv("UPSTREAM_KEEPALIVE_INTERVAL", "15.0"))
UPSTREAM_WAIT_CAP = float(os.getenv("UPSTREAM_WAIT_CAP", "240.0"))


def _format_seconds(val: float) -> str:
    try:
        if isinstance(val, (int, float)):
            if val >= 0.95 and abs(val - round(val)) < 0.05:
                return str(int(round(val)))
            r1 = round(val, 1)
            if abs(val - r1) < 0.02:
                return f"{r1:g}"
            r2 = round(val, 2)
            return f"{r2:g}"
    except Exception:
        pass
    return str(val)


def _get_keepalive_interval() -> float:
    try:
        env_val = os.getenv("UPSTREAM_KEEPALIVE_INTERVAL")
        if env_val is not None:
            return float(env_val)
    except Exception:
        pass
    return float(UPSTREAM_KEEPALIVE_INTERVAL)


def _get_wait_cap() -> float:
    try:
        env_val = os.getenv("UPSTREAM_WAIT_CAP")
        if env_val is not None:
            return float(env_val)
    except Exception:
        pass
    return float(UPSTREAM_WAIT_CAP)



class _ReminderLeaseKeeper:
    """发送期间延长到期提醒的处理时限。

    停止时先等正在更新的时间落库，避免结算使用旧时间而无法标记送达。
    """

    def __init__(self, pending_reminder_claims):
        self.claims = pending_reminder_claims
        self._task = None
        self._inflight = None

    def start(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def _renew_once(self):
        renewed = await db_memories.renew_reminder_claims(list(self.claims))
        self.claims[:] = renewed

    async def _loop(self):
        while True:
            await asyncio.sleep(memory_pipeline.REMINDER_LEASE_RENEW_SECONDS)
            self._inflight = asyncio.ensure_future(self._renew_once())
            try:
                await asyncio.shield(self._inflight)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"⚠️ 提醒处理时限延长失败: {e}")

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        inflight = self._inflight
        if inflight is not None and not inflight.done():
            try:
                await inflight
            except Exception as e:
                print(f"⚠️ 提醒处理时限延长失败: {e}")

    @property
    def done(self) -> bool:
        return self._task is None or self._task.done()


def _start_reminder_lease_keeper(pending_reminder_claims):
    """没有待处理提醒时不启动延时。"""
    if not pending_reminder_claims:
        return None
    return _ReminderLeaseKeeper(pending_reminder_claims).start()


async def _stop_reminder_lease_keeper(keeper):
    """停稳延时循环，确保结算拿到最新的待处理提醒。"""
    if keeper is not None:
        await keeper.stop()


async def _release_reminder_claims(pending_reminder_claims):
    """上游失败时交回到期提醒，让下次请求立即重试。"""
    if pending_reminder_claims:
        try:
            await db_memories.release_reminder_claims(list(pending_reminder_claims))
            pending_reminder_claims.clear()
        except Exception as e:
            print(f"⚠️ 提醒处理权交回失败，等处理时限结束后自动重试: {e}")


async def _settle_reminder_claims(lease_keeper, pending_reminder_claims, delivered_ok: bool):
    """请求结束时标记已送达提醒；未完整送达的交回给下次请求。"""
    await _stop_reminder_lease_keeper(lease_keeper)
    if not pending_reminder_claims:
        return
    if delivered_ok:
        try:
            delivered = await db_memories.mark_reminders_delivered(list(pending_reminder_claims))
        except Exception as e:
            print(f"⚠️ 提醒送达标记失败，交回后等下次请求重试: {e}")
            delivered = 0
        print(f"⏰ 提醒送达 {delivered}/{len(pending_reminder_claims)} 条")
        if delivered == len(pending_reminder_claims):
            pending_reminder_claims.clear()
            return
        print("⚠️ 提醒送达数不足，未送达项交回等待重试")
    await _release_reminder_claims(pending_reminder_claims)


async def _close_stream_and_release(stream, pending_reminder_claims):
    """响应任务退出后（自然结束、客户端断开、建好响应却没开始消费）显式关掉生成器，
    让它的 finally 立即运行；生成器未启动或收尾被取消时，这里交回剩余提醒。
    自然结束时两步都是空操作。"""
    try:
        await stream.aclose()
    except Exception as e:
        print(f"⚠️ 关闭流式生成器失败: {e}")
    finally:
        await _release_reminder_claims(pending_reminder_claims)


def _streaming_response(stream, pending_reminder_claims) -> StreamingResponse:
    """流式响应统一在这里建。Starlette 收到 http.disconnect 只取消发送任务，不会替我们关 body iterator，
    所以挂 BackgroundTask 在响应任务退出后显式 aclose 并交回剩余提醒。"""
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        background=BackgroundTask(_close_stream_and_release, stream, pending_reminder_claims),
    )


async def _mark_pending_seen(session_id, pending_fragment_ids, pending_memory_ids):
    if pending_fragment_ids:
        try:
            await db_conversations.mark_fragments_seen(
                session_id,
                pending_fragment_ids,
                shared.CONVERSATION_SEEN_TTL_HOURS,
            )
        except Exception as e:
            print(f"⚠️ 对话召回 seen 写入失败，保留待下次重试: {e}")

    if pending_memory_ids:
        try:
            await db_conversations.mark_memories_seen(
                session_id,
                pending_memory_ids,
                shared.MEMORY_SEEN_TTL_HOURS,
            )
        except Exception as e:
            print(f"⚠️ 记忆注入 seen 写入失败，保留待下次重试: {e}")

# ============================================================
# API 接口
# ============================================================

@router.get("/")
async def health_check():
    """健康检查"""
    resolved_system_prompt = await shared.get_system_prompt()
    memory_count = 0
    if shared.MEMORY_ENABLED:
        try:
            memory_count = await db_memories.get_all_memories_count()
        except:
            pass

    return {
        "status": "running",
        "gateway": "Pawwake v4.1.10",
        "system_prompt_loaded": len(resolved_system_prompt) > 0,
        "system_prompt_length": len(resolved_system_prompt),
        "database_enabled": shared.DATABASE_ENABLED,
        "conversation_persistence_enabled": shared.conversation_persistence_enabled(),
        "memory_enabled": shared.MEMORY_ENABLED,
        "memory_source_dedupe_enabled": shared.MEMORY_SOURCE_DEDUPE_ENABLED,
        "cache_partition_enabled": shared.CACHE_PARTITION_ENABLED,
        "conversation_recall_enabled": shared.CONVERSATION_RECALL_ENABLED,
        "memory_count": memory_count,
        "memory_extract_interval": shared.MEMORY_EXTRACT_INTERVAL,
    }


@router.get("/v1/models")
async def list_models():
    """模型列表（让客户端不报错）"""
    models = []
    if shared.DEFAULT_MODEL:
        models.append({
            "id": shared.DEFAULT_MODEL,
            "object": "model",
            "created": 1700000000,
            "owned_by": "pawwake",
        })
    return {
        "object": "list",
        "data": models,
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """核心转发接口"""
    if not shared.API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "API_KEY 未设置，请先在 Dashboard 的基础连接中配置"},
        )

    # 到期提醒由这条路由负责兜底：里层抛异常或非流式返回时仍未结清，就在这里交回；
    # 流式由生成器的 finally 结算
    pending_reminder_claims = []
    try:
        response = await _chat_completions_inner(request, pending_reminder_claims)
    except Exception:
        logger.exception("Chat completion gateway failed")
        await _release_reminder_claims(pending_reminder_claims)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Gateway internal error", "type": "gateway_error"}},
        )
    if pending_reminder_claims and not isinstance(response, StreamingResponse):
        await _release_reminder_claims(pending_reminder_claims)
    return response


# ============================================================
# 缓存前缀诊断与 Usage 日志（进程内保留，不入库）
# ============================================================

_LAST_REQUEST_CACHE: dict[str, dict] = {}


def _normalize_message_for_hash(msg: dict) -> tuple[str, str, str, int]:
    """Returns (role, normalized_text, tool_calls_json, char_len)."""
    if not isinstance(msg, dict):
        s = str(msg)
        return ("", s, "", len(s))
    role = str(msg.get("role") or "")
    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    try:
                        parts.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
                    except Exception:
                        parts.append(str(item))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        text = " ".join(parts)
    elif content is None:
        text = ""
    else:
        text = str(content)

    tc = msg.get("tool_calls")
    if tc:
        try:
            tc_str = json.dumps(tc, sort_keys=True, ensure_ascii=False)
        except Exception:
            tc_str = str(tc)
    else:
        tc_str = ""

    return (role, text, tc_str, len(text))


def _compute_message_hashes(messages: list) -> tuple[list[str], list[int], list[tuple[str, str]]]:
    import hashlib
    hash_list = []
    char_lens = []
    meta_list = []
    prev_h = ""
    if isinstance(messages, list):
        for msg in messages:
            role, text, tc_str, c_len = _normalize_message_for_hash(msg)
            char_lens.append(c_len)
            snippet = text.replace("\r", " ").replace("\n", " ").strip()[:40]
            meta_list.append((role, snippet))

            raw_repr = f"{prev_h}|{role}|{text}|{tc_str}"
            h = hashlib.sha256(raw_repr.encode("utf-8", errors="replace")).hexdigest()[:16]
            hash_list.append(h)
            prev_h = h
    return hash_list, char_lens, meta_list


def _compute_tools_hash(tools) -> str:
    import hashlib
    if not tools:
        return "empty"
    try:
        tools_str = json.dumps(tools, sort_keys=True, ensure_ascii=False)
    except Exception:
        tools_str = str(tools)
    return hashlib.sha256(tools_str.encode("utf-8", errors="replace")).hexdigest()[:16]


def diagnose_cache_prefix(
    session_id: str,
    body: dict,
    req_id: str = "",
    db_history: int = 0,
    client_sys: str = "",
) -> dict:
    """对即将发往模型的 messages 和 tools 计算哈希并与上一请求对比。整个包在 try/except 里。"""
    try:
        if not isinstance(body, dict):
            return {}

        diag_sid = str(body.get("session_id") or session_id or "default")
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        model = str(body.get("model") or "")
        tools = body.get("tools")
        rid = req_id or str(body.get("req_id") or "")
        req_prefix = f"req={rid} " if rid else ""

        if not client_sys and isinstance(messages, list):
            try:
                sys_text = partition_engine._extract_client_system_text(messages)
                client_sys = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()[:8] if sys_text else "none"
            except Exception:
                client_sys = "none"
        if not client_sys:
            client_sys = "none"

        curr_hash_list, curr_char_lens, curr_meta = _compute_message_hashes(messages)
        curr_tools_hash = _compute_tools_hash(tools)
        now_ts = time.time()

        prev_record = _LAST_REQUEST_CACHE.get(diag_sid)
        result_meta = {
            "session_id": diag_sid,
            "round": 1,
            "db_history": db_history,
            "client_sys": client_sys,
            "model": model,
            "tools_same": True,
            "common_prefix_count": 0,
            "total_messages": len(curr_hash_list),
            "common_chars": 0,
            "total_chars": sum(curr_char_lens),
            "pct": 0.0,
            "diff_idx": None,
            "is_first": True,
        }

        if prev_record is None:
            round_num = 1
            print(f"🧩 {req_prefix}前缀对比: round={round_num} db_history={db_history} client_sys={client_sys} model={model} | 无上一请求", flush=True)
            result_meta["round"] = round_num
        else:
            round_num = prev_record.get("round", 1) + 1
            result_meta["round"] = round_num
            result_meta["is_first"] = False

            last_time = prev_record.get("time", now_ts)
            elapsed = max(0.0, now_ts - last_time)
            elapsed_str = f"{elapsed:.1f}"

            prev_tools_hash = prev_record.get("tools_hash")
            tools_same = (curr_tools_hash == prev_tools_hash)
            tools_match_str = "Y" if tools_same else "N"
            result_meta["tools_same"] = tools_same

            prev_hash_list = prev_record.get("hash_list", [])
            n = len(curr_hash_list)
            k = 0
            min_len = min(n, len(prev_hash_list))
            while k < min_len and curr_hash_list[k] == prev_hash_list[k]:
                k += 1

            result_meta["common_prefix_count"] = k
            common_chars = sum(curr_char_lens[:k])
            total_chars = sum(curr_char_lens)
            result_meta["common_chars"] = common_chars
            result_meta["total_chars"] = total_chars

            pct = (common_chars / total_chars * 100) if total_chars > 0 else 0.0
            result_meta["pct"] = pct
            pct_str = f"{pct:.1f}%" if (pct % 1 != 0 and pct > 0) else f"{int(pct)}%"

            if k < n:
                diff_idx = k
                diff_role, diff_snippet = curr_meta[k]
                diff_str = f"首个差异 idx={diff_idx}, role={diff_role}, 片段={diff_snippet}"
                result_meta["diff_idx"] = diff_idx
                result_meta["diff_role"] = diff_role
                result_meta["diff_snippet"] = diff_snippet
            elif k < len(prev_hash_list):
                diff_idx = k
                diff_str = f"首个差异 idx={diff_idx}, role=(本次缩短), 片段=无本次消息"
                result_meta["diff_idx"] = diff_idx
            else:
                diff_str = "首个差异 无"
                result_meta["diff_idx"] = None

            print(
                f"🧩 {req_prefix}前缀对比: round={round_num} db_history={db_history} client_sys={client_sys} model={model} 距上次={elapsed_str}秒 | tools同={tools_match_str} | "
                f"共同前缀 {k}/{n} 条（约 {common_chars} 字符，占本次 {pct_str}）| {diff_str}",
                flush=True,
            )

        # 限制缓存容量避免内存泄露
        if len(_LAST_REQUEST_CACHE) > 500:
            oldest_key = next(iter(_LAST_REQUEST_CACHE))
            _LAST_REQUEST_CACHE.pop(oldest_key, None)

        _LAST_REQUEST_CACHE[diag_sid] = {
            "time": now_ts,
            "model": model,
            "tools_hash": curr_tools_hash,
            "hash_list": curr_hash_list,
            "char_lens": curr_char_lens,
            "meta": curr_meta,
            "round": round_num,
            "db_history": db_history,
            "client_sys": client_sys,
        }
        return result_meta
    except Exception as e:
        logger.warning("Cache prefix diagnostic failed: %s", e)
        return {}


def log_cache_usage(usage: dict, req_id: str = ""):
    """收到 usage 后打印缓存统计。没有字段按 0。整个包在 try/except 里。"""
    try:
        cached_tokens = 0
        prompt_tokens = 0
        rid = req_id or (usage.get("req_id") if isinstance(usage, dict) else "")
        req_prefix = f"req={rid} " if rid else ""

        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            if pt is not None and isinstance(pt, (int, float)):
                prompt_tokens = int(pt)

            pt_details = usage.get("prompt_tokens_details")
            if isinstance(pt_details, dict):
                ct = pt_details.get("cached_tokens")
                if ct is not None and isinstance(ct, (int, float)):
                    cached_tokens = int(ct)

            if cached_tokens == 0:
                ct = usage.get("cached_tokens")
                if ct is not None and isinstance(ct, (int, float)):
                    cached_tokens = int(ct)
                elif usage.get("cache_read_input_tokens") is not None:
                    try:
                        cached_tokens = int(usage.get("cache_read_input_tokens") or 0)
                    except Exception:
                        pass
                elif usage.get("prompt_cache_hit_tokens") is not None:
                    try:
                        cached_tokens = int(usage.get("prompt_cache_hit_tokens") or 0)
                    except Exception:
                        pass
                elif usage.get("cached_content_token_count") is not None:
                    try:
                        cached_tokens = int(usage.get("cached_content_token_count") or 0)
                    except Exception:
                        pass

        if prompt_tokens > 0:
            pct = (cached_tokens / prompt_tokens) * 100
            pct_str = f"{pct:.1f}%" if (pct % 1 != 0 and pct > 0) else f"{int(pct)}%"
        else:
            pct_str = "0%"

        print(f"🧮 {req_prefix}缓存: cached={cached_tokens} / prompt={prompt_tokens} ({pct_str})", flush=True)
    except Exception as e:
        logger.warning("Cache usage diagnostic failed: %s", e)


async def _chat_completions_inner(request: Request, pending_reminder_claims: list = None):
    body = await request.json()
    req_id = str(body.get("req_id") or request.headers.get("X-Request-ID") or uuid.uuid4().hex[:6])[:6]
    body.pop("req_id", None)
    messages = body.get("messages", [])
    try:
        client_sys_text = partition_engine._extract_client_system_text(messages)
        client_sys = hashlib.sha256(client_sys_text.encode("utf-8")).hexdigest()[:8] if client_sys_text else "none"
    except Exception:
        client_sys = "none"
    model = body.get("model") or shared.DEFAULT_MODEL
    if not model:
        return JSONResponse(
            status_code=400,
            content={"error": "模型未设置，请在请求中指定 model，或先在 Dashboard 设置默认模型"},
        )
    body["model"] = model
    pending_fragment_ids = []
    pending_memory_ids = []
    if pending_reminder_claims is None:
        pending_reminder_claims = []

    # ---------- 检测是否应跳过对话存储 ----------
    # 优先尊重客户端显式声明；无法加 header 的客户端则识别其标题生成模板。
    explicit_skip = request.headers.get("X-Skip-Conversation-Log", "").lower() == "true"
    auxiliary_title_request = partition_engine._is_title_generation_request(messages)
    skip_conversation_log = explicit_skip or auxiliary_title_request
    if auxiliary_title_request:
        print("⏭️  检测到标题生成请求：跳过分区缓存、记忆注入、对话存储和会话 Token 统计")

    # ---------- 提取用户最新消息 ----------
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            break

    # ---------- 构建 system prompt ----------
    # 先保存原始对话消息（不含 system prompt），用于记忆提取
    original_messages = [dict(msg) for msg in messages if msg.get("role") != "system"]
    extraction_context_messages = original_messages
    extraction_round_count = None
    resolved_system_prompt = "" if skip_conversation_log else await shared.get_system_prompt()

    # ---------- 检测工具调用消息 ----------
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    if tool_messages:
        print(f"🔧 检测到 {len(tool_messages)} 条工具结果消息")

    # ---------- 生成 session ID ----------
    session_id = str(uuid.uuid4())[:8]

    # ---------- 分区缓存模式 ----------
    db_history_count = 0
    if shared.CACHE_PARTITION_ENABLED and not skip_conversation_log:
        active_sid = shared.get_active_session_id()
        if active_sid:
            session_id = active_sid

        # 从DB读取历史
        try:
            db_history = await db_conversations.get_conversation_messages(session_id, limit=10000)
            db_msgs = []
            recall_history = []
            db_extraction_msgs = []
            for m in (db_history or []):
                msg = db_conversations.db_row_to_message(m)
                msg['created_at'] = m.get('created_at')  # 保留时间戳供分区时间窗口判断
                db_msgs.append(msg)
                recall_history.append({**msg, "id": m.get("id")})
                db_extraction_msgs.append({**msg, "_source_message_id": m.get("id")})
            db_history_count = len(db_msgs)
        except Exception as e:
            print(f"❌ 分区缓存不可用：读取对话历史失败: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "分区缓存需要数据库，当前无法读取对话历史。请检查数据库连接或暂时关闭 CACHE_PARTITION_ENABLED。",
                        "type": "partition_database_unavailable",
                    }
                },
            )

        # 提取客户端新消息（非system），可能是user、tool、或带tool_calls的assistant
        client_new_msgs = [m for m in messages if m.get("role") != "system"]
        # 分区模式下，assistant消息来自上一轮response（DB里已存），过滤掉避免重复
        client_new_msgs = [m for m in client_new_msgs if m.get("role") != "assistant"]
        # 分区模式下DB已有完整历史，客户端发来的旧user是冗余的。
        # 但有些客户端把图片和文字拆成多条连续的user发送（图在前文字在后），
        # 只留最后一条会把图那条当冗余丢掉（图片不入库，DB里也找不回来）。
        # 所以按原始消息顺序保留"末尾连续的user块"：历史冗余user总是被assistant隔开，不会混入。
        tail_user_ids = set()
        for m in reversed([m for m in messages if m.get("role") != "system"]):
            if m.get("role") == "user":
                tail_user_ids.add(id(m))
            else:
                break
        user_msgs = [m for m in client_new_msgs if m.get("role") == "user"]
        if len(user_msgs) > len(tail_user_ids):
            client_new_msgs = [
                m for m in client_new_msgs
                if m.get("role") != "user" or id(m) in tail_user_ids
            ]
            print(f"🔧 去重: 过滤{len(user_msgs)-len(tail_user_ids)}条冗余user，保留末尾连续{len(tail_user_ids)}条")
        # 工具结果轮次处理：基于DB状态 + 当前轮次tool_call_id精确判断
        client_tools = [m for m in client_new_msgs if m.get("role") == "tool"]
        if client_tools:
            # 判断DB是否处于"等待tool结果"状态（最后一条是assistant(tool_calls)）
            db_last = db_msgs[-1] if db_msgs else None
            db_expecting_tool = (db_last and db_last.get("role") == "assistant" and db_last.get("tool_calls"))

            if not db_expecting_tool:
                # DB不在等待tool结果 → 客户端的所有tool都是历史残留（含手动删除后的幽灵）
                stale_ids = [m.get('tool_call_id', '?') for m in client_tools]
                print(f"🔧 去重: DB未在等待tool结果，丢弃{len(client_tools)}条客户端tool (ids: {stale_ids})")
                client_new_msgs = [m for m in client_new_msgs if m.get("role") != "tool"]
            else:
                # DB在等待tool → 只保留匹配当前轮次assistant(tool_calls)的tool
                expected_tool_ids = {tc.get("id") for tc in db_last.get("tool_calls", []) if tc.get("id")}
                new_tools = [m for m in client_tools if m.get("tool_call_id") in expected_tool_ids]
                stale_tools = [m for m in client_tools if m.get("tool_call_id") not in expected_tool_ids]

                if stale_tools:
                    print(f"🔧 去重: 丢弃{len(stale_tools)}条非当前轮次tool (ids: {[m.get('tool_call_id','?') for m in stale_tools]})")
                if new_tools:
                    print(f"🔧 保留{len(new_tools)}条当前轮次tool (ids: {[m.get('tool_call_id','?') for m in new_tools]})")

                # 重建 client_new_msgs（user此时已只剩末尾连续块，全部保回，别把拆条发送的图丢了）
                tail_users = [m for m in client_new_msgs if m.get("role") == "user"]
                client_new_msgs = new_tools[:] + tail_users
        all_msgs = db_msgs + client_new_msgs
        extraction_context_messages = db_extraction_msgs + [dict(m) for m in client_new_msgs]
        extraction_round_count = len(partition_engine.group_by_rounds(all_msgs))

        # 同步更新tool_messages，避免process_memories_background存重复的旧tool
        tool_messages = [m for m in client_new_msgs if m.get("role") == "tool"]

        print(f"📦 分区模式: DB历史{len(db_msgs)}条 + 客户端消息{len(client_new_msgs)}条")

        partition_prompt = resolved_system_prompt
        if shared.memory_injection_enabled():
            partition_prompt = (resolved_system_prompt or "") + partition_engine.MEMORY_USAGE_GUIDE
        # 保留客户端自带的 system（工具说明等），拼接到网关 prompt 之后，
        # 与非分区路径的行为对齐（前端 system 稳定时不影响 BP1 缓存命中）
        client_system_text = partition_engine._extract_client_system_text(messages)
        if client_system_text:
            partition_prompt = ((partition_prompt or "") + "\n\n" + client_system_text).strip()
        try:
            recall_result = await memory_pipeline.build_conversation_recall_text(
                user_message,
                session_id,
                recall_history,
            )
            conversation_recall_text, pending_fragment_ids = recall_result[:2]
            recalled_message_ids = recall_result[2] if len(recall_result) > 2 else []

            async def build_session_memory_text(message: str):
                return await memory_pipeline.build_memory_text(
                    message,
                    session_id,
                    pending_memory_ids,
                    pending_reminder_claims,
                    recalled_message_ids,
                )

            messages = await partition_engine.build_partitioned_messages(
                session_id,
                all_msgs,
                partition_prompt,
                user_message,
                conversation_recall_text,
                build_session_memory_text,
            )
        except Exception as e:
            print(f"❌ 分区缓存不可用：读取轮转状态失败: {e}")
            await _release_reminder_claims(pending_reminder_claims)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "分区缓存需要数据库，当前无法读取轮转状态。请检查数据库连接或暂时关闭 CACHE_PARTITION_ENABLED。",
                        "type": "partition_database_unavailable",
                    }
                },
            )
        body["messages"] = messages

    else:
        # ---------- 原有逻辑：system prompt + 记忆注入 ----------
        if not skip_conversation_log and (resolved_system_prompt or (shared.MEMORY_ENABLED and user_message)):
            if shared.MEMORY_ENABLED and user_message:
                enhanced_prompt = await memory_pipeline.build_system_prompt_with_memories(
                    user_message,
                    resolved_system_prompt,
                    pending_reminder_claims,
                )
            else:
                enhanced_prompt = resolved_system_prompt

            if enhanced_prompt:
                has_system = any(msg.get("role") == "system" for msg in messages)
                if has_system:
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system":
                            messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                            break
                else:
                    messages.insert(0, {"role": "system", "content": enhanced_prompt})

        body["messages"] = messages

    # ---------- cache_control 兼容性处理 ----------
    if shared.CACHE_PARTITION_ENABLED and not partition_engine._is_anthropic_model(model):
        partition_engine._strip_cache_control(body.get("messages", []))

    # ---------- 转发请求 ----------
    headers = shared.build_chat_completion_headers(shared.API_BASE_URL, shared.API_KEY)
    if shared.is_vertex_endpoint():
        try:
            fresh_token = shared.get_vertex_access_token()
            headers["Authorization"] = f"Bearer {fresh_token}"
            print(f"🔑 Vertex Token 已注入, 前缀: {fresh_token[:20]}..., 长度: {len(fresh_token)}")
        except Exception as e:
            print(f"⚠️ 动态生成 Vertex Token 失败: {e}")
            from fastapi import HTTPException
            raise HTTPException(status_code=502, detail=f"Vertex AI 认证令牌获取失败: {str(e)}")
        if "/" not in body.get("model", ""):
            body["model"] = f"google/{body['model']}"
            print(f"🏷️ 已补全 Vertex publisher 前缀: model={body['model']}")

    is_stream = body.get("stream", False)

    # 强制流式传输（解决部分客户端不发stream=true的问题）
    if shared.FORCE_STREAM and not is_stream:
        is_stream = True
        body["stream"] = True
        print(f"⚡ 强制开启流式传输（FORCE_STREAM=true）")

    # 注入推理参数（解决客户端走网关时不带reasoning参数的问题）
    effective_effort = shared.REASONING_EFFORT
    effort_source = "default"
    req_prefix = f"req={req_id} " if req_id else ""
    try:
        raw_header = request.headers.get("X-Thinking-Level")
        if raw_header is not None:
            cleaned_header = raw_header.strip().lower()
            if cleaned_header in {"minimal", "low", "medium", "high"}:
                effective_effort = cleaned_header
                effort_source = "header"
            else:
                try:
                    logger.warning(f"⚠️ {req_prefix}忽略非法 X-Thinking-Level 头值: {raw_header!r}，沿用默认: {shared.REASONING_EFFORT}")
                except Exception:
                    pass
    except Exception as e:
        effective_effort = shared.REASONING_EFFORT
        effort_source = "default"
        try:
            logger.warning(f"⚠️ {req_prefix}读取 X-Thinking-Level 头异常: {e}，沿用默认")
        except Exception:
            pass

    if effective_effort and not skip_conversation_log:
        body.pop("reasoning_effort", None)
        body.pop("google", None)
        if shared.is_vertex_endpoint():
            body["google"] = {
                "thinking_config": {
                    "thinking_level": effective_effort,
                    "include_thoughts": True
                }
            }
            try:
                print(f"🧠 {req_prefix}注入推理参数(Gemini): thinking_level={effective_effort}, include_thoughts=True (source={effort_source})")
            except Exception:
                pass
        else:
            body["reasoning_effort"] = effective_effort
            try:
                print(f"🧠 {req_prefix}注入推理参数: reasoning_effort={effective_effort} (source={effort_source})")
            except Exception:
                pass

    req_prefix = f"req={req_id} " if req_id else ""
    print(f"📡 {req_prefix}请求: model={model}, stream={is_stream}, memory={'on' if shared.MEMORY_ENABLED else 'off'}", flush=True)

    # 调试：打印请求体中的推理相关字段
    debug_keys = {k: v for k, v in body.items() if k in ('reasoning_effort', 'google', 'reasoning')}
    if debug_keys:
        print(f"📡 推理字段: {debug_keys}", flush=True)

    # =========================================================================
    # [Gemini/Vertex AI 专用] 工具返回缓冲防 429 START
    # 仅在上游是 Gemini/Vertex 且发送的 messages 末尾角色为 tool 时生效；
    # 普通用户对白（user）严格不加任何延迟。（若无此问题可整段剥离）
    # =========================================================================
    _final_msgs = body.get("messages", [])
    if (shared.is_vertex_endpoint()
            and _final_msgs
            and _final_msgs[-1].get("role") == "tool"):
        print("⏳ Gemini/Vertex tool 返回缓冲: sleep 1.5s 防 429")
        await asyncio.sleep(1.5)
    # =========================================================================
    # [Gemini/Vertex AI 专用] 工具返回缓冲防 429 END
    # =========================================================================

    # 前缀诊断日志（仅主聊天请求，不改变任何行为，严格异常拦截）
    try:
        diagnose_cache_prefix(
            session_id,
            body,
            req_id=req_id,
            db_history=db_history_count,
            client_sys=client_sys,
        )
    except Exception as e:
        logger.warning("Cache prefix diagnostic failed: %s", e)

    if is_stream:
        stream = stream_and_capture(
            headers,
            body,
            session_id,
            user_message,
            model,
            extraction_context_messages,
            skip_conversation_log,
            tool_messages,
            pending_fragment_ids,
            extraction_round_count,
            pending_memory_ids,
            pending_reminder_claims,
            req_id=req_id,
        )
        return _streaming_response(stream, pending_reminder_claims)
    else:
        # 非流式发送期间延长提醒处理时限；退出时标记送达或交回等待重试
        delivered_ok = False
        lease_keeper = _start_reminder_lease_keeper(pending_reminder_claims)
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await _post_chat_completion_with_429_retry(client, shared.API_BASE_URL, shared.API_KEY, body, tag="非流式", max_retries=2 if shared.is_vertex_endpoint() else 0, req_id=req_id)

                if response.status_code == 200:
                    resp_data = response.json()
                    print(f"🔬 原始响应完整结构: {json.dumps(resp_data.get('choices', [{}])[0].get('message', {}), ensure_ascii=False)[:2000]}")
                    print(f"🔍 完整usage原始数据(排查缓存字段用): {json.dumps(resp_data.get('usage', {}), ensure_ascii=False)}")
                    try:
                        log_cache_usage(resp_data.get("usage", {}), req_id=req_id)
                    except Exception as e:
                        logger.warning("Cache usage diagnostic failed: %s", e)
                    assistant_msg = ""
                    assistant_tool_calls = None
                    assistant_reasoning = None
                    try:
                        msg_obj = resp_data["choices"][0]["message"]
                        assistant_msg = msg_obj.get("content") or ""
                        if msg_obj.get("tool_calls"):
                            assistant_tool_calls = msg_obj["tool_calls"]
                            print(f"🔧 Response 包含 {len(assistant_tool_calls)} 个工具调用")
                        if msg_obj.get("reasoning_content"):
                            assistant_reasoning = msg_obj["reasoning_content"]
                            print(f"🧠 Response 包含 reasoning_content ({len(assistant_reasoning)}字符)")
                    except (KeyError, IndexError):
                        pass

                    await _mark_pending_seen(session_id, pending_fragment_ids, pending_memory_ids)
                    delivered_ok = True

                    if shared.conversation_persistence_enabled() and (user_message or tool_messages):
                        asyncio.create_task(
                            memory_pipeline.process_memories_background(session_id, user_message, assistant_msg, model,
                                                        context_messages=extraction_context_messages,
                                                        context_round_count=extraction_round_count,
                                                        skip_conversation_log=skip_conversation_log,
                                                        tool_messages=tool_messages, assistant_tool_calls=assistant_tool_calls,
                                                        assistant_reasoning=assistant_reasoning)
                        )

                    return JSONResponse(status_code=200, content=resp_data)
                else:
                    try:
                        error_content = response.json()
                    except Exception:
                        error_content = {"error": {"message": response.text[:500], "type": "upstream_error"}}
                    return JSONResponse(status_code=response.status_code, content=error_content)
        finally:
            await _settle_reminder_claims(lease_keeper, pending_reminder_claims, delivered_ok)


# =========================================================================
# [Gemini/Vertex AI 专用] 上游 429 服务端退避重试 START
# =========================================================================
def _parse_retry_after(v):
    if not v:
        return None
    try:
        s = float(v.strip())
        return s if s >= 0 else None
    except ValueError:
        return None  # 日期串等格式：回退默认退避


async def _send_stream_with_429_retry(client, url, headers, body, max_retries=2, req_id: str = ""):
    """仅上游 429 时，在向客户端输出任何字节之前退避重试。
    返回 (最后一次 response, 已重试次数)，调用方负责 aclose。"""
    delays = (2.0, 4.0)
    attempt = 0
    rid = req_id or (str(body.get("req_id") or "") if isinstance(body, dict) else "")
    req_prefix = f"req={rid} " if rid else ""
    warn_timeout = float(os.getenv("UPSTREAM_SLOW_WARN_TIMEOUT", "60.0"))
    timeout_desc = f"{int(warn_timeout)}s" if warn_timeout.is_integer() else f"{warn_timeout:g}s"
    while True:
        attempt_num = attempt + 1
        req = client.build_request("POST", url, headers=headers, json=body)

        async def _warn_slow():
            try:
                await asyncio.sleep(warn_timeout)
                print(f"⏱ {req_prefix}尝试#{attempt_num} 已等待 {timeout_desc} 仍无响应", flush=True)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning("Warn slow upstream diagnostic failed: %s", e)

        warn_task = asyncio.create_task(_warn_slow())
        t_start = time.monotonic()
        try:
            response = await client.send(req, stream=True)
        finally:
            warn_task.cancel()

        header_wait = time.monotonic() - t_start
        try:
            print(f"⏱ {req_prefix}尝试#{attempt_num}: 发出→响应头 {header_wait:.1f}s status={response.status_code}", flush=True)
        except Exception as e:
            logger.warning("Header wait diagnostic failed: %s", e)

        if response.status_code != 429 or attempt >= max_retries:
            return response, attempt
        try:
            ra = _parse_retry_after(response.headers.get("retry-after"))
            await response.aread()
        finally:
            await response.aclose()
        wait = min(ra if ra is not None else delays[attempt] + random.uniform(0, 1), 8.0)
        attempt += 1
        print(f"⏳ {req_prefix}上游 429，{wait:.1f}s 后第 {attempt} 次重试")
        await asyncio.sleep(wait)


async def _post_chat_completion_with_429_retry(client, url, api_key, body, tag="兜底", max_retries=2, req_id: str = ""):
    """针对兜底阶段非流式请求的 429 退避重试（仅上游 429 时触发）。
    返回最后一次 response。"""
    delays = (2.0, 4.0)
    attempt = 0
    rid = req_id or (str(body.get("req_id") or "") if isinstance(body, dict) else "")
    req_prefix = f"req={rid} " if rid else ""
    while True:
        resp = await shared.post_chat_completion(client, url, api_key, body)
        if resp.status_code != 429 or attempt >= max_retries:
            if attempt > 0 and resp.status_code == 200:
                print(f"✅ [{tag}] 429 重试成功（第 {attempt} 次）", flush=True)
            return resp
        ra = _parse_retry_after(resp.headers.get("retry-after"))
        wait = min(ra if ra is not None else delays[attempt] + random.uniform(0, 1), 8.0)
        attempt += 1
        print(f"⏳ {req_prefix}[{tag}] 上游 429，{wait:.1f}s 后第 {attempt} 次重试", flush=True)
        await asyncio.sleep(wait)
# =========================================================================
# [Gemini/Vertex AI 专用] 上游 429 服务端退避重试 END
# =========================================================================


async def stream_and_capture(
    headers: dict,
    body: dict,
    session_id: str,
    user_message: str,
    model: str,
    extraction_context_messages: list = None,
    skip_conversation_log: bool = False,
    tool_messages: list = None,
    pending_fragment_ids: list = None,
    extraction_round_count: int = None,
    pending_memory_ids: list = None,
    pending_reminder_claims: list = None,
    req_id: str = "",
):
    """流式发送期间延长提醒处理时限；结束时标记送达或交回等待重试。"""
    rid = req_id or (str(body.get("req_id") or "") if isinstance(body, dict) else "")
    req_prefix = f"req={rid} " if rid else ""
    req_start_time = time.monotonic()
    stream_state = {"terminal": False, "phase": "等上游响应头", "completed": False}
    lease_keeper = _start_reminder_lease_keeper(pending_reminder_claims)
    try:
        async for chunk in _stream_and_capture_inner(
            headers,
            body,
            session_id,
            user_message,
            model,
            extraction_context_messages,
            skip_conversation_log,
            tool_messages,
            pending_fragment_ids,
            extraction_round_count,
            pending_memory_ids,
            pending_reminder_claims,
            stream_state,
            req_id=rid,
        ):
            yield chunk
    finally:
        try:
            if not stream_state.get("completed"):
                elapsed = time.monotonic() - req_start_time
                exc = sys.exc_info()[1]
                if exc is None or isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                    print(f"🔌 {req_prefix}客户端断开: 已用时 {elapsed:.1f} 秒, 阶段={stream_state.get('phase', '等上游响应头')}", flush=True)
                else:
                    exc_type = type(exc).__name__
                    exc_repr = repr(exc)
                    print(f"🔥 {req_prefix}网关异常终止: type={exc_type}, repr={exc_repr}, 已用时 {elapsed:.1f} 秒, 阶段={stream_state.get('phase', '等上游响应头')}", flush=True)
        except Exception as e:
            logger.warning("Stream termination diagnostic failed: %s", e)
        await _settle_reminder_claims(lease_keeper, pending_reminder_claims, stream_state["terminal"])


async def _stream_and_capture_inner(
    headers: dict,
    body: dict,
    session_id: str,
    user_message: str,
    model: str,
    extraction_context_messages: list = None,
    skip_conversation_log: bool = False,
    tool_messages: list = None,
    pending_fragment_ids: list = None,
    extraction_round_count: int = None,
    pending_memory_ids: list = None,
    pending_reminder_claims: list = None,
    stream_state: dict = None,
    req_id: str = "",
):
    """流式响应 + 捕获完整回复（原始字节透传，确保SSE格式和thinking数据完整）"""
    full_response = []
    full_reasoning = []
    stream_usage = {}
    saw_stream_usage = False
    logged_stream_usage = False
    line_buffer = ""
    accumulated_tool_calls = {}  # index -> OpenAI-compatible tool call
    stream_succeeded = False
    saw_terminal = False  # 见到 data: [DONE] 或 finish_reason 才算 SSE 完整结束
    captured_finish_reason = None  # 保留最终 finish_reason 供兜底逻辑判断
    held_done_bytes = None  # malformed 时拦截 [DONE]，等兜底完成后再放行
    rid = req_id or (str(body.get("req_id") or "") if isinstance(body, dict) else "")
    req_prefix = f"req={rid} " if rid else ""
    req_start_time = time.monotonic()
    saw_first_content = False

    def _mark_terminal():
        nonlocal saw_terminal
        saw_terminal = True
        if stream_state is not None:
            stream_state["terminal"] = True

    if shared.is_vertex_endpoint():
        try:
            fresh_token = shared.get_vertex_access_token()
            headers["Authorization"] = f"Bearer {fresh_token}"
            print(f"🔑 Vertex Token 已注入, 前缀: {fresh_token[:20]}..., 长度: {len(fresh_token)}")
        except Exception as e:
            print(f"⚠️ 动态生成 Vertex Token 失败: {e}")
            from fastapi import HTTPException
            if stream_state is not None:
                stream_state["completed"] = True
            raise HTTPException(status_code=502, detail=f"Vertex AI 认证令牌获取失败: {str(e)}")

    if stream_state is not None:
        stream_state["phase"] = "等上游响应头"

    async with httpx.AsyncClient(timeout=300) as client:
        # =========================================================================
        # [Gemini/Vertex AI 专用] 上游 429 重试调用 START
        # =========================================================================
        keepalive_interval = _get_keepalive_interval()
        wait_cap = _get_wait_cap()
        t_wait_start = time.monotonic()
        keepalive_count = 0
        cap_exceeded = False

        upstream_task = asyncio.create_task(
            _send_stream_with_429_retry(
                client, shared.API_BASE_URL, headers, body,
                max_retries=2 if shared.is_vertex_endpoint() else 0,
                req_id=rid,
            )
        )
        response = None
        try:
            while not upstream_task.done():
                now = time.monotonic()
                time_until_cap = (t_wait_start + wait_cap) - now
                if time_until_cap <= 0:
                    cap_exceeded = True
                    break

                time_until_keepalive = (t_wait_start + (keepalive_count + 1) * keepalive_interval) - now
                if time_until_keepalive <= 0:
                    keepalive_count += 1
                    if keepalive_count == 1:
                        try:
                            n_sec = _format_seconds(keepalive_interval)
                            print(f"📡 {req_prefix}等上游响应头已 {n_sec} 秒，开始发保活", flush=True)
                        except Exception as e:
                            logger.warning("Keepalive diagnostic failed: %s", e)
                    yield b": keep-alive\n\n"
                    continue

                wait_timeout = min(time_until_keepalive, time_until_cap)
                done, _ = await asyncio.wait([upstream_task], timeout=wait_timeout)
                if upstream_task.done():
                    break

            if cap_exceeded:
                upstream_task.cancel()
                try:
                    await upstream_task
                except (asyncio.CancelledError, Exception):
                    pass

                try:
                    cap_str = _format_seconds(wait_cap)
                    print(f"⏱ {req_prefix}等待超上限 {cap_str} 秒，已取消", flush=True)
                except Exception as e:
                    logger.warning("Wait cap diagnostic failed: %s", e)

                error_msg = f"上游 {cap_str} 秒无响应"
                err_payload = json.dumps(
                    {"error": {"message": error_msg, "type": "fallback_failed", "code": 502}},
                    ensure_ascii=False,
                )
                yield f"data: {err_payload}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

                if stream_state is not None:
                    stream_state["terminal"] = False
                    stream_state["completed"] = True
                return

            response, retried = await upstream_task
            header_wait = time.monotonic() - t_wait_start
            if keepalive_count > 0:
                try:
                    n_sec = _format_seconds(header_wait)
                    print(f"📡 {req_prefix}保活共 {keepalive_count} 次，响应头在 {n_sec} 秒到达", flush=True)
                except Exception as e:
                    logger.warning("Keepalive arrived diagnostic failed: %s", e)

            if stream_state is not None:
                stream_state["phase"] = "接收中"
            if retried and response.status_code == 200:
                print(f"✅ 429 重试成功（第 {retried} 次）")
        # =========================================================================
        # [Gemini/Vertex AI 专用] 上游 429 重试调用 END
        # =========================================================================
            # 打印上游响应头（排查thinking问题用）
            upstream_ct = response.headers.get("content-type", "")
            print(f"📨 {req_prefix}上游响应: status={response.status_code}, content-type={upstream_ct}", flush=True)

            # 上游非200时，提前打印messages结构方便debug
            if response.status_code != 200:
                msg_summary = [{"role": m.get("role"), "tool_calls": bool(m.get("tool_calls")), "tool_call_id": m.get("tool_call_id", ""), "content_type": type(m.get("content")).__name__} for m in body.get("messages", [])]
                print(f"❌ 发送的messages结构({len(msg_summary)}条): {msg_summary}", flush=True)

            error_body_parts = []
            is_error = response.status_code != 200

            async for chunk in response.aiter_bytes():
                if is_error:
                    error_body_parts.append(chunk)
                    yield chunk
                    continue

                # 旁路解析先于透传：按行处理每条SSE事件；命中思考内容（带extra_content标记）时
                # 改写为reasoning_content字段再转发，其余原样透传。
                # 终止信号要在 yield 之前登记，外壳才能据此结算送达
                text = chunk.decode("utf-8", errors="ignore")
                line_buffer += text
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    stripped_line = line.strip()
                    if stripped_line == "data: [DONE]":
                        _mark_terminal()
                        if captured_finish_reason == "malformed_function_call" or (not full_response and not accumulated_tool_calls):
                            held_done_bytes = (line + "\n").encode("utf-8")
                            print("🛡️ malformed/空正文已拦截 [DONE]，等兜底完成后放行")
                        else:
                            yield (line + "\n").encode("utf-8")
                        continue
                    if stripped_line.startswith("data: "):
                        try:
                            data = json.loads(stripped_line[6:])

                            if "usage" in data:
                                stream_usage = data["usage"]
                                saw_stream_usage = True
                                if not logged_stream_usage:
                                    logged_stream_usage = True
                                    try:
                                        log_cache_usage(stream_usage, req_id=rid)
                                    except Exception as e:
                                        logger.warning("Cache usage diagnostic failed: %s", e)

                            choices = data.get("choices") or [{}]
                            choice = first_choice = choices[0] if choices else {}
                            finish_reason = first_choice.get("finish_reason")
                            if finish_reason:
                                print(f"🏁 {req_prefix}finish_reason: {finish_reason}")
                                captured_finish_reason = finish_reason
                                _mark_terminal()
                                # malformed_function_call: 拦截这个 finish 信号，不透传给客户端
                                if finish_reason == "malformed_function_call" or (not full_response and not accumulated_tool_calls):
                                    print(f"🛡️ 拦截 {finish_reason} finish 信号（当前无正文/工具调用），客户端保持连接")
                                    continue
                            delta = first_choice.get("delta", {})
                            content = delta.get("content", "")
                            is_thinking_chunk = "extra_content" in delta and content

                            # === 监工排查专用日志 ===
                            if content and not is_thinking_chunk:
                                if not saw_first_content:
                                    saw_first_content = True
                                    try:
                                        ttft = time.monotonic() - req_start_time
                                        print(f"⏱ {req_prefix}首字节 {ttft:.1f}s", flush=True)
                                    except Exception as e:
                                        logger.warning("TTFT diagnostic failed: %s", e)
                                print(f"🟢 [排查] 收到正文块: {content[:30]!r}")
                            # =======================

                            if is_thinking_chunk:
                                full_reasoning.append(content)
                                delta["reasoning_content"] = content
                                delta.pop("content", None)
                                choices[0]["delta"] = delta
                                data["choices"] = choices
                                rewritten_line = "data: " + json.dumps(data, ensure_ascii=False)
                                yield (rewritten_line + "\n").encode("utf-8")
                            else:
                                if content:
                                    full_response.append(content)
                                yield (line + "\n").encode("utf-8")

                            reasoning = delta.get("reasoning_content", "")
                            if reasoning and not is_thinking_chunk:
                                full_reasoning.append(reasoning)

                            # 累积tool_calls
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    idx = tc.get("index", 0)
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "index": idx,
                                            "id": tc.get("id", ""),
                                            "type": tc.get("type", "function"),
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if tc.get("id"):
                                        accumulated_tool_calls[idx]["id"] = tc["id"]
                                    for key, value in tc.items():
                                        if key not in {"index", "id", "type", "function"}:
                                            accumulated_tool_calls[idx][key] = value
                                    if "function" in tc:
                                        fn = tc["function"]
                                        if fn.get("name"):
                                            accumulated_tool_calls[idx]["function"]["name"] = fn["name"]
                                        if "arguments" in fn:
                                            accumulated_tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                                        for key, value in fn.items():
                                            if key not in {"name", "arguments"}:
                                                accumulated_tool_calls[idx]["function"][key] = value
                        except (json.JSONDecodeError, KeyError, IndexError):
                            yield (line + "\n").encode("utf-8")
                    else:
                        yield (line + "\n").encode("utf-8")
            if line_buffer:
                yield line_buffer.encode("utf-8")
            stream_succeeded = response.status_code == 200
        finally:
            if not upstream_task.done():
                upstream_task.cancel()
                try:
                    await upstream_task
                except (asyncio.CancelledError, Exception):
                    pass
            if response is not None:
                await response.aclose()

    assistant_msg = "".join(full_response)
    assistant_reasoning = "".join(full_reasoning) if full_reasoning else None
    assistant_tool_calls = list(accumulated_tool_calls.values()) if accumulated_tool_calls else None

    # 状态码门禁：若上游本身返回非 200（如 429、500 等），错误已在流中原样透传，严格禁止触发补刀，
    # 立即打印错误日志并结束生成器，防止 429 连环加剧造成死循环
    if not stream_succeeded:
        if error_body_parts:
            error_text = b"".join(error_body_parts).decode("utf-8", errors="ignore")[:500]
            print(f"❌ 上游错误内容: {error_text}", flush=True)
        if stream_state is not None:
            stream_state["completed"] = True
        return

    # =========================================================================
    # [Gemini/Vertex AI 专用] 工具调用异常挽救与空回复无感兜底 START
    # -------------------------------------------------------------------------
    # 【设计背景与原理】
    # 1. 状态码门禁：仅在上游 HTTP 200 OK 且正文/工具空回时才允许补刀；非 200 上游错误严禁介入。
    # 2. 挽救逻辑（阶段一）：优先保留原请求中的 tools，将思考等级降为 low 进行
    #    轻量级非流式重试。此时模型不会发生思维溢出，能高概率成功产出合法的 tool_calls
    #    并以标准 OpenAI SSE 格式推送给客户端执行，彻底保住工具调用能力。
    # 3. 自然降级（阶段二）：若二次重试依然无工具，静默剥离 tools 重新生成自然回复。
    #    【思考链切除】：非流式补发拿到 content 后，自动剔除 <thought>/<think> 标签，防思考流出。
    #    【关键规范】：绝对不向 messages 插入任何系统或用户伪指令（避免破坏角色扮演人设）。
    #
    # 【未来剥离指引】
    # 若未来切换到 Claude / OpenAI 等原生 Function Calling 成熟稳定的模型：
    # 直接将本 START 至 END 区块（包含下方的 held_done_bytes 放行）整段安全删除即可。
    # =========================================================================
    is_malformed = (captured_finish_reason == "malformed_function_call")
    is_empty_reply = (not assistant_msg and not assistant_tool_calls)

    if is_malformed or is_empty_reply:
        phase = "兜底"
        _reason = "malformed_function_call" if is_malformed else "空正文且无 tool_calls"
        usage_str = "已收到" if (saw_stream_usage or bool(stream_usage)) else "未收到"
        if not captured_finish_reason:
            thought_chars = len(assistant_reasoning) if assistant_reasoning else 0
            content_chars = len(assistant_msg)
            print(f"⚠️ 流式未收到 finish_reason 即结束: 思考 {thought_chars} 字符 | 正文 {content_chars} 字符 | 上游usage: {usage_str}", flush=True)
        keepalive_task = None
        keepalive_count = 0
        fallback_start_time = time.monotonic()
        try:
            print(f"🩹 兜底触发 ({_reason}): 启动工具挽救与平稳恢复流程", flush=True)

            # 保活注释定时生成队列与后台任务（每隔 2 秒产生一次 keep-alive，配合取消防止泄漏）
            keepalive_queue = asyncio.Queue()

            async def _keepalive_loop():
                try:
                    while True:
                        await asyncio.sleep(2.0)
                        await keepalive_queue.put(b": keep-alive\n\n")
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            keepalive_task = asyncio.create_task(_keepalive_loop())

            async def _drain_keepalive_during(waitable):
                nonlocal keepalive_count
                task = asyncio.ensure_future(waitable)
                get_item_task = None
                try:
                    while not task.done():
                        get_item_task = asyncio.create_task(keepalive_queue.get())
                        done, _ = await asyncio.wait([task, get_item_task], return_when=asyncio.FIRST_COMPLETED)
                        if get_item_task in done:
                            keepalive_count += 1
                            item = get_item_task.result()
                            get_item_task = None
                            yield item
                        else:
                            get_item_task.cancel()
                            try:
                                await get_item_task
                            except asyncio.CancelledError:
                                pass
                            get_item_task = None
                finally:
                    if get_item_task and not get_item_task.done():
                        get_item_task.cancel()
                        try:
                            await get_item_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            logger.warning("取消保活队列读取任务异常", exc_info=True)
                    if not task.done():
                        task.cancel()
                        print("🛑 兜底等待任务已取消", flush=True)
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            logger.warning("取消兜底后台任务异常", exc_info=True)

            async for ping in _drain_keepalive_during(asyncio.sleep(0.8)):  # 缓冲防 Vertex 429 频控
                yield ping

            has_tools = bool(body.get("tools"))
            rescued_ok = False

            # --- 阶段一：若原请求带 tools，优先尝试低思考等级重试，挽救工具调用 ---
            if has_tools:
                try:
                    retry_body = dict(body)
                    retry_body["stream"] = False
                    # 针对 Vertex Gemini：将思考等级降为 low，避免思维溢出破坏工具协议
                    if "google" in retry_body and "thinking_config" in retry_body.get("google", {}):
                        retry_body["google"] = {
                            "thinking_config": {
                                "thinking_level": "low",
                                "include_thoughts": True,
                            }
                        }
                    async with httpx.AsyncClient(timeout=FALLBACK_PHASE1_TIMEOUT) as retry_client:
                        retry_task = asyncio.create_task(
                            _post_chat_completion_with_429_retry(
                                retry_client, shared.API_BASE_URL, shared.API_KEY, retry_body,
                                tag="工具挽救",
                                max_retries=2 if shared.is_vertex_endpoint() else 0,
                                req_id=rid,
                            )
                        )
                        async for ping in _drain_keepalive_during(retry_task):
                            yield ping
                        retry_resp = await retry_task

                        if retry_resp.status_code == 200:
                            retry_data = retry_resp.json()
                            first_c = retry_data.get("choices", [{}])[0]
                            r_msg = first_c.get("message", {})
                            r_tool_calls = r_msg.get("tool_calls")
                            r_content = r_msg.get("content") or ""
                            # 切除非流式思考链小作文标签（<thought>.*?</thought> 或 <think>.*?</think>）
                            r_content = re.sub(r'<(thought|think)>.*?</\1>', '', r_content, flags=re.DOTALL).strip()
                            r_finish = first_c.get("finish_reason")

                            # 成功挽救真实的工具调用！推送给客户端执行工具
                            if r_tool_calls:
                                if keepalive_task and not keepalive_task.done():
                                    keepalive_task.cancel()
                                r_tc_preview = json.dumps(r_tool_calls, ensure_ascii=False)
                                if len(r_tc_preview) > 200:
                                    r_tc_preview = r_tc_preview[:200] + "..."
                                print(f"🩹 兜底成功 (工具调用挽救): 捕获 {len(r_tool_calls)} 个工具调用 | 上游usage: {usage_str} | 内容: {r_tc_preview}", flush=True)
                                tc_payload = json.dumps(
                                    {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": r_tool_calls}}]},
                                    ensure_ascii=False,
                                )
                                yield f"data: {tc_payload}\n\n".encode("utf-8")
                                fr_payload = json.dumps(
                                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                                    ensure_ascii=False,
                                )
                                yield f"data: {fr_payload}\n\n".encode("utf-8")
                                assistant_tool_calls = r_tool_calls
                                for idx, tc in enumerate(r_tool_calls):
                                    accumulated_tool_calls[idx] = tc
                                rescued_ok = True

                            # 挽救得到正常的文本正文
                            elif r_content:
                                if keepalive_task and not keepalive_task.done():
                                    keepalive_task.cancel()
                                r_content_preview = r_content[:200] + ("..." if len(r_content) > 200 else "")
                                print(f"🩹 兜底成功 (直接输出正文): 补发 {len(r_content)} 字符 | 上游usage: {usage_str} | 内容: {r_content_preview}", flush=True)
                                if not saw_first_content:
                                    saw_first_content = True
                                    try:
                                        ttft = time.monotonic() - req_start_time
                                        print(f"⏱ {req_prefix}首字节 {ttft:.1f}s", flush=True)
                                    except Exception as e:
                                        logger.warning("TTFT diagnostic failed: %s", e)
                                c_payload = json.dumps(
                                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": r_content}}]},
                                    ensure_ascii=False,
                                )
                                yield f"data: {c_payload}\n\n".encode("utf-8")
                                fr_payload = json.dumps(
                                    {"choices": [{"index": 0, "delta": {}, "finish_reason": r_finish or "stop"}]},
                                    ensure_ascii=False,
                                )
                                yield f"data: {fr_payload}\n\n".encode("utf-8")
                                assistant_msg = r_content
                                full_response.append(r_content)
                                rescued_ok = True
                        else:
                            logger.warning("工具挽救重试失败: status=%s", retry_resp.status_code)
                except Exception:
                    logger.warning("兜底阶段一（工具挽救）异常，继续尝试阶段二", exc_info=True)

            # --- 阶段二：若无 tools 或工具挽救失败，无感剥离 tools 生成纯文本 ---
            # 注意：绝不向 messages 追加提示，仅底层取消 tools 定义，模型自然输出文本且不破人设
            if not rescued_ok:
                try:
                    print("🩹 工具挽救未果，平稳退化为无工具自然作答", flush=True)
                    text_body = dict(body)
                    text_body["stream"] = False
                    text_body.pop("tools", None)
                    text_body.pop("tool_choice", None)
                    async with httpx.AsyncClient(timeout=FALLBACK_PHASE2_TIMEOUT) as retry_client:
                        text_task = asyncio.create_task(
                            _post_chat_completion_with_429_retry(
                                retry_client, shared.API_BASE_URL, shared.API_KEY, text_body,
                                tag="降级纯文本",
                                max_retries=2 if shared.is_vertex_endpoint() else 0,
                                req_id=rid,
                            )
                        )
                        async for ping in _drain_keepalive_during(text_task):
                            yield ping
                        text_resp = await text_task

                        # 真正拿到 Vertex 响应后，及时取消保活任务
                        if keepalive_task and not keepalive_task.done():
                            keepalive_task.cancel()

                        if text_resp.status_code == 200:
                            text_data = text_resp.json()
                            text_c = text_data.get("choices", [{}])[0]
                            text_content = text_c.get("message", {}).get("content") or ""
                            # 切除非流式思考链小作文标签
                            text_content = re.sub(r'<(thought|think)>.*?</\1>', '', text_content, flags=re.DOTALL).strip()
                            if text_content:
                                text_preview = text_content[:200] + ("..." if len(text_content) > 200 else "")
                                print(f"🩹 降级纯文本成功: 补发 {len(text_content)} 字符 | 上游usage: {usage_str} | 内容: {text_preview}", flush=True)
                                if not saw_first_content:
                                    saw_first_content = True
                                    try:
                                        ttft = time.monotonic() - req_start_time
                                        print(f"⏱ {req_prefix}首字节 {ttft:.1f}s", flush=True)
                                    except Exception as e:
                                        logger.warning("TTFT diagnostic failed: %s", e)
                                c_payload = json.dumps(
                                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text_content}}]},
                                    ensure_ascii=False,
                                )
                                yield f"data: {c_payload}\n\n".encode("utf-8")
                                assistant_msg = text_content
                                full_response.append(text_content)
                        else:
                            logger.warning("降级纯文本请求失败: status=%s", text_resp.status_code)
                except Exception:
                    logger.warning("兜底阶段二（降级纯文本）异常", exc_info=True)

        except Exception:
            logger.warning("兜底流程异常，跳过", exc_info=True)
        finally:
            if keepalive_task and not keepalive_task.done():
                keepalive_task.cancel()
            fallback_duration = time.monotonic() - fallback_start_time
            print(f"🩹 兜底结束: 发送保活 {keepalive_count} 次 | 总耗时 {fallback_duration:.2f}s", flush=True)

    # 兜底全部失败且未向客户端发出任何正文或工具调用时，在放行 [DONE] 之前发 SSE 错误帧
    if (is_malformed or is_empty_reply) and not assistant_msg and not assistant_tool_calls:
        yield ('data: {"error":{"message":"上游暂时不可用，兜底未成功，请稍后重试","type":"fallback_failed","code":502}}\n\n').encode("utf-8")
        if held_done_bytes is None:
            held_done_bytes = b"data: [DONE]\n\n"

    # 放行被拦截的 [DONE]，让客户端正常关闭连接
    if held_done_bytes is not None:
        yield held_done_bytes

    # 兜底全部失败、无任何内容送达时，不调用记忆入库与后台落库，行为与上游 429 重试耗尽一致
    if (is_malformed or is_empty_reply) and not assistant_msg and not assistant_tool_calls:
        if stream_state is not None:
            stream_state["terminal"] = False
            stream_state["completed"] = True
        return
    # =========================================================================
    # [Gemini/Vertex AI 专用] 工具调用异常挽救与空回复无感兜底 END
    # =========================================================================

    if assistant_reasoning:
        print(f"🧠 Stream response 包含 reasoning_content ({len(assistant_reasoning)}字符)")

    # 打印上游错误内容
    if error_body_parts:
        error_text = b"".join(error_body_parts).decode("utf-8", errors="ignore")[:500]
        print(f"❌ 上游错误内容: {error_text}", flush=True)

    if assistant_tool_calls:
        print(f"🔧 Stream response 包含 {len(assistant_tool_calls)} 个工具调用")

    if stream_succeeded:
        # seen 账沿用 200 即成功的旧口径；提醒送达账由外壳按终止信号统一结算
        await _mark_pending_seen(session_id, pending_fragment_ids, pending_memory_ids)
        if pending_reminder_claims and not saw_terminal:
            print("⚠️ 流式 200 但未见到 [DONE]/finish_reason，提醒不记送达并交回")

    if stream_usage:
        pt = stream_usage.get("prompt_tokens", 0)
        ct = stream_usage.get("completion_tokens", 0)
        tt = stream_usage.get("total_tokens", 0)
        print(f"🔍 完整usage原始数据(排查缓存字段用): {json.dumps(stream_usage, ensure_ascii=False)}")
        if not logged_stream_usage:
            logged_stream_usage = True
            try:
                log_cache_usage(stream_usage, req_id=rid)
            except Exception as e:
                logger.warning("Cache usage diagnostic failed: %s", e)
        if shared.DATABASE_ENABLED and tt > 0 and not skip_conversation_log:
            asyncio.create_task(db_conversations.save_token_usage(session_id, model, pt, ct, tt))
            print(f"📊 {req_prefix}Stream Token: {pt} + {ct} = {tt}")

    if shared.conversation_persistence_enabled() and (user_message or tool_messages):
        asyncio.create_task(
            memory_pipeline.process_memories_background(session_id, user_message, assistant_msg, model,
                                        context_messages=extraction_context_messages,
                                        context_round_count=extraction_round_count,
                                        skip_conversation_log=skip_conversation_log,
                                        tool_messages=tool_messages, assistant_tool_calls=assistant_tool_calls,
                                        assistant_reasoning=assistant_reasoning)
        )

    if stream_state is not None:
        stream_state["completed"] = True

