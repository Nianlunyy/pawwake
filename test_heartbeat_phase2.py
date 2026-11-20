"""Autonomous Heartbeat Phase 2 unit tests: alarms, bounded tool exploration, prefix-cache invariance, and event capsules."""

import asyncio
import contextlib
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import heartbeat_service
import memory_pipeline
import partition_engine
import shared
from db import conversations as db_conversations
from db import core as db_core
from routes import chat as chat_routes
from routes import heartbeat as heartbeat_routes


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class HeartbeatPhase2Tests(unittest.TestCase):
    def test_next_wake_parse_strip_and_clamp(self):
        cleaned_low, spec_low = heartbeat_service.parse_and_strip_next_wake(
            "路上注意安全，到了告诉我。\n<next_wake minutes=\"3\" reason=\"确认阿狸到家\" />"
        )
        self.assertEqual(cleaned_low, "路上注意安全，到了告诉我。")
        self.assertEqual(spec_low, {"minutes": 12, "reason": "确认阿狸到家"})

        cleaned_high, spec_high = heartbeat_service.parse_and_strip_next_wake(
            "早点休息哦。<next_wake reason='明早叫她起床' minutes='1440'></next_wake>"
        )
        self.assertEqual(cleaned_high, "早点休息哦。")
        self.assertEqual(spec_high, {"minutes": 720, "reason": "明早叫她起床"})

        cleaned_none, spec_none = heartbeat_service.parse_and_strip_next_wake("普通消息，没有闹钟")
        self.assertEqual(cleaned_none, "普通消息，没有闹钟")
        self.assertIsNone(spec_none)

    def test_event_capsule_and_chat_splitting(self):
        capsules_only, rem_only = heartbeat_service.parse_event_capsules_and_chat(
            "<event_capsule title=\"逛了会儿论坛\">看到一个很好笑的猫猫帖子</event_capsule>"
        )
        self.assertEqual(capsules_only, [{"title": "逛了会儿论坛", "body": "看到一个很好笑的猫猫帖子"}])
        self.assertEqual(rem_only, "")

        capsules_both, rem_both = heartbeat_service.parse_event_capsules_and_chat(
            "<event_capsule title=\"查天气\">南京夜里有点起风</event_capsule>\n\n记得多穿件外套呀！\n```monologue\n风挺凉的\n```"
        )
        self.assertEqual(len(capsules_both), 1)
        self.assertEqual(capsules_both[0]["title"], "查天气")
        self.assertIn("记得多穿件外套呀", rem_both)
        self.assertNotIn("<event_capsule", rem_both)

    def test_normal_chat_strips_next_wake_and_persists_alarm(self):
        saved_contents = []

        async def fake_save_message(session_id, role, content, model="", metadata=None, **kwargs):
            saved_contents.append((role, content))
            return 201 if role == "user" else 202

        async def run_normal():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "MEMORY_ENABLED", False), \
                 patch.object(db_conversations, "get_last_user_content", new=AsyncMock(return_value="上句话")), \
                 patch.object(db_conversations, "save_message", side_effect=fake_save_message), \
                 patch.object(db_conversations, "upsert_wake_alarm", new=AsyncMock()) as mock_alarm:
                await memory_pipeline.process_memories_background(
                    "s-main",
                    "我骑车回家了",
                    "好，小心看路，二十分钟后我再来问你。\n<next_wake minutes=\"20\" reason=\"等阿狸骑车到家确认安全\" />",
                    "gemini-3.5-flash",
                )
                mock_alarm.assert_awaited_once_with(
                    "s-main",
                    20,
                    "等阿狸骑车到家确认安全",
                    source_message_id=202,
                )

        asyncio.run(run_normal())
        self.assertEqual(saved_contents[1][0], "assistant")
        self.assertNotIn("<next_wake", saved_contents[1][1])
        self.assertEqual(saved_contents[1][1], "好，小心看路，二十分钟后我再来问你。")

    def test_gatekeeper_alarm_priority_quiet_deferral_and_event_cooldown(self):
        now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)  # 14:00 UTC+8
        quiet_now = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)  # 02:00 UTC+8

        async def run_gate():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "HEARTBEAT_MAX_UNREAD", 6), \
                 patch.object(shared, "HEARTBEAT_BASE_COOLDOWN_MINUTES", 60), \
                 patch.object(shared, "HEARTBEAT_QUIET_START", "00:00"), \
                 patch.object(shared, "HEARTBEAT_QUIET_END", "08:00"):

                # 1) 未触发的未来闹钟优先：即使常规冷却已满，未到闹钟时间也拦截
                stats_future_alarm = {
                    "latest_id": 10,
                    "last_normal_at": now - timedelta(hours=5),
                    "last_user_content": "先忙啦",
                    "unread_proactive_chat": 0,
                    "unread_proactive_event": 0,
                    "last_proactive_chat_at": None,
                    "last_proactive_event_at": None,
                    "pending_alarm": {
                        "id": 7,
                        "due_at": now + timedelta(minutes=30),
                        "reason": "半小时后叫她喝水",
                    },
                }
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_future_alarm)):
                    g_wait = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now)
                    self.assertFalse(g_wait["allowed"])
                    self.assertEqual(g_wait["status"], "skipped_waiting_alarm")

                # 2) 到点闹钟无视指数退避直接放行，且携带 reason
                stats_due_alarm = {
                    "latest_id": 11,
                    "last_normal_at": now - timedelta(hours=3),
                    "last_user_content": "我去开会啦",
                    "unread_proactive_chat": 3,
                    "unread_proactive_event": 0,
                    "last_proactive_chat_at": now - timedelta(minutes=15),
                    "last_proactive_event_at": None,
                    "pending_alarm": {
                        "id": 8,
                        "due_at": now - timedelta(minutes=2),
                        "reason": "问她开完会了没",
                    },
                }
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_due_alarm)):
                    g_due = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                    self.assertTrue(g_due["allowed"])
                    self.assertTrue(g_due["has_due_alarm"])
                    self.assertEqual(g_due["due_alarm"]["reason"], "问她开完会了没")

                    # 静默时段内顺延，不消费闹钟
                    g_quiet = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=quiet_now)
                    self.assertFalse(g_quiet["allowed"])
                    self.assertEqual(g_quiet["status"], "skipped_quiet")

                # 3) 仅记录了事件胶囊后，不增加未读聊天数，但按 event 时间防止 12 分钟连续连发
                stats_recent_event = {
                    "latest_id": 12,
                    "last_normal_at": now - timedelta(hours=3),
                    "last_user_content": "随便聊聊",
                    "unread_proactive_chat": 0,
                    "unread_proactive_event": 1,
                    "last_proactive_chat_at": None,
                    "last_proactive_event_at": now - timedelta(minutes=15),
                    "pending_alarm": None,
                }
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_recent_event)):
                    g_ev = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                    self.assertFalse(g_ev["allowed"])
                    self.assertEqual(g_ev["status"], "skipped_cooldown")

        asyncio.run(run_gate())

    def test_tool_exploration_loop_preserves_top_level_tools_and_circuit_breaks(self):
        now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)
        stats = {
            "latest_id": 88,
            "last_normal_at": now - timedelta(minutes=90),
            "last_user_content": "刚才那个话题不错",
            "unread_proactive_chat": 1,
            "unread_proactive_event": 0,
            "last_proactive_chat_at": now - timedelta(minutes=10),
            "last_proactive_event_at": None,
            "pending_alarm": {
                "id": 99,
                "due_at": now - timedelta(minutes=1),
                "reason": "等阿狸到家确认安全",
            },
        }
        snapshot_tools = [
            {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "lutopia_browse", "parameters": {"type": "object"}}},
        ]
        snapshot = {
            "client_system_text": "MCP说明",
            "tools": snapshot_tools,
            "thinking_level": "high",
            "model": "google/gemini-3.5-flash",
            "tool_executor_origin": "https://cuplivo.test",
        }
        db_rows = [
            {"id": 87, "role": "user", "content": "我出发回家了", "created_at": now - timedelta(minutes=91), "message_type": "normal"},
            {"id": 88, "role": "assistant", "content": "好，路上慢点。", "created_at": now - timedelta(minutes=90), "message_type": "normal"},
        ]

        captured_bodies = []
        call_idx = 0

        async def fake_post(client, url, api_key, body):
            nonlocal call_idx
            call_idx += 1
            captured_bodies.append(json.loads(json.dumps(body)))
            if call_idx <= 3:
                return FakeResponse(200, {
                    "choices": [{
                        "message": {
                            "content": "",
                            "tool_calls": [{"id": f"tc_{call_idx}", "type": "function", "function": {"name": "lutopia_browse", "arguments": "{\"page\":1}"}}],
                        }
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                })
            # 第 4 次（收到强制收尾系统提示后）完成总结输出：胶囊 + 正文 + 新闹钟
            return FakeResponse(200, {
                "choices": [{
                    "message": {
                        "content": (
                            "<event_capsule title=\"逛了会儿 Lutopia\">看了一圈大家都在晒晚饭</event_capsule>\n\n"
                            "刚才顺便逛了圈论坛，你安全到家了没？<next_wake minutes=\"25\" reason=\"睡前再看一眼\" />"
                        ),
                        "reasoning_content": "先逛逛再问候她",
                    }
                }],
                "usage": {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110},
            })

        saved_calls = []

        async def fake_save_msg(session_id, role, content, model="", metadata=None, **kwargs):
            saved_calls.append({"content": content, "metadata": metadata, **kwargs})
            return 200 + len(saved_calls)

        async def run_loop():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SESSION_ID", "main-s"), \
                 patch.object(shared, "HEARTBEAT_SURF_MAX_STEPS", 3), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", True), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_session_cache_state", new=AsyncMock(return_value={"summary_parts": [], "a_start_round": 0})), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 88})), \
                 patch.object(db_conversations, "save_token_usage", new=AsyncMock()), \
                 patch.object(db_conversations, "consume_wake_alarm", new=AsyncMock(return_value=True)) as mock_consume, \
                 patch.object(db_conversations, "upsert_wake_alarm", new=AsyncMock()) as mock_upsert, \
                 patch.object(db_conversations, "save_message", side_effect=fake_save_msg), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="南京 晴 18°C")), \
                 patch.object(heartbeat_service, "execute_background_tool_call", new=AsyncMock(return_value="{\"posts\": []}")) as mock_tool_exec, \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post):

                res = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res["status"], "delivered")
                self.assertEqual(res["event_ids"], [201])
                self.assertEqual(res["message_id"], 202)
                self.assertEqual(mock_tool_exec.await_count, 3)
                mock_consume.assert_awaited_once_with(99, consumed_message_id=202)
                mock_upsert.assert_awaited_once()

        asyncio.run(run_loop())

        # 死守前缀缓存：全部 4 次请求的顶层 tools 必须与 Snapshot 字节级完全一致
        self.assertEqual(len(captured_bodies), 4)
        for body in captured_bodies:
            self.assertEqual(body["tools"], snapshot_tools)

        # 首次动态尾巴提醒了闹钟原由；第 4 次最后一条 tool_result 末尾追加了系统强制收尾提示
        self.assertIn("等阿狸到家确认安全", captured_bodies[0]["messages"][-1]["content"])
        last_tool_msg = captured_bodies[3]["messages"][-1]
        self.assertEqual(last_tool_msg["role"], "tool")
        self.assertIn(heartbeat_service.TOOL_FORCE_WRAP_UP_NOTICE, last_tool_msg["content"])

        # 双气泡落库顺序与隔离校验
        self.assertEqual(len(saved_calls), 2)
        self.assertEqual(saved_calls[0]["message_type"], "proactive_event")
        self.assertEqual(saved_calls[0]["proactive_meta"]["event_title"], "逛了会儿 Lutopia")
        self.assertTrue(saved_calls[0]["skip_memory_extract"])
        self.assertEqual(saved_calls[1]["message_type"], "proactive_chat")
        self.assertEqual(saved_calls[1]["content"], "刚才顺便逛了圈论坛，你安全到家了没？")
        self.assertNotIn("<next_wake", saved_calls[1]["content"])
        self.assertNotIn("<event_capsule", saved_calls[1]["content"])
        chat_meta = json.loads(saved_calls[1]["metadata"])
        self.assertEqual(chat_meta["usage"]["prompt_tokens"], 230)
        self.assertEqual(chat_meta["usage"]["completion_tokens"], 60)
        self.assertEqual(chat_meta["usage"]["total_tokens"], 290)
        self.assertEqual(chat_meta["timing"]["thinking_level"], "high")

        # 验证 cached_tokens / reasoning_tokens 规范化聚合为前端标准格式
        agg_norm = heartbeat_service._aggregate_heartbeat_usage([
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cached_tokens": 80, "reasoning_tokens": 10},
            {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150, "prompt_tokens_details": {"cached_tokens": 90}},
        ])
        self.assertEqual(agg_norm["prompt_tokens_details"]["cached_tokens"], 170)
        self.assertEqual(agg_norm["completion_tokens_details"]["reasoning_tokens"], 10)

    def test_60s_time_budget_circuit_breaker_and_partition_event_title(self):
        # 1) 验证时间超 60s 熔断不执行新工具、不删顶层 tools，直接拦截返回
        now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)
        stats = {
            "latest_id": 10,
            "last_normal_at": now - timedelta(hours=2),
            "last_user_content": "哈喽",
            "unread_proactive_chat": 0,
            "unread_proactive_event": 0,
            "last_proactive_chat_at": None,
            "last_proactive_event_at": None,
            "pending_alarm": None,
        }
        snapshot = {
            "client_system_text": "",
            "tools": [{"type": "function", "function": {"name": "search_web"}}],
            "model": "google/gemini-3.5-flash",
            "tool_executor_origin": "https://cuplivo.test",
        }
        db_rows = [
            {"id": 10, "role": "assistant", "content": "嗨", "created_at": now - timedelta(hours=2), "message_type": "normal"},
        ]

        mono_vals = iter([0.0, 0.1, 61.0, 62.0, 62.1])

        async def fake_post_always_tools(client, url, api_key, body):
            self.assertEqual(body["tools"], snapshot["tools"])
            return FakeResponse(200, {
                "choices": [{
                    "message": {
                        "content": "<event_capsule title=\"独自逛贴吧\">逛了一会儿发现快超时先记下来</event_capsule>",
                        "tool_calls": [{"id": "tc_over", "function": {"name": "search_web", "arguments": "{}"}}],
                    }
                }],
            })

        async def run_time_breaker():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SESSION_ID", "main-s"), \
                 patch.object(shared, "HEARTBEAT_SURF_MAX_STEPS", 3), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", False), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 10})), \
                 patch.object(db_conversations, "save_message", new=AsyncMock(return_value=301)) as mock_save, \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch.object(heartbeat_service, "execute_background_tool_call", new=AsyncMock()) as mock_exec, \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post_always_tools), \
                 patch("heartbeat_service.time.monotonic", side_effect=lambda: next(mono_vals, 65.0)):

                res = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res["status"], "event_recorded")
                self.assertEqual(res["message_type"], "proactive_event")
                mock_exec.assert_not_awaited()
                mock_save.assert_awaited_once()
                self.assertEqual(mock_save.await_args.kwargs["message_type"], "proactive_event")

        asyncio.run(run_time_breaker())

        # 2) 验证 partition_engine 历史回看时会带上胶囊标题，不污染主聊天阅读
        header = partition_engine._proactive_history_prefix({
            "message_type": "proactive_event",
            "proactive_meta": {"event_title": "独自逛贴吧"},
            "created_at": now,
        })
        self.assertIn("《独自逛贴吧》", header)

    def test_direct_tool_execution_bypasses_cloudflare_access_302(self):
        async def run_direct():
            with patch.object(shared, "DATABASE_ENABLED", False), \
                 patch.object(shared, "HEARTBEAT_TOOL_CONFIG", {}), \
                 patch.object(
                     heartbeat_service,
                     "_execute_tavily_direct",
                     new=AsyncMock(return_value='{"items":[{"id":"c1","title":"新闻"}]}'),
                 ) as mock_tavily, \
                 patch.object(
                     heartbeat_service,
                     "_call_mcp_session",
                     new=AsyncMock(return_value="论坛早报内容"),
                 ) as mock_mcp:
                await heartbeat_service.save_heartbeat_tool_config({
                    "tavily_api_key": "tvly-direct-123",
                    "mcp_tool_routes": {"lutopia_cli": "https://lutopia.app/mcp/abc/sse"},
                    "mcp_servers": [{"name": "Lutopia", "url": "https://lutopia.app/mcp/abc/sse", "token": "lut-tok"}],
                })

                # 1) search_web 直接走 Tavily，不走 tool_executor_origin（彻底绕过 CF Access 302）
                r_search = await heartbeat_service.execute_background_tool_call(
                    {"id": "t1", "function": {"name": "search_web", "arguments": '{"query":"寂静之地3"}'}},
                    tool_executor_origin="https://cuplivo-web.nianlunyy2000.workers.dev",
                )
                self.assertIn("新闻", r_search)
                mock_tavily.assert_awaited_once_with("寂静之地3", "tvly-direct-123", timeout=20.0)

                # 2) lutopia_cli 直接连外部 MCP SSE，携带对应 Bearer token
                r_mcp = await heartbeat_service.execute_background_tool_call(
                    {"id": "t2", "function": {"name": "lutopia_cli", "arguments": '{"cmd":"home"}'}},
                    tool_executor_origin="https://cuplivo-web.nianlunyy2000.workers.dev",
                )
                self.assertEqual(r_mcp, "论坛早报内容")
                mock_mcp.assert_awaited_once()
                call_args, call_kwargs = mock_mcp.await_args
                self.assertEqual(call_args[0], "https://lutopia.app/mcp/abc/sse")
                self.assertEqual(call_args[1]["Authorization"], "Bearer lut-tok")
                self.assertEqual(call_kwargs["tool_name"], "lutopia_cli")

        asyncio.run(run_direct())

    def test_partial_merge_tool_config_and_http_sync_endpoint(self):
        class _FakeSyncReq:
            def __init__(self, body_bytes: bytes, headers: dict | None = None):
                self._body_bytes = body_bytes
                self.headers = headers or {}

            async def body(self):
                return self._body_bytes

        async def run_sync():
            persisted: list[tuple[str, str]] = []

            async def fake_set_cfg(key, val):
                persisted.append((key, val))

            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_TOOL_CONFIG", {}), \
                 patch.object(shared, "CUPLIVO_ORIGIN", ""), \
                 patch.object(db_core, "get_gateway_config", new=AsyncMock(return_value="")), \
                 patch.object(db_core, "set_gateway_config", side_effect=fake_set_cfg), \
                 patch.object(heartbeat_service, "dispatch_heartbeat_tick", new=AsyncMock()) as mock_dispatch:
                await heartbeat_service.save_heartbeat_tool_config({
                    "tavily_api_key": "tvly-secret-1",
                    "tavily_max_results": 12,
                    "mcp_tool_routes": {"lutopia_cli": "https://lutopia.app/mcp"},
                    "mcp_servers": [{"name": "Lutopia", "url": "https://lutopia.app/mcp", "token": "t1"}],
                })
                # D1 异常推送空字段时不得抹掉已有 Tavily Key 与 MCP 路由
                await heartbeat_service.save_heartbeat_tool_config({
                    "tavily_api_key": "",
                    "mcp_tool_routes": {},
                    "mcp_servers": [],
                })
                cfg_after_empty = await heartbeat_service.get_effective_tool_config()
                self.assertEqual(cfg_after_empty["tavily_api_key"], "tvly-secret-1")
                self.assertEqual(cfg_after_empty["tavily_max_results"], 12)
                self.assertEqual(cfg_after_empty["mcp_tool_routes"], {"lutopia_cli": "https://lutopia.app/mcp"})
                self.assertEqual(len(cfg_after_empty["mcp_servers"]), 1)

                # 通过 POST /api/heartbeat/tool-config 即时更新且不触发唤醒、不回显明文密钥
                req = _FakeSyncReq(
                    json.dumps({
                        "tool_config": {"tavily_api_key": "tvly-secret-2", "tavily_max_results": 15}
                    }).encode("utf-8"),
                    headers={"X-Cuplivo-Origin": "https://cuplivo.test/"},
                )
                res = await heartbeat_routes.api_heartbeat_tool_config(req)
                self.assertEqual(res, {"ok": True, "tavily_ready": True, "tavily_key_source": "config"})
                self.assertNotIn("tvly-secret-2", json.dumps(res))
                self.assertEqual(shared.CUPLIVO_ORIGIN, "https://cuplivo.test")
                mock_dispatch.assert_not_awaited()
                cfg_updated = await heartbeat_service.get_effective_tool_config()
                self.assertEqual(cfg_updated["tavily_api_key"], "tvly-secret-2")
                self.assertEqual(cfg_updated["mcp_tool_routes"], {"lutopia_cli": "https://lutopia.app/mcp"})

                # 显式 clear_tavily_api_key=True 可清空密钥
                clear_req = _FakeSyncReq(
                    json.dumps({"tool_config": {"tavily_api_key": "", "clear_tavily_api_key": True}}).encode("utf-8")
                )
                with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
                    clear_res = await heartbeat_routes.api_heartbeat_tool_config(clear_req)
                    self.assertEqual(clear_res, {"ok": True, "tavily_ready": False, "tavily_key_source": "none"})
                    self.assertEqual(shared.HEARTBEAT_TOOL_CONFIG["tavily_api_key"], "")

        asyncio.run(run_sync())

    def test_tavily_direct_guard_fetch_pagination_and_continuous_citations(self):
        class _MockAsyncClient:
            def __init__(self, handler):
                self._handler = handler

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, **kwargs):
                return self._handler("POST", url, **kwargs)

            async def get(self, url, **kwargs):
                return self._handler("GET", url, **kwargs)

        async def run_tavily_checks():
            # 1) 空 query 拦截，不发请求
            with patch("heartbeat_service.httpx.AsyncClient") as mock_client_cls:
                empty_res = json.loads(await heartbeat_service._execute_tavily_direct("   ", "tvly-k"))
                self.assertIn("缺少查询参数 query", empty_res["error"])
                mock_client_cls.assert_not_called()

            # 2) start_index=9 顺延编号 c9..c10
            captured_json = {}

            def _tavily_handler(method, url, **kwargs):
                captured_json.update(kwargs.get("json") or {})
                return FakeResponse(200, {
                    "answer": "南京今天晴。",
                    "results": [
                        {"title": "南京天气预报", "url": "https://weather.example/nj", "content": "晴天"},
                        {"title": "穿衣指南", "url": "https://weather.example/cloth", "content": "薄外套"},
                    ],
                })

            with patch("heartbeat_service.httpx.AsyncClient", side_effect=lambda **kw: _MockAsyncClient(_tavily_handler)):
                res_raw = await heartbeat_service._execute_tavily_direct(
                    "南京天气", "tvly-k", start_index=9, max_results=5
                )
                res_obj = json.loads(res_raw)
                self.assertEqual(captured_json["max_results"], 5)
                self.assertEqual(res_obj["items"][0]["id"], "c9")
                self.assertEqual(res_obj["items"][0]["index"], 9)
                self.assertEqual(res_obj["items"][1]["id"], "c10")

            # 3) _execute_fetch_url_direct 分段续读补齐 next_start_index
            long_text = "A" * 12500

            def _fetch_handler(method, url, **kwargs):
                if "api.tavily.com/extract" in url:
                    return FakeResponse(200, {"results": [{"raw_content": long_text}]})
                return FakeResponse(404, {})

            with patch("heartbeat_service.httpx.AsyncClient", side_effect=lambda **kw: _MockAsyncClient(_fetch_handler)):
                p1 = json.loads(await heartbeat_service._execute_fetch_url_direct("https://example.com/a", 0, "tvly-k"))
                self.assertTrue(p1["has_more"])
                self.assertEqual(p1["next_start_index"], 10000)
                p2 = json.loads(await heartbeat_service._execute_fetch_url_direct("https://example.com/a", 10000, "tvly-k"))
                self.assertFalse(p2["has_more"])
                self.assertNotIn("next_start_index", p2)

            # 4) _run_heartbeat_tick_locked 多轮 search_web 连续编号 + citations 落库（保 BP1 tools 字节一致）
            now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)
            stats = {
                "latest_id": 88,
                "last_normal_at": now - timedelta(minutes=90),
                "last_user_content": "在干嘛",
                "unread_proactive_chat": 0,
                "unread_proactive_event": 0,
                "last_proactive_chat_at": None,
                "last_proactive_event_at": None,
                "pending_alarm": None,
            }
            snapshot_tools = [
                {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "fetch_url", "parameters": {"type": "object"}}},
            ]
            snapshot = {
                "client_system_text": "MCP说明",
                "tools": snapshot_tools,
                "model": "google/gemini-3.5-flash",
                "tool_executor_origin": "https://cuplivo.test",
            }
            db_rows = [
                {"id": 88, "role": "assistant", "content": "嗯嗯", "created_at": now - timedelta(minutes=90), "message_type": "normal"},
            ]
            saved_msgs = []
            captured_bodies = []
            seen_start_indices = []
            llm_step = 0

            async def fake_post(client, url, api_key, body):
                nonlocal llm_step
                llm_step += 1
                captured_bodies.append(json.loads(json.dumps(body)))
                if llm_step == 1:
                    return FakeResponse(200, {"choices": [{"message": {"content": "", "tool_calls": [
                        {"id": "tc1", "function": {"name": "search_web", "arguments": '{"query":"猫咖"}'}},
                    ]}}]})
                if llm_step == 2:
                    return FakeResponse(200, {"choices": [{"message": {"content": "", "tool_calls": [
                        {"id": "tc2", "function": {"name": "search_web", "arguments": '{"query":"三花猫"}'}},
                    ]}}]})
                return FakeResponse(200, {"choices": [{"message": {
                    "content": '<event_capsule title="查了猫咖">看到两家店[cite:c1]和三花[cite:c3]</event_capsule>\n\n刚看到一家新猫咖[cite:c1, c3]！'
                }}]})

            async def fake_bg_tool(tc, *, tool_executor_origin="", timeout=20.0, search_start_index=1):
                seen_start_indices.append(search_start_index)
                if search_start_index == 1:
                    return json.dumps({
                        "items": [
                            {"id": "c1", "index": 1, "title": "南京新猫咖", "url": "https://cat.example/1", "text": "好评"},
                            {"id": "c2", "index": 2, "title": "非法脚本", "url": "javascript:alert(1)", "text": "过滤"},
                        ]
                    }, ensure_ascii=False)
                return json.dumps({
                    "items": [
                        {"id": f"c{search_start_index}", "index": search_start_index, "title": "三花猫图鉴", "url": "https://cat.example/3", "text": "可爱"},
                    ]
                }, ensure_ascii=False)

            async def fake_save_msg(session_id, role, content, **kwargs):
                saved_msgs.append({"content": content, **kwargs})
                return 300 + len(saved_msgs)

            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SESSION_ID", "main-s"), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", False), \
                 patch.object(shared, "HEARTBEAT_TOOL_CONFIG", {"tavily_api_key": "tvly-k"}), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 88})), \
                 patch.object(db_conversations, "get_active_proactive_topics", new=AsyncMock(return_value=[])), \
                 patch.object(db_conversations, "increment_daily_surf_count", new=AsyncMock()), \
                 patch.object(db_conversations, "upsert_proactive_topic", new=AsyncMock()), \
                 patch.object(db_conversations, "record_heartbeat_attempt", new=AsyncMock()), \
                 patch.object(db_conversations, "save_message", side_effect=fake_save_msg), \
                 patch.object(heartbeat_service, "TOOL_STEP_PACING_SECONDS", 0.0), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch.object(heartbeat_service, "execute_background_tool_call", side_effect=fake_bg_tool), \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post):
                res = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res["status"], "delivered")

            self.assertEqual(seen_start_indices, [1, 3])
            for body in captured_bodies:
                self.assertEqual(body["tools"], snapshot_tools)
            self.assertEqual(len(saved_msgs), 2)
            for m in saved_msgs:
                cites = m["proactive_meta"].get("citations") or {}
                self.assertIn("c1", cites)
                self.assertIn("c3", cites)
                self.assertNotIn("c2", cites)
                self.assertEqual(cites["c1"]["url"], "https://cat.example/1")
                self.assertEqual(cites["c3"]["index"], 3)

        asyncio.run(run_tavily_checks())

    def test_image_turn_preserves_envelope_snapshot_tools(self):
        # 1) 验证 DB 层 SQL 在入参 tools 为空数组时保留已有非空 tools
        fake_conn = MagicMock()
        fake_conn.execute = AsyncMock()
        fake_acq = MagicMock()
        fake_acq.__aenter__ = AsyncMock(return_value=fake_conn)
        fake_acq.__aexit__ = AsyncMock(return_value=None)
        fake_pool = MagicMock()
        fake_pool.acquire.return_value = fake_acq

        async def run_image_preserve():
            with patch.object(db_core, "get_pool", new=AsyncMock(return_value=fake_pool)):
                await db_conversations.save_session_envelope_snapshot("s-img", {"tools": []})
            sql_text = fake_conn.execute.await_args.args[0]
            self.assertIn("jsonb_set", sql_text)
            self.assertIn("jsonb_array_length", sql_text)

            # 2) 验证 routes/chat.py 连续请求（纯文本带 tools -> 发图不带 tools）不会用空列表覆盖镜像
            class _FakeChatReq:
                def __init__(self, body: dict, headers: dict | None = None):
                    self._body = body
                    self.headers = headers or {}

                async def json(self):
                    return self._body

            class _FakeUpstreamResp:
                status_code = 200

                def json(self):
                    return {
                        "choices": [{"message": {"role": "assistant", "content": "看到了。"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }

            saved_snapshots = []

            async def fake_save_snap(sid, snap):
                saved_snapshots.append((sid, json.loads(json.dumps(snap))))

            tools_def = [{"type": "function", "function": {"name": "search_web", "parameters": {}}}]
            chat_routes._LAST_ENVELOPE_HASH.clear()
            chat_routes._LAST_NONEMPTY_TOOLS.clear()
            try:
                with patch.object(shared, "API_KEY", "k-test"), \
                     patch.object(shared, "DATABASE_ENABLED", True), \
                     patch.object(shared, "MEMORY_ENABLED", False), \
                     patch.object(shared, "CACHE_PARTITION_ENABLED", True), \
                     patch.object(shared, "CACHE_PARTITION_X", 10), \
                     patch.object(shared, "PARTITION_SESSION_ID", "s-img"), \
                     patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="SYS")), \
                     patch("db.conversations.get_partition_conversation_messages", new=AsyncMock(return_value=([], None, 0, False))), \
                     patch("db.conversations.get_session_cache_state", new=AsyncMock(return_value={"summary_parts": [], "a_start_round": 0})), \
                     patch("db.conversations.save_session_envelope_snapshot", side_effect=fake_save_snap), \
                     patch("memory_pipeline.build_conversation_recall_text", new=AsyncMock(return_value=("", [], []))), \
                     patch.object(chat_routes, "_post_chat_completion_with_429_retry", new=AsyncMock(return_value=_FakeUpstreamResp())):
                    await chat_routes._chat_completions_inner(_FakeChatReq({
                        "model": "google/gemini-3.5-flash",
                        "stream": False,
                        "tools": tools_def,
                        "messages": [{"role": "user", "content": "帮我搜一下"}],
                    }))
                    await asyncio.sleep(0)

                    # 第二轮：发图请求，前端未带 tools 且切换了 X-Thinking-Level 以触发新 snapshot 写入
                    await chat_routes._chat_completions_inner(
                        _FakeChatReq(
                            {
                                "model": "google/gemini-3.5-flash",
                                "stream": False,
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": "看这张图"},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                ]}],
                            },
                            headers={"X-Thinking-Level": "low"},
                        )
                    )
                    await asyncio.sleep(0)
            finally:
                chat_routes._LAST_ENVELOPE_HASH.clear()
                chat_routes._LAST_NONEMPTY_TOOLS.clear()

            self.assertEqual(len(saved_snapshots), 2)
            self.assertEqual(saved_snapshots[0][1]["tools"], tools_def)
            self.assertEqual(saved_snapshots[1][1]["tools"], tools_def)

        asyncio.run(run_image_preserve())

    def test_heartbeat_cumulative_search_results_budget(self):
        async def run_budget():
            # 1) save_heartbeat_tool_config 支持 1~50 区间配置（对齐前端输入上限）
            with patch.object(shared, "DATABASE_ENABLED", False), \
                 patch.object(shared, "HEARTBEAT_TOOL_CONFIG", {}):
                saved_cfg = await heartbeat_service.save_heartbeat_tool_config({
                    "tavily_api_key": "tvly-k",
                    "tavily_max_results": 35,
                })
                self.assertEqual(saved_cfg["tavily_max_results"], 35)

            # 2) 单次唤醒累计条数扣减：上限设为 10 -> 第1轮8条(c1..c8)，第2轮限2条(c9..c10)，第3轮零请求温和拦截
            now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)
            stats = {
                "latest_id": 88,
                "last_normal_at": now - timedelta(minutes=90),
                "last_user_content": "搜下资料",
                "unread_proactive_chat": 0,
                "unread_proactive_event": 0,
                "last_proactive_chat_at": None,
                "last_proactive_event_at": None,
                "pending_alarm": None,
            }
            snapshot_tools = [
                {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object"}}},
            ]
            snapshot = {
                "client_system_text": "MCP说明",
                "tools": snapshot_tools,
                "model": "google/gemini-3.5-flash",
                "tool_executor_origin": "https://cuplivo.test",
            }
            db_rows = [
                {"id": 88, "role": "assistant", "content": "嗯", "created_at": now - timedelta(minutes=90), "message_type": "normal"},
            ]
            captured_bodies = []
            exec_calls = []
            saved_msgs = []
            llm_step = 0

            async def fake_post(client, url, api_key, body):
                nonlocal llm_step
                llm_step += 1
                captured_bodies.append(json.loads(json.dumps(body)))
                if llm_step <= 3:
                    return FakeResponse(200, {"choices": [{"message": {
                        "content": "",
                        "tool_calls": [{"id": f"tc_{llm_step}", "function": {"name": "search_web", "arguments": f'{{"query":"q{llm_step}"}}'}}],
                    }}]})
                return FakeResponse(200, {"choices": [{"message": {"content": "搜完了[cite:c1, c9]。"}}]})

            async def fake_exec(tc, *, tool_executor_origin="", timeout=20.0, search_start_index=1, search_max_results=None):
                exec_calls.append({"start": search_start_index, "max": search_max_results})
                count = search_max_results if search_max_results is not None else 8
                items = [
                    {
                        "id": f"c{search_start_index + i}",
                        "index": search_start_index + i,
                        "title": f"结果{search_start_index + i}",
                        "url": f"https://res.example/{search_start_index + i}",
                        "text": "内容",
                    }
                    for i in range(count)
                ]
                return json.dumps({"items": items}, ensure_ascii=False)

            async def fake_save(session_id, role, content, **kwargs):
                saved_msgs.append({"content": content, **kwargs})
                return 400 + len(saved_msgs)

            with contextlib.ExitStack() as stack:
                for cm in (
                    patch.object(shared, "DATABASE_ENABLED", True),
                    patch.object(shared, "HEARTBEAT_ENABLED", True),
                    patch.object(shared, "HEARTBEAT_SESSION_ID", "main-s"),
                    patch.object(shared, "HEARTBEAT_SURF_MAX_STEPS", 5),
                    patch.object(shared, "API_KEY", "test-key"),
                    patch.object(shared, "CACHE_PARTITION_ENABLED", False),
                    patch.object(shared, "HEARTBEAT_TOOL_CONFIG", {"tavily_api_key": "tvly-k", "tavily_max_results": 10}),
                    patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="人设")),
                    patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats)),
                    patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)),
                    patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))),
                    patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 88})),
                    patch.object(db_conversations, "get_active_proactive_topics", new=AsyncMock(return_value=[])),
                    patch.object(db_conversations, "increment_daily_surf_count", new=AsyncMock()),
                    patch.object(db_conversations, "upsert_proactive_topic", new=AsyncMock()),
                    patch.object(db_conversations, "record_heartbeat_attempt", new=AsyncMock()),
                    patch.object(db_conversations, "save_message", side_effect=fake_save),
                    patch.object(heartbeat_service, "TOOL_STEP_PACING_SECONDS", 0.0),
                    patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")),
                    patch.object(heartbeat_service, "execute_background_tool_call", side_effect=fake_exec),
                    patch.object(shared, "post_chat_completion", side_effect=fake_post),
                ):
                    stack.enter_context(cm)
                res = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res["status"], "delivered")

            self.assertEqual(exec_calls, [{"start": 1, "max": None}, {"start": 9, "max": 2}])
            self.assertEqual(len(captured_bodies), 4)
            for body in captured_bodies:
                self.assertEqual(body["tools"], snapshot_tools)
            # 第3轮拦截后回传给模型的 tool message 应包含上限提示
            limit_tool_msg = captured_bodies[3]["messages"][-1]
            self.assertEqual(limit_tool_msg["role"], "tool")
            self.assertIn("已达累计上限（累计最多 10 条）", limit_tool_msg["content"])
            cites = saved_msgs[0]["proactive_meta"]["citations"]
            self.assertEqual(len(cites), 10)
            self.assertIn("c1", cites)
            self.assertIn("c10", cites)

        asyncio.run(run_budget())


if __name__ == "__main__":
    unittest.main()
