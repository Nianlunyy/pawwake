"""Autonomous Heartbeat Phase 2 unit tests: alarms, bounded tool exploration, prefix-cache invariance, and event capsules."""

import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import heartbeat_service
import memory_pipeline
import partition_engine
import shared
from db import conversations as db_conversations
from routes import chat as chat_routes


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


if __name__ == "__main__":
    unittest.main()
