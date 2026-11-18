import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import db.conversations as db_conversations
import db.core as db_core
import heartbeat_service
import memory_pipeline
import partition_engine
import routes.chat as chat_routes
import routes.heartbeat as heartbeat_routes
import shared


class FakeResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


class HeartbeatPhase1Tests(unittest.TestCase):
    def setUp(self):
        chat_routes._ACTIVE_CHAT_COUNTS.clear()
        chat_routes._LAST_REQUEST_CACHE.clear()
        chat_routes._LAST_ENVELOPE_HASH.clear()
        heartbeat_service._WEATHER_CACHE.update({"text": "", "fetched_at": 0.0, "key": ""})

    def tearDown(self):
        chat_routes._ACTIVE_CHAT_COUNTS.clear()
        chat_routes._LAST_REQUEST_CACHE.clear()
        chat_routes._LAST_ENVELOPE_HASH.clear()

    def test_phase1_memory_usage_guide_has_no_alarm_syntax(self):
        self.assertNotIn("[ALARM:", partition_engine.MEMORY_USAGE_GUIDE)

    def test_quiet_hours_and_multipliers(self):
        tz_utc8 = timezone(timedelta(hours=8))
        self.assertTrue(
            heartbeat_service.is_quiet_hours(datetime(2026, 11, 16, 23, 45, tzinfo=tz_utc8), "23:30", "08:00")
        )
        self.assertTrue(
            heartbeat_service.is_quiet_hours(datetime(2026, 11, 17, 7, 59, tzinfo=tz_utc8), "23:30", "08:00")
        )
        self.assertFalse(
            heartbeat_service.is_quiet_hours(datetime(2026, 11, 17, 8, 1, tzinfo=tz_utc8), "23:30", "08:00")
        )
        self.assertTrue(
            heartbeat_service.is_quiet_hours(datetime(2026, 11, 17, 0, 5, tzinfo=tz_utc8), "00:00", "08:00")
        )
        self.assertFalse(
            heartbeat_service.is_quiet_hours(datetime(2026, 11, 16, 23, 50, tzinfo=tz_utc8), "00:00", "08:00")
        )
        self.assertEqual(heartbeat_service.classify_user_context_multiplier("晚安啦我睡了")[0], 2.2)
        self.assertEqual(heartbeat_service.classify_user_context_multiplier("你吃了吗？")[0], 0.75)
        self.assertEqual(heartbeat_service.unread_backoff_multiplier(0), 1.0)
        self.assertEqual(heartbeat_service.unread_backoff_multiplier(1), 1.5)
        self.assertEqual(heartbeat_service.unread_backoff_multiplier(2), 2.5)
        self.assertEqual(heartbeat_service.unread_backoff_multiplier(3), 4.0)
        self.assertEqual(heartbeat_service.unread_backoff_multiplier(4), 6.0)

    def test_gatekeeper_busy_cap_and_stepped_backoff(self):
        now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)  # 14:00 UTC+8
        base_stats = {
            "latest_id": 50,
            "last_normal_at": now - timedelta(minutes=60),
            "last_user_content": "今天有点忙但还好",
            "unread_proactive_chat": 0,
            "unread_proactive_event": 2,  # event 胶囊不阻塞常规门槛
            "last_proactive_chat_at": None,
        }

        async def run_cases():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "HEARTBEAT_MAX_UNREAD", 5), \
                 patch.object(shared, "HEARTBEAT_BASE_COOLDOWN_MINUTES", 45):

                # 1) 主聊天占线让路
                chat_routes._ACTIVE_CHAT_COUNTS["s1"] = 1
                g_busy = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now)
                self.assertFalse(g_busy["allowed"])
                self.assertEqual(g_busy["status"], "skipped_busy")
                chat_routes._ACTIVE_CHAT_COUNTS.clear()

                # 2) 未读=0, 60min >= 45min -> 放行
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=base_stats)):
                    g_ok = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                    self.assertTrue(g_ok["allowed"])

                # 3) 未读=2 -> 阶梯退避 45 * 2.5 = 112.5min; 若仅过 80min 必须拦截
                stats_u2 = {
                    **base_stats,
                    "unread_proactive_chat": 2,
                    "last_proactive_chat_at": now - timedelta(minutes=80),
                }
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_u2)):
                    g_u2_block = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                    self.assertFalse(g_u2_block["allowed"])
                    self.assertEqual(g_u2_block["status"], "skipped_cooldown")

                    # 到期闹钟不受退避拉长限制
                    g_u2_alarm = await heartbeat_service.evaluate_heartbeat_gate(
                        "s1", now_utc=now, has_due_alarm=True, jitter_fn=lambda: 1.0
                    )
                    self.assertTrue(g_u2_alarm["allowed"])

                # 4) 未读=5 -> 空闲封顶；到期闹钟享 +1 豁免，未读=6 触硬上限
                stats_u5 = {**base_stats, "unread_proactive_chat": 5, "last_proactive_chat_at": now - timedelta(hours=10)}
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_u5)):
                    g_cap = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now)
                    self.assertFalse(g_cap["allowed"])
                    self.assertEqual(g_cap["status"], "skipped_unread_cap")

                    g_alarm_ok = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, has_due_alarm=True)
                    self.assertTrue(g_alarm_ok["allowed"])

                stats_u6 = {**base_stats, "unread_proactive_chat": 6, "last_proactive_chat_at": now - timedelta(hours=10)}
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_u6)):
                    g_hard = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, has_due_alarm=True)
                    self.assertFalse(g_hard["allowed"])
                    self.assertEqual(g_hard["status"], "skipped_hard_cap")

        asyncio.run(run_cases())

    def test_partition_no_rotation_and_merge_consecutive_assistants(self):
        t0 = datetime(2026, 11, 16, 8, 10, tzinfo=timezone.utc)  # 16:10 UTC+8 (周一)
        history = [
            {"role": "user", "content": "我先去忙啦", "created_at": t0 - timedelta(minutes=60)},
            {"role": "assistant", "content": "好，忙完跟我说。", "created_at": t0 - timedelta(minutes=59)},
            {
                "role": "assistant",
                "content": "忙完了没？刚才路过看到一家猫咖。\n```monologue\n记着她喜欢三花\n```",
                "created_at": t0,
                "message_type": "proactive_chat",
                "proactive_meta": {"trigger_source": "cron"},
            },
            {"role": "user", "content": "刚忙完！", "created_at": t0 + timedelta(minutes=10)},
        ]

        async def run_build():
            with patch.object(shared, "CACHE_PARTITION_X", 1), \
                 patch.object(shared, "CACHE_SUMMARY_MODEL", "sum-model"), \
                 patch.object(db_conversations, "get_session_cache_state", new=AsyncMock(return_value={"summary_parts": ["旧摘要"], "a_start_round": 0})), \
                 patch.object(partition_engine, "generate_summary", new=AsyncMock()) as mock_sum:
                msgs = await partition_engine.build_partitioned_messages(
                    "s1",
                    history,
                    "基础人设",
                    "",
                    allow_rotation=False,
                    temporary_tail_message={"role": "user", "content": "（临时纸条）"},
                )
                mock_sum.assert_not_awaited()
                return msgs

        msgs = asyncio.run(run_build())
        # system + summary(user/assistant) + A区(user, merged assistant) + B区(user merged with temporary tail)
        # 检查：没有任何相邻两个 assistant；内部字段被完全剥离；包含绝对时间场景头
        roles = [m["role"] for m in msgs]
        for i in range(len(roles) - 1):
            self.assertFalse(roles[i] == "assistant" and roles[i + 1] == "assistant", f"发现连续 assistant: {roles}")
            self.assertFalse(roles[i] == "user" and roles[i + 1] == "user", f"发现连续 user: {roles}")

        for m in msgs:
            for forbidden in ("created_at", "message_type", "proactive_meta", "skip_memory_extract"):
                self.assertNotIn(forbidden, m)

        joined_assistant = "\n".join(
            json.dumps(m["content"], ensure_ascii=False)
            for m in msgs if m["role"] == "assistant"
        )
        self.assertIn("16:10 隔着手机屏幕给阿狸留的简短微信", joined_assistant)
        self.assertIn("（临时纸条）", json.dumps(msgs[-1]["content"], ensure_ascii=False))

    def test_heartbeat_run_reuses_envelope_and_handles_silent_tool_conflict(self):
        now = datetime(2026, 11, 16, 6, 0, tzinfo=timezone.utc)
        stats = {
            "latest_id": 88,
            "last_normal_at": now - timedelta(minutes=90),
            "last_user_content": "刚才那个话题还挺有意思的",
            "unread_proactive_chat": 0,
            "unread_proactive_event": 0,
            "last_proactive_chat_at": None,
        }
        snapshot = {
            "client_system_text": "前端MCP工具说明",
            "tools": [{"type": "function", "function": {"name": "search_web", "parameters": {}}}],
            "thinking_level": "high",
            "model": "google/gemini-3.1-pro-preview",
        }
        db_rows = [
            {"id": 87, "role": "user", "content": "先忙会", "created_at": now - timedelta(minutes=91), "message_type": "normal"},
            {"id": 88, "role": "assistant", "content": "嗯，我在这。", "created_at": now - timedelta(minutes=90), "message_type": "normal"},
        ]

        captured_payloads = []

        async def fake_post(client, url, api_key, body):
            captured_payloads.append(body)
            return FakeResponse(200, {
                "choices": [{"message": {"content": "刚才看见楼下猫趴在窗台晒太阳，想起你了。\n```monologue\n希望她别太累\n```", "reasoning_content": "先想想她在干嘛"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            })

        async def run_scenarios():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", True), \
                 patch.object(shared, "PARTITION_SESSION_ID", "main-s"), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="核心人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_session_cache_state", new=AsyncMock(return_value={"summary_parts": [], "a_start_round": 0})), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 88})), \
                 patch.object(db_conversations, "save_token_usage", new=AsyncMock()), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="南京 多云 16°C")), \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post), \
                 patch.object(db_conversations, "save_message", new=AsyncMock(return_value=89)) as mock_save:

                res_ok = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res_ok["status"], "delivered")
                self.assertEqual(res_ok["message_id"], 89)
                mock_save.assert_awaited_once()
                save_kwargs = mock_save.await_args.kwargs
                self.assertEqual(save_kwargs["message_type"], "proactive_chat")
                self.assertFalse(save_kwargs["memory_extracted"])
                self.assertIn("reasoning_content", save_kwargs["metadata"])

                # 校验 payload 原样携带了 snapshot 的 tools 与 client_system_text
                sent_body = captured_payloads[0]
                self.assertEqual(sent_body["tools"], snapshot["tools"])
                self.assertIn("前端MCP工具说明", json.dumps(sent_body["messages"][0], ensure_ascii=False))

            # 场景 B：模型持续吐出 tool_calls 直到步数熔断且无有效正文 -> 按 silent 熔断不落库
            async def fake_post_tools(client, url, api_key, body):
                return FakeResponse(200, {
                    "choices": [{"message": {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "search_web", "arguments": "{\"query\":\"cat\"}"}}]}}],
                })

            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", True), \
                 patch.object(shared, "PARTITION_SESSION_ID", "main-s"), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="核心人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_session_cache_state", new=AsyncMock(return_value={"summary_parts": [], "a_start_round": 0})), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 88})), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch.object(heartbeat_service, "execute_background_tool_call", new=AsyncMock(return_value="{\"items\": []}")), \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post_tools), \
                 patch.object(db_conversations, "save_message", new=AsyncMock()) as mock_save_2:
                res_tc = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res_tc["status"], "silent")
                self.assertEqual(res_tc["reason"], "tool_circuit_breaker")
                mock_save_2.assert_not_awaited()

            # 场景 C：生成期间用户发了新消息（latest_id 变更）-> 乐观锁丢弃
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", True), \
                 patch.object(shared, "PARTITION_SESSION_ID", "main-s"), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="核心人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_session_cache_state", new=AsyncMock(return_value={"summary_parts": [], "a_start_round": 0})), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 99})), \
                 patch.object(db_conversations, "save_token_usage", new=AsyncMock()), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post), \
                 patch.object(db_conversations, "save_message", new=AsyncMock()) as mock_save_3:
                res_conflict = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res_conflict["status"], "aborted_conflict")
                mock_save_3.assert_not_awaited()

        asyncio.run(run_scenarios())

    def test_reentrancy_lock_blocks_overlapping_heartbeat(self):
        async def run_locked():
            async with heartbeat_service._active_heartbeat_lock:
                res = await heartbeat_service.run_heartbeat_tick()
                self.assertEqual(res["status"], "skipped_reentrant")

        asyncio.run(run_locked())

    def test_deferred_memory_extraction_includes_pending_proactive_and_skips_event(self):
        ctx = [
            {"role": "user", "content": "我去开会了", "_source_message_id": 10},
            {"role": "assistant", "content": "好，开完叫我", "_source_message_id": 11},
            {"role": "assistant", "content": "看天快下雨了", "_source_message_id": 12, "skip_memory_extract": True},
            {"role": "user", "content": "开完啦", "_source_message_id": 13},
        ]
        pending_proactive = [
            {"id": 120, "role": "assistant", "content": "记得带伞呀"},
        ]
        captured = {}

        async def fake_extract(msgs, existing_memories=None):
            captured["msgs"] = msgs
            return []

        async def run_mem():
            with patch.object(memory_pipeline.shared, "MEMORY_ENABLED", True), \
                 patch.object(memory_pipeline.shared, "CACHE_PARTITION_ENABLED", True), \
                 patch.object(memory_pipeline.shared, "MEMORY_EXTRACT_INTERVAL", 1), \
                 patch.object(memory_pipeline.memory_extractor, "MEMORY_MODEL", "test-model"), \
                 patch.object(memory_pipeline.db_conversations, "get_last_user_content", new=AsyncMock(return_value="我去开会了")), \
                 patch.object(memory_pipeline.db_conversations, "save_message", new=AsyncMock(side_effect=[13, 14])), \
                 patch.object(memory_pipeline.db_conversations, "get_pending_proactive_for_extraction", new=AsyncMock(return_value=pending_proactive)), \
                 patch.object(memory_pipeline.db_conversations, "mark_proactive_extracted", new=AsyncMock(return_value=1)) as mock_mark, \
                 patch.object(memory_pipeline.db_memories, "get_extraction_candidates", new=AsyncMock(return_value=[])), \
                 patch.object(memory_pipeline, "extract_memories", side_effect=fake_extract):
                await memory_pipeline.process_memories_background(
                    session_id="s1",
                    user_msg="开完啦",
                    assistant_msg="辛苦啦，喝口水",
                    model="test-model",
                    context_messages=ctx,
                    context_round_count=2,
                )
                mock_mark.assert_awaited_once_with([120])

        asyncio.run(run_mem())
        contents = [m["content"] for m in captured["msgs"]]
        self.assertIn("记得带伞呀", contents)
        self.assertNotIn("看天快下雨了", contents)

    def test_heartbeat_routes_and_background_dispatch(self):
        self.assertLess(heartbeat_service.HEARTBEAT_UPSTREAM_TIMEOUT, 100.0)

        async def run_routes():
            with patch.object(heartbeat_service, "run_heartbeat_tick", new=AsyncMock(return_value={"status": "silent"})) as mock_run, \
                 patch.object(heartbeat_service, "resolve_target_session_id", new=AsyncMock(return_value="s-main")), \
                 patch.object(db_conversations, "get_proactive_messages_since", new=AsyncMock(return_value=[{"id": 42, "content": "想你了"}])):
                r_tick = await heartbeat_routes.api_heartbeat_tick()
                self.assertTrue(r_tick["ok"])
                self.assertEqual(r_tick["status"], "silent")
                mock_run.assert_awaited_once()

                r_sync = await heartbeat_routes.api_proactive_messages(after_id=40)
                self.assertEqual(r_sync["session_id"], "s-main")
                self.assertEqual(r_sync["latest_id"], 42)
                self.assertEqual(len(r_sync["messages"]), 1)

            # background=True 放行后立刻返回 accepted 并转后台任务
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(heartbeat_service, "resolve_target_session_id", new=AsyncMock(return_value="s-main")), \
                 patch.object(heartbeat_service, "evaluate_heartbeat_gate", new=AsyncMock(return_value={"allowed": True, "status": "allowed"})), \
                 patch.object(heartbeat_service, "run_heartbeat_tick", new=AsyncMock(return_value={"status": "delivered"})) as mock_bg_run:
                r_bg = await heartbeat_routes.api_heartbeat_tick(background=True)
                self.assertTrue(r_bg["ok"])
                self.assertEqual(r_bg["status"], "accepted")
                await asyncio.sleep(0)
                mock_bg_run.assert_awaited_once()

        asyncio.run(run_routes())

    def test_stream_disconnect_recovery_sync_and_usage_persistence(self):
        fake_conn = MagicMock()
        fake_conn.fetch = AsyncMock(return_value=[
            {
                "id": 201,
                "session_id": "s-main",
                "role": "assistant",
                "content": "完整回复",
                "model": "gemini-4-hydrogen",
                "metadata": json.dumps({
                    "reasoning_content": "完整思维链",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }),
                "message_type": "normal",
                "proactive_meta": None,
                "created_at": datetime(2026, 11, 18, 2, 56, 55, tzinfo=timezone.utc),
            }
        ])
        fake_acquire = MagicMock()
        fake_acquire.__aenter__ = AsyncMock(return_value=fake_conn)
        fake_acquire.__aexit__ = AsyncMock(return_value=None)
        fake_pool = MagicMock()
        fake_pool.acquire.return_value = fake_acquire

        async def run_recovery_checks():
            # 1) recover_since_ms=0 走原有仅 proactive 的 SQL；>0 带上常规 assistant 窗口条件
            with patch.object(db_core, "get_pool", new=AsyncMock(return_value=fake_pool)):
                await db_conversations.get_proactive_messages_since("s-main", after_id=100, limit=50)
                sql_default = fake_conn.fetch.await_args.args[0]
                self.assertNotIn("to_timestamp", sql_default)

                rows = await db_conversations.get_proactive_messages_since(
                    "s-main", after_id=100, limit=50, recover_since_ms=1794970475267
                )
                sql_recover = fake_conn.fetch.await_args.args[0]
                self.assertIn("to_timestamp($4 / 1000.0)", sql_recover)
                self.assertEqual(fake_conn.fetch.await_args.args[4], 1794970475267 - 15000)
                self.assertEqual(rows[0]["reasoning"], "完整思维链")
                self.assertEqual(rows[0]["usage"]["total_tokens"], 150)

            # 2) api_proactive_messages 仅在 recover_since_ms > 0 时透传该参数
            with patch.object(heartbeat_service, "resolve_target_session_id", new=AsyncMock(return_value="s-main")), \
                 patch.object(db_conversations, "get_proactive_messages_since", new=AsyncMock(return_value=[])) as mock_get:
                await heartbeat_routes.api_proactive_messages(after_id=100)
                self.assertNotIn("recover_since_ms", mock_get.await_args.kwargs)

                await heartbeat_routes.api_proactive_messages(after_id=100, recover_since_ms=1794970475267)
                self.assertEqual(mock_get.await_args.kwargs["recover_since_ms"], 1794970475267)

            # 3) process_memories_background 将 assistant_usage 写入 metadata
            saved_calls = []
            async def fake_save(sid, role, content, model="", metadata=None, **kwargs):
                saved_calls.append({"role": role, "metadata": metadata})
                return len(saved_calls)

            with patch.object(memory_pipeline.shared, "MEMORY_ENABLED", False), \
                 patch.object(memory_pipeline.db_conversations, "get_last_user_content", new=AsyncMock(return_value="上一句")), \
                 patch.object(memory_pipeline.db_conversations, "save_message", side_effect=fake_save):
                await memory_pipeline.process_memories_background(
                    session_id="s-main",
                    user_msg="新问题",
                    assistant_msg="新回复",
                    model="test-model",
                    assistant_reasoning="思考",
                    assistant_usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                )
            ast_saved = [c for c in saved_calls if c["role"] == "assistant"][0]
            meta_obj = json.loads(ast_saved["metadata"])
            self.assertEqual(meta_obj["reasoning_content"], "思考")
            self.assertEqual(meta_obj["usage"]["total_tokens"], 30)

        asyncio.run(run_recovery_checks())


if __name__ == "__main__":
    unittest.main()
