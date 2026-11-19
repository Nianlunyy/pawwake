"""Autonomous Heartbeat Phase 3 unit tests: daily surf quota, tone sanitizer, 96h topic rumination gate, smart alarm reconciliation, and status dashboard."""

import asyncio
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import heartbeat_service
import memory_pipeline
import shared
from db import conversations as db_conversations
from db import core as db_core
from routes import heartbeat as heartbeat_routes


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def _make_fake_pool(fake_conn):
    fake_acquire = MagicMock()
    fake_acquire.__aenter__ = AsyncMock(return_value=fake_conn)
    fake_acquire.__aexit__ = AsyncMock(return_value=None)
    fake_pool = MagicMock()
    fake_pool.acquire.return_value = fake_acquire
    return fake_pool


class HeartbeatPhase3Tests(unittest.TestCase):
    def test_topic_tag_strict_stripping_and_extraction(self):
        # 1) 独立 <topic> 与自闭合 <topic /> 必须彻底剥离，且兼容不破坏无 topic 胶囊
        text_1 = "晚点早点睡哦，别又看手机到凌晨。<topic>熬夜看手机</topic>\n<next_wake minutes=\"30\" reason=\"督促睡觉\" />"
        no_wake, wake = heartbeat_service.parse_and_strip_next_wake(text_1)
        cleaned, explicit_topic = heartbeat_service.parse_and_strip_topic(no_wake)
        self.assertEqual(wake, {"minutes": 30, "reason": "督促睡觉"})
        self.assertEqual(explicit_topic, "熬夜看手机")
        self.assertEqual(cleaned, "晚点早点睡哦，别又看手机到凌晨。")
        self.assertNotIn("<topic", cleaned)

        # 2) <event_capsule topic="..."> 提取并保持无 topic 时与二阶段格式完全兼容
        cap_with_topic = '<event_capsule title="论坛围观" topic="猫粮测评">大家在讨论冻干</event_capsule>\n\n顺便跟你说一声。<topic>换猫粮</topic>'
        no_topic_text, top_tag = heartbeat_service.parse_and_strip_topic(cap_with_topic)
        caps, chat_rem = heartbeat_service.parse_event_capsules_and_chat(no_topic_text)
        self.assertEqual(top_tag, "换猫粮")
        self.assertEqual(caps, [{"title": "论坛围观", "body": "大家在讨论冻干", "topic": "猫粮测评"}])
        self.assertEqual(chat_rem, "顺便跟你说一声。")
        self.assertNotIn("<topic", chat_rem)

        # Fallback: 未显式传 <topic> 时依次从 capsule topic -> capsule title -> chat 短句推导
        k1, s1 = heartbeat_service.derive_proactive_topic(caps, chat_rem, top_tag)
        self.assertEqual((k1, s1), ("换猫粮", "换猫粮"))
        k2, s2 = heartbeat_service.derive_proactive_topic(caps, chat_rem, None)
        self.assertEqual((k2, s2), ("猫粮测评", "猫粮测评"))
        k3, s3 = heartbeat_service.derive_proactive_topic([{"title": "南京初雪", "body": "下雪啦"}], "下雪啦", None)
        self.assertEqual((k3, s3), ("南京初雪", "南京初雪"))
        k4, s4 = heartbeat_service.derive_proactive_topic([], "记得多喝温水，别吹风着凉了！", None)
        self.assertEqual((k4, s4), ("记得多喝温水，别吹风着凉了", "记得多喝温水，别吹风着凉了"))

        # 3) 常规对话落库也会剥离 <topic> 标签，绝不泄露进气泡
        saved = []

        async def fake_save(sid, role, content, model="", metadata=None, **kwargs):
            saved.append((role, content))
            return len(saved)

        async def run_normal_strip():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "MEMORY_ENABLED", False), \
                 patch.object(db_conversations, "get_last_user_content", new=AsyncMock(return_value="嗨")), \
                 patch.object(db_conversations, "get_pending_wake_alarm", new=AsyncMock(return_value=None)), \
                 patch.object(db_conversations, "save_message", side_effect=fake_save):
                await memory_pipeline.process_memories_background(
                    "s-main",
                    "今天好冷",
                    "多穿件羽绒服呀。\n<topic>降温保暖</topic>",
                    "gemini-3.5-flash",
                )

        asyncio.run(run_normal_strip())
        self.assertEqual(saved[1], ("assistant", "多穿件羽绒服呀。"))

    def test_daily_surf_quota_db_reset_tiered_steps_and_deduction(self):
        # 1) DB 层自然日重置与累加 SQL 验证
        fake_conn = MagicMock()
        fake_conn.fetchrow = AsyncMock(return_value={"daily_surf_date": date(2026, 11, 17), "daily_surf_count": 2})
        fake_conn.execute = AsyncMock()
        fake_conn.fetchval = AsyncMock(return_value=1)
        fake_pool = _make_fake_pool(fake_conn)

        async def run_db_surf():
            with patch.object(db_core, "get_pool", new=AsyncMock(return_value=fake_pool)):
                cnt = await db_conversations.get_daily_surf_state("s1", date(2026, 11, 18))
                self.assertEqual(cnt, 0)
                fake_conn.execute.assert_awaited_once()

                new_cnt = await db_conversations.increment_daily_surf_count("s1", date(2026, 11, 18))
                self.assertEqual(new_cnt, 1)
                self.assertIn("daily_surf_count", fake_conn.fetchval.await_args.args[0])

        asyncio.run(run_db_surf())

        # 2) 门卫跨日自动归零判定（UTC+8 14:00）
        now = datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc)
        stats_yesterday = {
            "latest_id": 10,
            "last_normal_at": now - timedelta(hours=3),
            "last_user_content": "哈喽",
            "unread_proactive_chat": 0,
            "unread_proactive_event": 0,
            "last_proactive_chat_at": None,
            "last_proactive_event_at": None,
            "pending_alarm": None,
            "daily_surf_date": date(2026, 11, 17),
            "daily_surf_count": 2,
        }

        async def run_gate_reset():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "HEARTBEAT_SURF_DAILY_LIMIT", 2), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_yesterday)), \
                 patch.object(db_conversations, "get_daily_surf_state", new=AsyncMock(return_value=0)) as mock_reset:
                gate = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                self.assertTrue(gate["allowed"])
                self.assertEqual(gate["daily_surf_count"], 0)
                self.assertEqual(gate["surf_quota_remaining"], 2)
                mock_reset.assert_awaited_once_with("s1", date(2026, 11, 18))

        asyncio.run(run_gate_reset())

        # 3) 配额未满 -> 放宽至 5 步 + 注入剩余额度 + 调用外部社区工具追加净味提醒 + 扣减配额
        snapshot_tools = [
            {"type": "function", "function": {"name": "lutopia_browse", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object"}}},
        ]
        snapshot = {
            "client_system_text": "主对话MCP说明",
            "tools": snapshot_tools,
            "model": "google/gemini-3.5-flash",
            "tool_executor_origin": "https://cuplivo.test",
        }
        db_rows = [
            {"id": 10, "role": "assistant", "content": "嗯哼", "created_at": now - timedelta(hours=3), "message_type": "normal"},
        ]

        captured_full = []
        call_n = 0

        async def fake_post_5_steps(client, url, api_key, body):
            nonlocal call_n
            call_n += 1
            captured_full.append(json.loads(json.dumps(body)))
            if call_n <= 5:
                return FakeResponse(200, {
                    "choices": [{
                        "message": {
                            "content": "",
                            "tool_calls": [{"id": f"tc_{call_n}", "function": {"name": "lutopia_browse", "arguments": "{}"}}],
                        }
                    }],
                })
            return FakeResponse(200, {
                "choices": [{
                    "message": {
                        "content": "<event_capsule title=\"逛论坛\" topic=\"赛博闲聊\">看帖收工</event_capsule>\n\n今天外面风挺舒服的。<topic>散步心情</topic>"
                    }
                }],
            })

        async def run_tiered():
            nonlocal call_n
            stats_avail = {**stats_yesterday, "daily_surf_date": date(2026, 11, 18), "daily_surf_count": 0}
            saved_list = []

            async def fake_save(sid, role, content, **kwargs):
                saved_list.append({"content": content, **kwargs})
                return len(saved_list) + 100

            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SESSION_ID", "s1"), \
                 patch.object(shared, "API_KEY", "k"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", False), \
                 patch.object(shared, "HEARTBEAT_SURF_DAILY_LIMIT", 2), \
                 patch.object(shared, "HEARTBEAT_SURF_MAX_STEPS", 8), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="静态人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_avail)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 10})), \
                 patch.object(db_conversations, "get_active_proactive_topics", new=AsyncMock(return_value=[{"topic_key": "买菜", "topic_summary": "买菜"}])), \
                 patch.object(db_conversations, "increment_daily_surf_count", new=AsyncMock(return_value=1)) as mock_inc, \
                 patch.object(db_conversations, "upsert_proactive_topic", new=AsyncMock()) as mock_upsert_topic, \
                 patch.object(db_conversations, "save_message", side_effect=fake_save), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="南京 晴 20°C")), \
                 patch.object(heartbeat_service, "TOOL_STEP_PACING_SECONDS", 0.0), \
                 patch.object(heartbeat_service, "execute_background_tool_call", new=AsyncMock(return_value="{\"posts\": []}")) as mock_exec, \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post_5_steps):

                res = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res["status"], "delivered")
                self.assertTrue(res["deep_surf_charged"])
                # min(HEARTBEAT_SURF_MAX_STEPS=8, 5) == 5 步
                self.assertEqual(mock_exec.await_count, 5)
                mock_inc.assert_awaited_once_with("s1", date(2026, 11, 18))
                mock_upsert_topic.assert_awaited_once()
                self.assertEqual(mock_upsert_topic.await_args.args[1], "散步心情")

            # 前缀缓存红线：所有 6 次请求顶层 tools 与静态 system 字节级一致，动态尾巴仅出现在末尾 user 消息
            for body in captured_full:
                self.assertEqual(body["tools"], snapshot_tools)
                self.assertEqual(body["messages"][0]["content"], "静态人设\n\n主对话MCP说明")
            tail_content = captured_full[0]["messages"][-1]["content"]
            self.assertIn("今日深度外出冲浪剩余额度：2/2 次", tail_content)
            self.assertIn("【翻篇提醒】你最近已经聊过或记过这些事：「买菜」", tail_content)
            # lutopia 工具结果回传后附加了【语感净味提醒】，第 5 轮满后附加了系统强收尾提示
            self.assertIn(heartbeat_service.TONE_SANITIZER_NOTICE, captured_full[1]["messages"][-1]["content"])
            self.assertIn(heartbeat_service.TOOL_FORCE_WRAP_UP_NOTICE, captured_full[5]["messages"][-1]["content"])
            # 气泡正文绝无 <topic> 残留
            self.assertEqual(saved_list[1]["content"], "今天外面风挺舒服的。")
            self.assertNotIn("<topic", saved_list[1]["content"])

            # 4) 配额耗尽 -> 自动切回轻量唤醒模式，上限收敛至 1 步，且 1 步纯文字不扣减冲浪配额
            captured_light = []
            light_idx = 0

            async def fake_post_light(client, url, api_key, body):
                nonlocal light_idx
                light_idx += 1
                captured_light.append(json.loads(json.dumps(body)))
                if light_idx == 1:
                    return FakeResponse(200, {
                        "choices": [{
                            "message": {
                                "content": "",
                                "tool_calls": [{"id": "tc_light", "function": {"name": "search_web", "arguments": "{}"}}],
                            }
                        }],
                    })
                return FakeResponse(200, {
                    "choices": [{"message": {"content": "只是突然想起你。<topic>随口想念</topic>"}}],
                })

            stats_exhausted = {**stats_yesterday, "daily_surf_date": date(2026, 11, 18), "daily_surf_count": 2}
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SESSION_ID", "s1"), \
                 patch.object(shared, "API_KEY", "k"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", False), \
                 patch.object(shared, "HEARTBEAT_SURF_DAILY_LIMIT", 2), \
                 patch.object(shared, "HEARTBEAT_SURF_MAX_STEPS", 5), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="静态人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_exhausted)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 10})), \
                 patch.object(db_conversations, "get_active_proactive_topics", new=AsyncMock(return_value=[])), \
                 patch.object(db_conversations, "increment_daily_surf_count", new=AsyncMock()) as mock_inc_light, \
                 patch.object(db_conversations, "upsert_proactive_topic", new=AsyncMock()), \
                 patch.object(db_conversations, "save_message", new=AsyncMock(return_value=501)), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch.object(heartbeat_service, "TOOL_STEP_PACING_SECONDS", 0.0), \
                 patch.object(heartbeat_service, "execute_background_tool_call", new=AsyncMock(return_value="{\"items\": []}")) as mock_exec_light, \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post_light):

                res_light = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res_light["status"], "delivered")
                self.assertFalse(res_light["deep_surf_charged"])
                self.assertEqual(mock_exec_light.await_count, 1)
                mock_inc_light.assert_not_awaited()

            self.assertIn("【轻量唤醒模式】", captured_light[0]["messages"][-1]["content"])
            self.assertIn(heartbeat_service.TOOL_FORCE_WRAP_UP_NOTICE, captured_light[1]["messages"][-1]["content"])
            # 仅调用了内置 search_web 而非论坛工具时，不注入口癖净味提醒
            self.assertNotIn(heartbeat_service.TONE_SANITIZER_NOTICE, captured_light[1]["messages"][-1]["content"])

        asyncio.run(run_tiered())

    def test_topic_rumination_96h_ledger_db_logic(self):
        now = datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc)
        fake_conn = MagicMock()
        fake_pool = _make_fake_pool(fake_conn)

        async def run_topic_db():
            # 1) 首次出现 -> hit_count=1, cooldown_until=None
            fake_conn.fetchrow = AsyncMock(side_effect=[
                None,
                {"id": 1, "session_id": "s1", "topic_key": "多喝水", "topic_summary": "多喝水", "hit_count": 1,
                 "first_seen_at": now, "last_seen_at": now, "cooldown_until": None},
            ])
            with patch.object(db_core, "get_pool", new=AsyncMock(return_value=fake_pool)):
                r1 = await db_conversations.upsert_proactive_topic("s1", "多喝水", now_utc=now)
                self.assertEqual(r1["hit_count"], 1)
                self.assertIsNone(r1["cooldown_until"])

            # 2) 96h 内第二次命中 -> hit_count=2, cooldown_until = now + 96h
            fake_conn.fetchrow = AsyncMock(side_effect=[
                {"id": 1, "topic_key": "多喝水", "topic_summary": "多喝水", "hit_count": 1,
                 "first_seen_at": now - timedelta(hours=12), "last_seen_at": now - timedelta(hours=5), "cooldown_until": None},
                {"id": 1, "session_id": "s1", "topic_key": "多喝水", "topic_summary": "多喝水", "hit_count": 2,
                 "first_seen_at": now - timedelta(hours=12), "last_seen_at": now, "cooldown_until": now + timedelta(hours=96)},
            ])
            with patch.object(db_core, "get_pool", new=AsyncMock(return_value=fake_pool)):
                r2 = await db_conversations.upsert_proactive_topic("s1", "多喝水", now_utc=now)
                self.assertEqual(r2["hit_count"], 2)
                self.assertEqual(r2["cooldown_until"], now + timedelta(hours=96))

            # 3) 超过 96h 后再次提及 -> 重置 hit_count=1, cooldown_until=NULL
            fake_conn.fetchrow = AsyncMock(side_effect=[
                {"id": 1, "topic_key": "多喝水", "topic_summary": "多喝水", "hit_count": 3,
                 "first_seen_at": now - timedelta(hours=150), "last_seen_at": now - timedelta(hours=97),
                 "cooldown_until": now - timedelta(hours=1)},
                {"id": 1, "session_id": "s1", "topic_key": "多喝水", "topic_summary": "多喝水", "hit_count": 1,
                 "first_seen_at": now, "last_seen_at": now, "cooldown_until": None},
            ])
            with patch.object(db_core, "get_pool", new=AsyncMock(return_value=fake_pool)):
                r3 = await db_conversations.upsert_proactive_topic("s1", "多喝水", now_utc=now)
                self.assertEqual(r3["hit_count"], 1)
                self.assertIsNone(r3["cooldown_until"])

        asyncio.run(run_topic_db())

    def test_smart_alarm_user_reply_reconciliation(self):
        now = datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc)  # 14:00 UTC+8

        async def run_reconciliation():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "test-key"), \
                 patch.object(shared, "HEARTBEAT_BASE_COOLDOWN_MINUTES", 60):

                # 1) 短期跟进闹钟（距用户回复 < 45 分钟）：自动 supersede 并回落常规冷却
                short_alarm_stats = {
                    "latest_id": 20,
                    "last_user_at": now - timedelta(minutes=10),
                    "last_normal_at": now - timedelta(minutes=10),
                    "last_user_content": "我到家啦不用担心",
                    "unread_proactive_chat": 0,
                    "unread_proactive_event": 0,
                    "last_proactive_chat_at": None,
                    "last_proactive_event_at": None,
                    "pending_alarm": {
                        "id": 31,
                        "due_at": now - timedelta(minutes=2),
                        "created_at": now - timedelta(minutes=25),
                        "reason": "确认阿狸到家了没",
                    },
                }
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=short_alarm_stats)), \
                     patch.object(db_conversations, "consume_wake_alarm", new=AsyncMock(return_value=True)) as mock_sup:
                    g_short = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                    mock_sup.assert_awaited_once_with(31, status="superseded")
                    self.assertFalse(g_short["allowed"])
                    self.assertEqual(g_short["status"], "skipped_cooldown")

                # 2) 远期预约闹钟（距用户回复 >= 45 分钟）：不误杀，到期放行并注入【互动衔接提醒】
                long_alarm_future = {
                    "latest_id": 21,
                    "last_user_at": now - timedelta(minutes=120),
                    "last_normal_at": now - timedelta(minutes=120),
                    "last_user_content": "先去忙啦",
                    "unread_proactive_chat": 0,
                    "unread_proactive_event": 0,
                    "last_proactive_chat_at": None,
                    "last_proactive_event_at": None,
                    "pending_alarm": {
                        "id": 32,
                        "due_at": now + timedelta(minutes=30),
                        "created_at": now - timedelta(minutes=150),
                        "reason": "下午四点半提醒取快递",
                    },
                }
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=long_alarm_future)), \
                     patch.object(db_conversations, "consume_wake_alarm", new=AsyncMock()) as mock_keep:
                    g_wait = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                    mock_keep.assert_not_awaited()
                    self.assertEqual(g_wait["status"], "skipped_waiting_alarm")
                    self.assertTrue(g_wait["alarm_interacted_since_created"])

                # 到期触发时放行并带出互动衔接提醒
                long_alarm_due = {
                    **long_alarm_future,
                    "pending_alarm": {
                        "id": 32,
                        "due_at": now - timedelta(minutes=1),
                        "created_at": now - timedelta(minutes=180),
                        "reason": "下午四点半提醒取快递",
                    },
                }
                with patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=long_alarm_due)), \
                     patch.object(db_conversations, "consume_wake_alarm", new=AsyncMock()) as mock_due_no_sup:
                    g_due = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                    mock_due_no_sup.assert_not_awaited()
                    self.assertTrue(g_due["allowed"])
                    self.assertTrue(g_due["alarm_interacted_since_created"])
                    note = heartbeat_service.build_heartbeat_tail_note(
                        heartbeat_service._to_local_dt(now),
                        g_due["interaction_gap_minutes"],
                        due_alarm_reason=g_due["due_alarm"]["reason"],
                        alarm_interacted_since_created=g_due["alarm_interacted_since_created"],
                    )
                    self.assertIn("【互动衔接提醒】", note)
                    self.assertIn("下午四点半提醒取快递", note)

            # 3) memory_pipeline 用户正常发消息后：短期闹钟作废、远期闹钟保留
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "MEMORY_ENABLED", False), \
                 patch.object(db_conversations, "get_last_user_content", new=AsyncMock(return_value="old")), \
                 patch.object(db_conversations, "save_message", new=AsyncMock(side_effect=[1, 2])), \
                 patch.object(db_conversations, "get_pending_wake_alarm", new=AsyncMock(return_value={
                     "id": 41, "due_at": datetime.now(timezone.utc) + timedelta(minutes=20)
                 })), \
                 patch.object(db_conversations, "consume_wake_alarm", new=AsyncMock()) as mock_mp_sup:
                await memory_pipeline.process_memories_background("s1", "到家啦", "好滴", "m")
                mock_mp_sup.assert_awaited_once_with(41, status="superseded")

            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "MEMORY_ENABLED", False), \
                 patch.object(db_conversations, "get_last_user_content", new=AsyncMock(return_value="old")), \
                 patch.object(db_conversations, "save_message", new=AsyncMock(side_effect=[3, 4])), \
                 patch.object(db_conversations, "get_pending_wake_alarm", new=AsyncMock(return_value={
                     "id": 42, "due_at": datetime.now(timezone.utc) + timedelta(minutes=120)
                 })), \
                 patch.object(db_conversations, "consume_wake_alarm", new=AsyncMock()) as mock_mp_keep:
                await memory_pipeline.process_memories_background("s1", "先忙会", "嗯嗯", "m")
                mock_mp_keep.assert_not_awaited()

        asyncio.run(run_reconciliation())

    def test_status_endpoint_and_settings_clamping(self):
        # 1) 验证 shared SETTINGS_TYPES 范围限幅
        self.assertEqual(shared.SETTINGS_TYPES["HEARTBEAT_SURF_DAILY_LIMIT"]("-5"), 0)
        self.assertEqual(shared.SETTINGS_TYPES["HEARTBEAT_SURF_DAILY_LIMIT"]("99"), 10)
        self.assertEqual(shared.SETTINGS_TYPES["HEARTBEAT_SURF_MAX_STEPS"]("1"), 2)
        self.assertEqual(shared.SETTINGS_TYPES["HEARTBEAT_SURF_MAX_STEPS"]("20"), 10)

        # 2) 验证 /api/heartbeat/status 返回三阶段配额、闹钟与冷却话题（含静默期跳过 gate 时兜底读取 stats）
        now = datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc)
        fake_stats = {
            "unread_proactive_chat": 2,
            "unread_proactive_event": 1,
            "daily_surf_date": heartbeat_service._to_local_dt(now).date(),
            "daily_surf_count": 1,
            "pending_alarm": {
                "id": 88,
                "due_at": now + timedelta(minutes=40),
                "reason": "记得喝水",
                "status": "pending",
            },
        }
        fake_topics = [
            {
                "topic_key": "感冒吃药",
                "topic_summary": "感冒吃药",
                "hit_count": 2,
                "is_cooling": True,
                "last_seen_at": now - timedelta(hours=2),
                "cooldown_until": now + timedelta(hours=94),
            }
        ]

        async def run_status():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SURF_DAILY_LIMIT", 3), \
                 patch.object(shared, "HEARTBEAT_SURF_MAX_STEPS", 5), \
                 patch.object(heartbeat_service, "resolve_target_session_id", new=AsyncMock(return_value="s-main")), \
                 patch.object(heartbeat_service, "evaluate_heartbeat_gate", new=AsyncMock(return_value={"allowed": False, "status": "skipped_quiet"})), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=fake_stats)), \
                 patch.object(db_conversations, "get_active_proactive_topics", new=AsyncMock(return_value=fake_topics)), \
                 patch.object(heartbeat_service, "_to_local_dt", return_value=heartbeat_service._to_local_dt(now)):
                res = await heartbeat_routes.api_heartbeat_status()

            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["gate_status"], "skipped_quiet")
            self.assertEqual(res["unread_proactive_chat"], 2)
            self.assertEqual(res["daily_surf_count"], 1)
            self.assertEqual(res["surf_daily_limit"], 3)
            self.assertEqual(res["surf_max_steps"], 5)
            self.assertEqual(res["surf_quota_remaining"], 2)
            self.assertEqual(res["pending_alarm"]["reason"], "记得喝水")
            self.assertEqual(len(res["cooling_topics"]), 1)
            self.assertTrue(res["cooling_topics"][0]["is_cooling"])

        asyncio.run(run_status())

    def test_anti_429_silent_cooldown_pacing_and_error_recovery(self):
        now = datetime(2026, 11, 18, 6, 0, tzinfo=timezone.utc)

        # 1) DB 层 record_heartbeat_attempt 更新 last_proactive_at，且 db/core.py 建表迁移必须包含该列
        with open("/Users/nianlunyy/pawwake/db/core.py", "r", encoding="utf-8") as f:
            schema_sql = f.read()
        for col in ("last_proactive_at", "daily_surf_date", "daily_surf_count", "proactive_topics"):
            self.assertIn(col, schema_sql)

        fake_conn = MagicMock()
        fake_conn.execute = AsyncMock()
        fake_pool = _make_fake_pool(fake_conn)

        async def run_all():
            with patch.object(db_core, "get_pool", new=AsyncMock(return_value=fake_pool)):
                await db_conversations.record_heartbeat_attempt("s1", now_utc=now)
                fake_conn.execute.assert_awaited_once()
                self.assertIn("last_proactive_at", fake_conn.execute.await_args.args[0])

            # 2) [SILENT] 推进 last_proactive_at 后，12 分钟后门卫必须按冷却拦截（绝不每 12 分钟连刷）
            stats_after_silent = {
                "latest_id": 10,
                "last_normal_at": now - timedelta(hours=3),
                "last_user_content": "去忙啦",
                "unread_proactive_chat": 0,
                "unread_proactive_event": 0,
                "last_proactive_at": now - timedelta(minutes=12),
                "last_proactive_chat_at": None,
                "last_proactive_event_at": None,
                "pending_alarm": None,
            }
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "k"), \
                 patch.object(shared, "HEARTBEAT_BASE_COOLDOWN_MINUTES", 45), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_after_silent)):
                g_silent_cooldown = await heartbeat_service.evaluate_heartbeat_gate("s1", now_utc=now, jitter_fn=lambda: 1.0)
                self.assertFalse(g_silent_cooldown["allowed"])
                self.assertEqual(g_silent_cooldown["status"], "skipped_cooldown")

            # [SILENT] 返回路径也会调用 record_heartbeat_attempt
            stats_ready = {
                **stats_after_silent,
                "last_proactive_at": now - timedelta(minutes=120),
            }
            snapshot = {
                "client_system_text": "",
                "tools": [{"type": "function", "function": {"name": "search_web"}}],
                "model": "google/gemini-3.5-flash",
                "tool_executor_origin": "https://cuplivo.test",
            }
            db_rows = [{"id": 10, "role": "assistant", "content": "嗯嗯", "created_at": now - timedelta(hours=3), "message_type": "normal"}]

            async def fake_post_silent(client, url, api_key, body):
                return FakeResponse(200, {"choices": [{"message": {"content": "[SILENT]"}}]})

            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SESSION_ID", "s1"), \
                 patch.object(shared, "API_KEY", "k"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", False), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_ready)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 10})), \
                 patch.object(db_conversations, "get_active_proactive_topics", new=AsyncMock(return_value=[])), \
                 patch.object(db_conversations, "record_heartbeat_attempt", new=AsyncMock()) as mock_rec_silent, \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post_silent):
                res_s = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res_s["status"], "silent")
                mock_rec_silent.assert_awaited_once_with("s1", now_utc=now)

            # 3) 非 Vertex 接口也执行工具轮间 pacing，且 2 轮工具后遇上游 429 会记冷却、消闹钟、扣冲浪额度
            stats_due = {
                **stats_ready,
                "pending_alarm": {
                    "id": 77,
                    "due_at": now - timedelta(minutes=1),
                    "reason": "记得看一眼",
                },
            }
            step_idx = 0

            async def fake_post_429_mid_surf(client, url, api_key, body):
                nonlocal step_idx
                step_idx += 1
                if step_idx <= 2:
                    return FakeResponse(200, {
                        "choices": [{
                            "message": {
                                "content": "",
                                "tool_calls": [{"id": f"tc_{step_idx}", "function": {"name": "search_web", "arguments": "{}"}}],
                            }
                        }],
                    })
                return FakeResponse(429, {"error": {"message": "Rate limit reached"}})

            real_sleep = asyncio.sleep
            sleep_calls = []

            async def fake_sleep(sec):
                sleep_calls.append(sec)
                await real_sleep(0)

            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_SESSION_ID", "s1"), \
                 patch.object(shared, "API_KEY", "k"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", False), \
                 patch.object(shared, "is_vertex_endpoint", return_value=False), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="人设")), \
                 patch.object(db_conversations, "get_heartbeat_session_stats", new=AsyncMock(return_value=stats_due)), \
                 patch.object(db_conversations, "get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch.object(db_conversations, "get_partition_conversation_messages", new=AsyncMock(return_value=(db_rows, None, 0, False))), \
                 patch.object(db_conversations, "get_latest_message_meta", new=AsyncMock(return_value={"id": 10})), \
                 patch.object(db_conversations, "get_active_proactive_topics", new=AsyncMock(return_value=[])), \
                 patch.object(db_conversations, "record_heartbeat_attempt", new=AsyncMock()) as mock_rec_err, \
                 patch.object(db_conversations, "consume_wake_alarm", new=AsyncMock(return_value=True)) as mock_consume_err, \
                 patch.object(db_conversations, "increment_daily_surf_count", new=AsyncMock()) as mock_inc_err, \
                 patch.object(heartbeat_service, "TOOL_STEP_PACING_SECONDS", 2.5), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch.object(heartbeat_service, "execute_background_tool_call", new=AsyncMock(return_value="{\"ok\":true}")), \
                 patch.object(heartbeat_service.asyncio, "sleep", side_effect=fake_sleep), \
                 patch.object(shared, "post_chat_completion", side_effect=fake_post_429_mid_surf):
                res_err = await heartbeat_service.run_heartbeat_tick(now_utc=now, jitter_fn=lambda: 1.0)
                self.assertEqual(res_err["status"], "upstream_error")
                self.assertEqual(res_err["code"], 429)
                self.assertEqual(sleep_calls, [2.5, 2.5])
                mock_rec_err.assert_awaited_once_with("s1", now_utc=now)
                mock_consume_err.assert_awaited_once_with(77, consumed_message_id=None)
                mock_inc_err.assert_awaited_once()

        asyncio.run(run_all())


if __name__ == "__main__":
    unittest.main()
