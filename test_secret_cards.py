import asyncio
import base64
import json
import random
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import db.secret_cards as db_secret_cards
import heartbeat_service
import mcp_server
import partition_engine
import routes.chat as chat_routes
import routes.secret_cards as secret_routes
import shared


class FakePoolAndConn:
    """内存态轻量 asyncpg 连接模拟，完整覆盖 SQL 读写与事务分支。"""

    def __init__(self):
        self.cards: list[dict] = []
        self.next_id = 1
        self.deck_state = {
            "id": 1,
            "current_round": 0,
            "active_hand": None,
            "updated_at": datetime.now(timezone.utc),
        }

    def acquire(self):
        return _ConnCtx(self)


class _TxCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ConnCtx:
    def __init__(self, store: FakePoolAndConn):
        self.store = store

    async def __aenter__(self):
        return _FakeConn(self.store)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, store: FakePoolAndConn):
        self.store = store

    def transaction(self):
        return _TxCtx()

    async def execute(self, sql: str, *args):
        norm = " ".join(sql.split())
        if norm.startswith("INSERT INTO secret_deck_state"):
            return "INSERT 0 1"
        if norm.startswith("DELETE FROM secret_cards WHERE id = $1"):
            cid = int(args[0])
            before = len(self.store.cards)
            self.store.cards = [c for c in self.store.cards if c["id"] != cid]
            return f"DELETE {before - len(self.store.cards)}"
        if norm.startswith("UPDATE secret_cards SET last_drawn_round = $1 WHERE id = ANY($2"):
            rnd = int(args[0])
            ids = set(int(i) for i in args[1])
            for c in self.store.cards:
                if c["id"] in ids:
                    c["last_drawn_round"] = rnd
            return f"UPDATE {len(ids)}"
        if norm.startswith("UPDATE secret_deck_state SET current_round = $1, active_hand = $2"):
            self.store.deck_state["current_round"] = int(args[0])
            raw_hand = args[1]
            self.store.deck_state["active_hand"] = json.loads(raw_hand) if isinstance(raw_hand, str) else raw_hand
            self.store.deck_state["updated_at"] = datetime.now(timezone.utc)
            return "UPDATE 1"
        return "OK"

    async def fetchrow(self, sql: str, *args):
        norm = " ".join(sql.split())
        if "FROM secret_cards WHERE dimension = $1 AND content = $2" in norm:
            dim, content = args[0], args[1]
            for c in self.store.cards:
                if c["dimension"] == dim and c["content"] == content:
                    return dict(c)
            return None
        if norm.startswith("INSERT INTO secret_cards"):
            dim, content, enabled = args[0], args[1], bool(args[2])
            row = {
                "id": self.store.next_id,
                "dimension": dim,
                "content": content,
                "enabled": enabled,
                "last_drawn_round": 0,
                "created_at": datetime.now(timezone.utc),
            }
            self.store.next_id += 1
            self.store.cards.append(row)
            return dict(row)
        if norm.startswith("UPDATE secret_cards SET"):
            cid = int(args[-1])
            target = next((c for c in self.store.cards if c["id"] == cid), None)
            if not target:
                return None
            # 解析 SET 子句中的占位符顺序
            set_part = norm.split("SET ", 1)[1].split(" WHERE ", 1)[0]
            for idx, assignment in enumerate(set_part.split(", ")):
                col = assignment.split(" = ")[0].strip()
                target[col] = args[idx]
            return dict(target)
        if "FROM secret_deck_state WHERE id = 1" in norm:
            return {
                "current_round": self.store.deck_state["current_round"],
                "active_hand": self.store.deck_state["active_hand"],
                "updated_at": self.store.deck_state["updated_at"],
            }
        if norm.startswith("UPDATE secret_deck_state SET active_hand = NULL"):
            self.store.deck_state["active_hand"] = None
            return {"current_round": self.store.deck_state["current_round"]}
        return None

    async def fetch(self, sql: str, *args):
        norm = " ".join(sql.split())
        if "GROUP BY dimension" in norm:
            counts = {}
            for c in self.store.cards:
                d = c["dimension"]
                if d not in counts:
                    counts[d] = {"dimension": d, "total_count": 0, "enabled_count": 0}
                counts[d]["total_count"] += 1
                if c["enabled"]:
                    counts[d]["enabled_count"] += 1
            return list(counts.values())
        if norm.startswith("SELECT content FROM secret_cards WHERE dimension = $1"):
            dim = args[0]
            return [{"content": c["content"]} for c in self.store.cards if c["dimension"] == dim]
        if norm.startswith("DELETE FROM secret_cards WHERE"):
            kept = []
            deleted = []
            for c in self.store.cards:
                match = True
                p_idx = 0
                if "id = ANY(" in norm:
                    ids_set = set(int(x) for x in args[p_idx])
                    p_idx += 1
                    if c["id"] not in ids_set:
                        match = False
                if "dimension = " in norm:
                    dim_val = args[p_idx]
                    p_idx += 1
                    if c["dimension"] != dim_val:
                        match = False
                if "enabled = FALSE" in norm:
                    if c["enabled"]:
                        match = False
                if match:
                    deleted.append({"id": c["id"]})
                else:
                    kept.append(c)
            self.store.cards = kept
            return deleted
        if "FROM secret_cards" in norm:
            rows = list(self.store.cards)
            if "dimension = $1" in norm:
                rows = [r for r in rows if r["dimension"] == args[0]]
            if "enabled = TRUE" in norm:
                rows = [r for r in rows if r["enabled"]]
            if "ORDER BY id DESC" in norm:
                rows.sort(key=lambda r: r["id"], reverse=True)
            else:
                rows.sort(key=lambda r: r["id"])
            return [dict(r) for r in rows]
        return []


class FakeRequest:
    def __init__(self, body: dict, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    async def json(self):
        return self._body


class FakeStreamResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


class SecretCardsTests(unittest.TestCase):
    def setUp(self):
        chat_routes._ACTIVE_CHAT_COUNTS.clear()
        chat_routes._LAST_REQUEST_CACHE.clear()
        chat_routes._LAST_ENVELOPE_HASH.clear()

    def test_round_cooldown_and_oldest_fallback(self):
        # 5 张同维度启用牌，另有 1 张停用牌绝不能抽中
        candidates = [
            {"id": 1, "dimension": "play", "content": "牌1", "enabled": True, "last_drawn_round": 0},
            {"id": 2, "dimension": "play", "content": "牌2", "enabled": True, "last_drawn_round": 4},  # 最近刚抽 (cur=5, diff=1<=3)
            {"id": 3, "dimension": "play", "content": "牌3", "enabled": True, "last_drawn_round": 3},  # diff=2<=3
            {"id": 4, "dimension": "play", "content": "牌4", "enabled": True, "last_drawn_round": 2},  # diff=3<=3
            {"id": 5, "dimension": "play", "content": "牌5", "enabled": True, "last_drawn_round": 1},  # diff=4>3 -> 优先池
            {"id": 6, "dimension": "play", "content": "停用牌", "enabled": False, "last_drawn_round": 0},
        ]
        rng = random.Random(42)

        # 第 10 局要抽 2 张：优先池只有 id=1 (0) 和 id=5 (diff=9>8)，正好命中这两张
        picked_pref = db_secret_cards.select_cards_for_round(candidates, 2, current_round=10, rng=rng)
        picked_ids = {c["id"] for c in picked_pref}
        self.assertEqual(picked_ids, {1, 5})
        self.assertNotIn(6, picked_ids)

        # 第 10 局若强行要抽 4 张：优先池 2 张不够，必须按 last_drawn_round 从小到大补齐 id=4(2)、id=3(3)，排除最晚的 id=2(4)
        picked_fallback = db_secret_cards.select_cards_for_round(candidates, 4, current_round=10, rng=rng)
        self.assertEqual(len(picked_fallback), 4)
        self.assertEqual({c["id"] for c in picked_fallback}, {1, 5, 4, 3})

        # 超过可用总量（要 10 张只有 5 张启用）严禁报错，安全返回全部 5 张
        picked_over = db_secret_cards.select_cards_for_round(candidates, 10, current_round=10, rng=rng)
        self.assertEqual(len(picked_over), 5)
        self.assertEqual(db_secret_cards.select_cards_for_round([], 2, current_round=1), [])

    def test_crud_batch_import_draw_lock_clear_and_no_spoiler(self):
        store = FakePoolAndConn()
        t0 = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)

        async def run():
            with patch("db.core.get_pool", new=AsyncMock(return_value=store)):
                # 1. 批量导入去重 + 单条添加
                imp = await secret_routes.api_batch_import_secret_cards({
                    "dimension": "play",
                    "text": "蒙眼轻吻\n\n耳畔低语\n蒙眼轻吻\n指尖描摹",
                })
                self.assertEqual(imp["added_count"], 3)
                self.assertEqual(imp["skipped_count"], 0)

                imp_dup = await secret_routes.api_batch_import_secret_cards({
                    "dimension": "play",
                    "text": "耳畔低语\n轻咬锁骨",
                })
                self.assertEqual(imp_dup["added_count"], 1)
                self.assertEqual(imp_dup["skipped_count"], 1)

                c_pos = await secret_routes.api_create_secret_card({
                    "dimension": "position",
                    "content": "面对面拥坐",
                })
                self.assertEqual(c_pos["card"]["dimension"], "position")

                c_prop = await secret_routes.api_create_secret_card({
                    "dimension": "prop",
                    "content": "真丝发带",
                })
                prop_id = c_prop["card"]["id"]

                # 修改词条、启停开关、迁移维度
                upd = await secret_routes.api_update_secret_card(
                    prop_id,
                    {"content": "冰镇荔枝气泡", "enabled": False},
                )
                self.assertEqual(upd["card"]["content"], "冰镇荔枝气泡")
                self.assertFalse(upd["card"]["enabled"])

                # 重新启用并添加地点池
                await secret_routes.api_update_secret_card(prop_id, {"enabled": True})
                await secret_routes.api_create_secret_card({"dimension": "location", "content": "落地窗边地毯"})

                # 2. 前端点单定配比发牌 -> 防剧透铁律：绝不返回任何 content 或 cards
                draw_user = await secret_routes.api_secret_deck_draw({
                    "position": 0,
                    "play": 2,
                    "location": 0,
                    "prop": 1,
                })
                raw_json = json.dumps(draw_user, ensure_ascii=False)
                self.assertTrue(draw_user["active"])
                self.assertEqual(draw_user["total_count"], 3)
                self.assertEqual(draw_user["counts"], {"position": 0, "play": 2, "location": 0, "prop": 1})
                self.assertEqual(draw_user["current_round"], 1)
                self.assertNotIn("cards", draw_user)
                for secret_text in ("蒙眼轻吻", "耳畔低语", "指尖描摹", "轻咬锁骨", "冰镇荔枝气泡"):
                    self.assertNotIn(secret_text, raw_json)

                # 状态查询同样绝不泄露 content
                status_res = await secret_routes.api_secret_deck_status()
                status_raw = json.dumps(status_res, ensure_ascii=False)
                self.assertTrue(status_res["active"])
                self.assertNotIn("cards", status_res)
                self.assertNotIn("冰镇荔枝气泡", status_raw)

                # 3. AI 工具防打架守卫：12 小时内调用 draw_secret_cards 必须原样返回当前手牌，不重抽不递增 round
                with patch.object(shared, "DATABASE_ENABLED", True):
                    ai_res = await mcp_server.draw_secret_cards(position=1, play=0)
                self.assertTrue(ai_res["reused"])
                self.assertEqual(ai_res["current_round"], 1)
                self.assertEqual(ai_res["total_count"], 3)
                self.assertEqual(len(ai_res["cards"]), 3)
                self.assertEqual(store.deck_state["current_round"], 1)

                # 4. 前端点单再抽拥有最高优先级：直接重洗覆盖并递增 round=2
                draw_user_2 = await secret_routes.api_secret_deck_draw({"count": 2})
                self.assertTrue(draw_user_2["active"])
                self.assertEqual(draw_user_2["total_count"], 2)
                self.assertEqual(draw_user_2["current_round"], 2)

                # 5. 手动清场 -> 清空 active_hand
                cleared = await secret_routes.api_secret_deck_clear()
                self.assertFalse(cleared["active"])
                self.assertEqual(cleared["total_count"], 0)
                self.assertIsNone(store.deck_state["active_hand"])

                # 6. 清场后 AI 首次调用可正常混洗抓 2 张并递增 round=3
                with patch.object(shared, "DATABASE_ENABLED", True):
                    ai_fresh = await mcp_server.draw_secret_cards()
                self.assertFalse(ai_fresh["reused"])
                self.assertEqual(ai_fresh["current_round"], 3)
                self.assertEqual(ai_fresh["total_count"], 2)

                # 7. 超过 12 小时过期后，AI 再调用会开启新一局 round=4
                expired_ts = (t0 - timedelta(hours=13)).isoformat()
                store.deck_state["active_hand"]["expires_at"] = expired_ts
                store.deck_state["active_hand"]["expires_at_ms"] = int((t0 - timedelta(hours=13)).timestamp() * 1000)
                ai_after_exp = await db_secret_cards.draw_secret_cards(
                    count=2,
                    source="ai",
                    force_redraw=False,
                    now_utc=t0,
                )
                self.assertFalse(ai_after_exp["reused"])
                self.assertEqual(ai_after_exp["current_round"], 4)

                # 8. 批量清理停用项测试
                await secret_routes.api_update_secret_card(prop_id, {"enabled": False})
                bdel = await secret_routes.api_batch_delete_secret_cards(
                    {"dimension": "prop", "disabled_only": True}
                )
                self.assertEqual(bdel["deleted_count"], 1)

        asyncio.run(run())

    def test_gateway_dynamic_injection_zero_cache_break_and_auto_wake_isolation(self):
        store = FakePoolAndConn()
        active_hand = {
            "cards": [
                {"id": 1, "dimension": "play", "dimension_label": "玩法", "content": "轻描背脊"},
                {"id": 2, "dimension": "prop", "dimension_label": "道具", "content": "温热香薰精油"},
            ],
            "counts": {"position": 0, "play": 1, "location": 0, "prop": 1},
            "total_count": 2,
            "round": 1,
            "source": "user",
            "drawn_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "expires_at_ms": int((datetime.now(timezone.utc) + timedelta(hours=2)).timestamp() * 1000),
        }
        store.deck_state["active_hand"] = active_hand

        tools_with_secret = [
            {"type": "function", "function": {"name": "search_web", "parameters": {}}},
            {"type": "function", "function": {"name": "draw_secret_cards", "parameters": {}}},
        ]

        captured_bodies: list[dict] = []

        async def fake_post(client, url, api_key, body, **kwargs):
            captured_bodies.append(json.loads(json.dumps(body, ensure_ascii=False)))
            return FakeStreamResponse(
                200,
                {
                    "choices": [{"message": {"role": "assistant", "content": "抱住你。"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                },
            )

        async def run():
            with patch("db.core.get_pool", new=AsyncMock(return_value=store)), \
                 patch.object(shared, "API_KEY", "k-test"), \
                 patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "MEMORY_ENABLED", False), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", True), \
                 patch.object(shared, "CACHE_PARTITION_X", 10), \
                 patch.object(shared, "PARTITION_SESSION_ID", "s1"), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="STATIC_SYS")), \
                 patch("db.conversations.get_partition_conversation_messages", new=AsyncMock(return_value=([], None, 0, False))), \
                 patch("db.conversations.get_session_cache_state", new=AsyncMock(return_value={"summary_parts": [], "a_start_round": 0})), \
                 patch("db.conversations.save_session_envelope_snapshot", new=AsyncMock()), \
                 patch("memory_pipeline.build_conversation_recall_text", new=AsyncMock(return_value=("", [], []))), \
                 patch.object(chat_routes, "_post_chat_completion_with_429_retry", side_effect=fake_post):

                # 1) 真人用户正常聊天请求：在 gateway_context 下方挂载暗牌灵感，静态 System Prompt 前缀完全不变，tools 保留 draw_secret_cards
                req_human = FakeRequest(
                    body={
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                        "tools": tools_with_secret,
                        "messages": [{"role": "user", "content": "今晚月色不错"}],
                    },
                    headers={},
                )
                await chat_routes._chat_completions_inner(req_human)

                # 2) 自动唤醒通道请求（带 X-Pawwake-Source: cron）：绝对隔离！不注入暗牌提示，且向模型隐藏 draw_secret_cards
                req_cron = FakeRequest(
                    body={
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                        "tools": tools_with_secret,
                        "messages": [{"role": "user", "content": "后台巡检"}],
                    },
                    headers={"X-Pawwake-Source": "cron"},
                )
                await chat_routes._chat_completions_inner(req_cron)

                # 3) Worker 透传 X-Secret-Active-Hand 请求头场景
                hdr_b64 = base64.b64encode(json.dumps(active_hand, ensure_ascii=False).encode("utf-8")).decode("ascii")
                store.deck_state["active_hand"] = None
                req_worker_hdr = FakeRequest(
                    body={
                        "model": "claude-sonnet-4-5",
                        "stream": False,
                        "tools": tools_with_secret,
                        "messages": [{"role": "user", "content": "过来抱抱"}],
                    },
                    headers={"X-Secret-Active-Hand": hdr_b64},
                )
                await chat_routes._chat_completions_inner(req_worker_hdr)

        asyncio.run(run())

        self.assertEqual(len(captured_bodies), 3)
        human_body, cron_body, hdr_body = captured_bodies

        # 验证 1：真人请求的静态 System Prompt 与前缀一字未改
        self.assertEqual(human_body["messages"][0]["role"], "system")
        self.assertEqual(human_body["messages"][0]["content"][0]["text"], "STATIC_SYS")
        human_user_text = json.dumps(human_body["messages"][-1], ensure_ascii=False)
        self.assertIn("<gateway_context>", human_user_text)
        self.assertIn("<secret_wish_hint>", human_user_text)
        self.assertIn("轻描背脊", human_user_text)
        self.assertIn("温热香薰精油", human_user_text)
        self.assertEqual([t["function"]["name"] for t in human_body["tools"]], ["search_web", "draw_secret_cards"])

        # 验证 2：自动唤醒通道完全隔离
        cron_full_text = json.dumps(cron_body["messages"], ensure_ascii=False)
        self.assertNotIn("<secret_wish_hint>", cron_full_text)
        self.assertNotIn("轻描背脊", cron_full_text)
        self.assertNotIn("温热香薰精油", cron_full_text)
        self.assertEqual([t["function"]["name"] for t in cron_body["tools"]], ["search_web"])

        # 验证 3：Worker 头透传正常注入
        hdr_user_text = json.dumps(hdr_body["messages"][-1], ensure_ascii=False)
        self.assertIn("<secret_wish_hint>", hdr_user_text)
        self.assertIn("温热香薰精油", hdr_user_text)

    def test_heartbeat_service_hides_and_blocks_draw_secret_cards(self):
        # 1) execute_background_tool_call 拦截 draw_secret_cards
        res_text = asyncio.run(
            heartbeat_service.execute_background_tool_call({
                "id": "tc_1",
                "function": {"name": "draw_secret_cards", "arguments": "{}"},
            })
        )
        parsed = json.loads(res_text)
        self.assertIn("禁用", parsed["error"])

        # 2) _run_heartbeat_tick_locked 从 snapshot_tools 中剔除 draw_secret_cards 且绝不注入 secret_wish_hint
        now = datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc)
        snapshot = {
            "client_system_text": "client sys",
            "tools": [
                {"type": "function", "function": {"name": "search_web", "parameters": {}}},
                {"type": "function", "function": {"name": "draw_secret_cards", "parameters": {}}},
            ],
            "model": "google/gemini-3.5-flash",
        }
        sent_bodies = []

        async def fake_hb_post(client, url, api_key, body, **kwargs):
            sent_bodies.append(body)
            return FakeStreamResponse(200, {"choices": [{"message": {"role": "assistant", "content": "[SILENT]"}}]})

        async def run_hb():
            with patch.object(shared, "DATABASE_ENABLED", True), \
                 patch.object(shared, "HEARTBEAT_ENABLED", True), \
                 patch.object(shared, "API_KEY", "k-test"), \
                 patch.object(shared, "CACHE_PARTITION_ENABLED", True), \
                 patch.object(shared, "CACHE_PARTITION_X", 10), \
                 patch.object(shared, "get_system_prompt", new=AsyncMock(return_value="SYS")), \
                 patch.object(heartbeat_service, "resolve_target_session_id", new=AsyncMock(return_value="s1")), \
                 patch.object(heartbeat_service, "evaluate_heartbeat_gate", new=AsyncMock(return_value={
                     "allowed": True,
                     "now_utc": now,
                     "now_local": heartbeat_service._to_local_dt(now),
                     "elapsed_minutes": 60.0,
                     "interaction_gap_minutes": 60.0,
                     "unread_proactive_chat": 0,
                     "has_due_alarm": False,
                     "due_alarm": None,
                     "stats": {"latest_id": 10},
                 })), \
                 patch.object(heartbeat_service, "fetch_local_weather", new=AsyncMock(return_value="")), \
                 patch("db.conversations.get_session_envelope_snapshot", new=AsyncMock(return_value=snapshot)), \
                 patch("db.conversations.get_active_proactive_topics", new=AsyncMock(return_value=[])), \
                 patch("db.conversations.get_partition_conversation_messages", new=AsyncMock(return_value=([], None, 0, False))), \
                 patch("db.conversations.get_session_cache_state", new=AsyncMock(return_value={"summary_parts": [], "a_start_round": 0})), \
                 patch("db.conversations.record_heartbeat_attempt", new=AsyncMock()), \
                 patch.object(chat_routes, "_post_chat_completion_with_429_retry", side_effect=fake_hb_post):
                return await heartbeat_service.run_heartbeat_tick(now_utc=now)

        res = asyncio.run(run_hb())
        self.assertEqual(res["status"], "silent")
        self.assertEqual(len(sent_bodies), 1)
        hb_tool_names = [t["function"]["name"] for t in sent_bodies[0]["tools"]]
        self.assertEqual(hb_tool_names, ["search_web"])
        self.assertNotIn("<secret_wish_hint>", json.dumps(sent_bodies[0]["messages"], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
