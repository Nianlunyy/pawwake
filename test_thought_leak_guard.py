import asyncio
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Request
import heartbeat_service
import partition_engine
import shared
import thought_guard
from routes import chat as chat_routes


SAMPLE_TOOLS = [
    {"type": "function", "function": {"name": "search_memories", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "browse_xiaohongshu", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "tavily_search", "parameters": {"type": "object"}}},
]

INCIDENT_1425_DRAFT = (
    "Aha!! So *that's* what happened! She was talking about that trending drama earlier. "
    "Let me call browse_xiaohongshu to search and see what everyone is talking about! "
    "(Note: I just used 1 turn for search_memories, so I still have plenty of budget left.)"
)


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text_data="", headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text_data or json.dumps(self._json, ensure_ascii=False)
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._json


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self.headers = {"content-type": "text/event-stream"}
        self._lines = lines
        self.closed = False

    async def aiter_bytes(self):
        for line in self._lines:
            yield (line + "\n").encode("utf-8")

    async def aclose(self):
        self.closed = True


class ThoughtGuardUnitTests(unittest.TestCase):
    def test_is_thinking_part_distinguishes_thought_vs_signature(self):
        self.assertTrue(thought_guard.is_thinking_part({"extra_content": {"google": {"thought": True}}}))
        self.assertFalse(thought_guard.is_thinking_part({"extra_content": {"google": {"thought": False}}}))
        self.assertFalse(thought_guard.is_thinking_part({"extra_content": {"google": {"thought_signature": "sig123"}}}))
        self.assertTrue(thought_guard.is_thinking_part({"thought": True}))
        self.assertTrue(thought_guard.is_thinking_part({"type": "thinking", "thinking": "hmm"}))
        self.assertFalse(thought_guard.is_thinking_part({"type": "text", "text": "hello"}))

    def test_extract_nonstream_message_content_handles_extra_content_and_multipart(self):
        # Case 1: whole message marked as thought via extra_content
        msg1 = {
            "role": "assistant",
            "content": "Aha! Let me think about this...",
            "extra_content": {"google": {"thought": True}},
        }
        vis1, rsn1 = thought_guard.extract_nonstream_message_content(msg1)
        self.assertEqual(vis1, "")
        self.assertIn("Aha! Let me think", rsn1)

        # Case 2: multi-part content list mixing thought and visible text
        msg2 = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Internal thought draft", "extra_content": {"google": {"thought": True}}},
                {"type": "thinking", "thinking": "Step 2 planning"},
                {"type": "text", "text": "阿狸我刚才帮你看了一眼～"},
            ],
        }
        vis2, rsn2 = thought_guard.extract_nonstream_message_content(msg2)
        self.assertEqual(vis2, "阿狸我刚才帮你看了一眼～")
        self.assertIn("Internal thought draft", rsn2)
        self.assertIn("Step 2 planning", rsn2)

        # Case 3: <thought> block and unclosed leading <thought>
        msg3 = {"role": "assistant", "content": "<thought>Let me check</thought>你好呀"}
        vis3, rsn3 = thought_guard.extract_nonstream_message_content(msg3)
        self.assertEqual(vis3, "你好呀")
        self.assertEqual(rsn3, "Let me check")

        msg4 = {"role": "assistant", "content": "<thought>Unclosed internal reasoning draft"}
        vis4, rsn4 = thought_guard.extract_nonstream_message_content(msg4)
        self.assertEqual(vis4, "")
        self.assertIn("Unclosed internal reasoning", rsn4)

        # Case 5: Vertex/Gemini non-stream gateway merges <think>...</think> + Chinese reply
        # into a single message string while still carrying extra_content.google.thought=True
        msg5 = {
            "role": "assistant",
            "content": "<think> **My Internal Monologue** Alright, let's reply. </think>\n\n阿狸中午好呀，吃过饭了吗？",
            "extra_content": {"google": {"thought": True}},
        }
        vis5, rsn5 = thought_guard.extract_nonstream_message_content(msg5)
        self.assertEqual(vis5, "阿狸中午好呀，吃过饭了吗？")
        self.assertEqual(rsn5, "**My Internal Monologue** Alright, let's reply.")

        # Case 6: Same merged <think>...</think> + reply inside a multipart text dict with extra_content
        msg6 = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<think>Internal plan</think>下午记得喝水哦",
                    "extra_content": {"google": {"thought": True}},
                }
            ],
        }
        vis6, rsn6 = thought_guard.extract_nonstream_message_content(msg6)
        self.assertEqual(vis6, "下午记得喝水哦")
        self.assertEqual(rsn6, "Internal plan")

    def test_detects_leaked_english_tool_reasoning_drafts(self):
        # Exact 14:25 incident
        self.assertTrue(
            thought_guard.is_leaked_tool_reasoning_draft(
                INCIDENT_1425_DRAFT,
                declared_tools=SAMPLE_TOOLS,
                after_tool_result=True,
                is_heartbeat=True,
            )
        )
        # Same pattern in normal chat after tool result
        self.assertTrue(
            thought_guard.is_leaked_tool_reasoning_draft(
                INCIDENT_1425_DRAFT,
                declared_tools=SAMPLE_TOOLS,
                after_tool_result=True,
                user_message="帮我看看小红书大家都在聊啥",
            )
        )
        # Post-tool English ReAct monologue without explicit tool name
        monologue_no_tool = (
            "Wait, the previous result didn't give me enough details. "
            "Let me check if I should query with another keyword or just stay silent. "
            "(Note: I have 2 turns left in my budget.)"
        )
        self.assertTrue(
            thought_guard.is_leaked_tool_reasoning_draft(
                monologue_no_tool,
                declared_tools=SAMPLE_TOOLS,
                after_tool_result=True,
                is_heartbeat=True,
            )
        )
        # Even with a couple of quoted Chinese terms inside English ReAct draft
        mixed_quoted_leak = (
            "Aha!! So *that's* what happened with '小红书' earlier! "
            "Let me call browse_xiaohongshu to search for more posts about 奶茶 right now."
        )
        self.assertTrue(
            thought_guard.is_leaked_tool_reasoning_draft(
                mixed_quoted_leak,
                declared_tools=SAMPLE_TOOLS,
                after_tool_result=True,
                user_message="小红书最近流行什么奶茶",
            )
        )

    def test_zero_false_positives_on_normal_replies_code_and_cuplivo_ui(self):
        # 1. Chinese + English technical terms + tool names
        cn_en_mixed = (
            "我刚才用 search_memories 查了一下历史记录，又用 browse_xiaohongshu 翻了翻小红书，"
            "发现大家都在吐槽 Vertex API 429 rate limit 的问题，建议先把 backoff 调高一点。"
        )
        self.assertFalse(
            thought_guard.is_leaked_tool_reasoning_draft(
                cn_en_mixed, declared_tools=SAMPLE_TOOLS, after_tool_result=True, user_message="帮我查下"
            )
        )

        # 2. cuplivo-ui interactive card with short Chinese intro
        cuplivo_card = (
            "给你做好了一个小组件：\n"
            "```cuplivo-ui\n"
            "<!-- title: 每日打卡 -->\n"
            "<div class='card'>Let me call browse_xiaohongshu button demo</div>\n"
            "<script>function handleClick() { console.log('Note: 1 turn used'); }</script>\n"
            "```"
        )
        self.assertFalse(
            thought_guard.is_leaked_tool_reasoning_draft(
                cuplivo_card, declared_tools=SAMPLE_TOOLS, after_tool_result=True, user_message="做个卡片"
            )
        )

        # 3. Pure cuplivo-ui card with zero Chinese prose outside
        pure_card = (
            "```cuplivo-ui\n"
            "<div class='app'>Aha!! Welcome to your dashboard. Let me call api_fetch()</div>\n"
            "```"
        )
        self.assertFalse(
            thought_guard.is_leaked_tool_reasoning_draft(
                pure_card, declared_tools=SAMPLE_TOOLS, after_tool_result=True, user_message="更新卡片"
            )
        )

        # 4. Regular English reply to an English-speaking user
        en_reply = (
            "You can call search_memories with a query string to look up past notes. "
            "It returns ranked snippets along with timestamps so you can inspect the context."
        )
        self.assertFalse(
            thought_guard.is_leaked_tool_reasoning_draft(
                en_reply,
                declared_tools=SAMPLE_TOOLS,
                after_tool_result=False,
                user_message="How do I use search_memories in this gateway?",
            )
        )

        # 5. Heartbeat <event_capsule> with Chinese body
        capsule_reply = (
            "<event_capsule title=\"刷小红书看到好玩的\">"
            "刚才刷小红书看到一家超可爱的咖啡馆，顺手记下来晚点分享给阿狸。"
            "</event_capsule>\n[SILENT]"
        )
        self.assertFalse(
            thought_guard.is_leaked_tool_reasoning_draft(
                capsule_reply, declared_tools=SAMPLE_TOOLS, after_tool_result=True, is_heartbeat=True
            )
        )


class HeartbeatThoughtLeakAsyncTests(unittest.IsolatedAsyncioTestCase):
    def _base_patches(self, fake_post, save_mock):
        gate_allowed = {
            "allowed": True,
            "status": "allowed",
            "now_utc": datetime(2026, 11, 26, 6, 25, tzinfo=timezone.utc),
            "now_local": datetime(2026, 11, 26, 14, 25, tzinfo=timezone.utc),
            "elapsed_minutes": 180.0,
            "interaction_gap_minutes": 180.0,
            "required_minutes": 60.0,
            "context_kind": "normal",
            "unread_proactive_chat": 0,
            "has_due_alarm": False,
            "due_alarm": None,
            "alarm_interacted_since_created": False,
            "stats": {"latest_id": 42},
            "daily_surf_count": 0,
            "surf_daily_limit": 2,
            "surf_max_steps": 5,
            "surf_quota_remaining": 2,
        }
        envelope = {
            "model": "google/gemini-2.5-pro",
            "client_system_text": "主对话MCP说明",
            "tools": SAMPLE_TOOLS,
            "thinking_level": "medium",
        }
        return [
            patch.object(heartbeat_service, "TOOL_STEP_PACING_SECONDS", 0.0),
            patch.object(heartbeat_service, "THOUGHT_LEAK_COOLDOWN_SECONDS", 0.0),
            patch.object(shared, "DATABASE_ENABLED", True),
            patch.object(shared, "HEARTBEAT_ENABLED", True),
            patch.object(shared, "CACHE_PARTITION_ENABLED", False),
            patch.object(shared, "API_BASE_URL", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi/chat/completions"),
            patch.object(shared, "API_KEY", "test-key"),
            patch.object(shared, "get_system_prompt", AsyncMock(return_value="静态人设")),
            patch.object(heartbeat_service, "resolve_target_session_id", AsyncMock(return_value="s1")),
            patch.object(heartbeat_service.db_conversations, "get_latest_message_meta", AsyncMock(return_value={"id": 42})),
            patch.object(heartbeat_service.db_conversations, "get_session_envelope_snapshot", AsyncMock(return_value=envelope)),
            patch.object(heartbeat_service.db_conversations, "get_active_proactive_topics", AsyncMock(return_value=[])),
            patch.object(heartbeat_service.db_conversations, "save_token_usage", AsyncMock()),
            patch.object(heartbeat_service.db_conversations, "save_message", save_mock),
            patch.object(heartbeat_service.db_conversations, "increment_daily_surf_count", AsyncMock()),
            patch.object(heartbeat_service.db_conversations, "upsert_proactive_topic", AsyncMock()),
            patch.object(heartbeat_service, "evaluate_heartbeat_gate", AsyncMock(return_value=gate_allowed)),
            patch.object(heartbeat_service, "get_effective_tool_config", AsyncMock(return_value={"tavily_api_key": "tvly-test"})),
            patch.object(heartbeat_service, "fetch_local_weather", AsyncMock(return_value="晴 20°C")),
            patch.object(heartbeat_service, "_load_partition_history_for_heartbeat", AsyncMock(return_value=[{"role": "user", "content": "下午好"}])),
            patch.object(heartbeat_service, "_record_heartbeat_attempt_safe", AsyncMock()),
            patch.object(heartbeat_service, "execute_background_tool_call", AsyncMock(return_value='{"ok": true}')),
            patch.object(chat_routes, "is_session_chat_busy", return_value=False),
            patch.object(chat_routes, "_post_chat_completion_with_429_retry", fake_post),
        ]

    async def test_heartbeat_recovers_from_1425_thought_leak_preserving_prefix(self):
        captured_bodies = []
        responses = [
            # Step 0: normal tool call (search_memories)
            _FakeResponse(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "search_memories", "arguments": '{"query":"drama"}'}}],
                    }
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }),
            # Step 1 (fault turn): leaked English reasoning draft, NO tool_calls
            _FakeResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": INCIDENT_1425_DRAFT}}],
                "usage": {"prompt_tokens": 150, "completion_tokens": 45, "total_tokens": 195},
            }),
            # Step 1 conservative retry: returns browse_xiaohongshu tool call!
            _FakeResponse(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "tc2", "type": "function", "function": {"name": "browse_xiaohongshu", "arguments": '{"keyword":"热门"}'}}],
                    }
                }],
                "usage": {"prompt_tokens": 150, "completion_tokens": 22, "total_tokens": 172},
            }),
            # Step 2: normal Chinese proactive reply
            _FakeResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "阿狸，我刚刷小红书看到大家都在聊这个，笑死我了～"}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
            }),
        ]

        async def fake_post(client, url, api_key, req_body, **kwargs):
            captured_bodies.append(json.loads(json.dumps(req_body)))
            return responses.pop(0)

        save_mock = AsyncMock(return_value=777)
        patches = self._base_patches(fake_post, save_mock)
        for p in patches:
            p.start()
        try:
            res = await heartbeat_service._run_heartbeat_tick_locked(timeout_seconds=90.0)
        finally:
            for p in reversed(patches):
                p.stop()

        self.assertEqual(res["status"], "delivered")
        self.assertEqual(len(captured_bodies), 4)
        # Prefix cache iron rule: fault turn (idx 1) and retry (idx 2) have byte-identical messages and tools!
        self.assertEqual(captured_bodies[1]["messages"], captured_bodies[2]["messages"])
        self.assertEqual(captured_bodies[1]["tools"], captured_bodies[2]["tools"])
        self.assertEqual(captured_bodies[2]["google"]["thinking_config"]["thinking_level"], "low")
        # Leaked draft never persisted to DB; only clean Chinese message persisted
        save_mock.assert_awaited_once()
        saved_content = save_mock.call_args.args[2]
        self.assertIn("阿狸，我刚刷小红书", saved_content)
        self.assertNotIn("Aha!!", saved_content)

    async def test_heartbeat_silent_circuit_breaker_when_retry_also_leaks(self):
        captured_bodies = []
        responses = [
            _FakeResponse(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "search_memories", "arguments": '{}'}}],
                    }
                }],
            }),
            # Fault turn
            _FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": INCIDENT_1425_DRAFT}}]}),
            # Retry also leaks
            _FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": INCIDENT_1425_DRAFT}}]}),
        ]

        async def fake_post(client, url, api_key, req_body, **kwargs):
            captured_bodies.append(json.loads(json.dumps(req_body)))
            return responses.pop(0)

        save_mock = AsyncMock(return_value=999)
        patches = self._base_patches(fake_post, save_mock)
        for p in patches:
            p.start()
        try:
            res = await heartbeat_service._run_heartbeat_tick_locked(timeout_seconds=90.0)
        finally:
            for p in reversed(patches):
                p.stop()

        self.assertEqual(res["status"], "aborted_thought_leak")
        self.assertEqual(res["retries"], 1)
        # Strictly capped at 1 retry (1 initial + 1 fault + 1 retry = 3 upstream calls total)
        self.assertEqual(len(captured_bodies), 3)
        # Zero DB persistence!
        save_mock.assert_not_awaited()

    async def test_heartbeat_strips_extra_content_thought_only_response(self):
        responses = [
            _FakeResponse(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "search_memories", "arguments": '{}'}}],
                    }
                }],
            }),
            # Upstream returns thought-only message via extra_content.google.thought
            _FakeResponse(200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Internal English thought tagged by extra_content",
                        "extra_content": {"google": {"thought": True}},
                    }
                }],
            }),
            # Retry recovers with normal Chinese reply
            _FakeResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "阿狸在干嘛呢？"}}],
            }),
        ]

        async def fake_post(client, url, api_key, req_body, **kwargs):
            return responses.pop(0)

        save_mock = AsyncMock(return_value=888)
        patches = self._base_patches(fake_post, save_mock)
        for p in patches:
            p.start()
        try:
            res = await heartbeat_service._run_heartbeat_tick_locked(timeout_seconds=90.0)
        finally:
            for p in reversed(patches):
                p.stop()

        self.assertEqual(res["status"], "delivered")
        save_mock.assert_awaited_once()
        self.assertEqual(save_mock.call_args.args[2], "阿狸在干嘛呢？")


class StreamChatThoughtLeakAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_intercepts_post_tool_leaked_draft_and_rescues_tool_call(self):
        stream_lines = [
            'data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Aha!! So *that\'s* what happened! "}}]}',
            'data: {"choices": [{"index": 0, "delta": {"content": "Let me call browse_xiaohongshu to search and see what everyone is talking about! (Note: I just used 1 turn for search_memories.)"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]
        fake_stream_resp = _FakeStreamResponse(stream_lines)

        captured_retry_bodies = []
        rescued_tc = [{"id": "tc_rescued", "type": "function", "function": {"name": "browse_xiaohongshu", "arguments": '{"keyword":"test"}'}}]

        async def fake_send_stream(client, url, headers, body, **kwargs):
            return fake_stream_resp, 0

        async def fake_post_retry(client, url, api_key, body, **kwargs):
            captured_retry_bodies.append(json.loads(json.dumps(body)))
            return _FakeResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": rescued_tc}, "finish_reason": "tool_calls"}]
            })

        req_body = {
            "model": "google/gemini-2.5-pro",
            "stream": True,
            "tools": SAMPLE_TOOLS,
            "google": {"thinking_config": {"thinking_level": "medium", "include_thoughts": True}},
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "帮我看看小红书"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "search_memories", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "tc1", "content": '{"items": []}'},
            ],
        }
        mem_bg_mock = AsyncMock()

        with patch.object(chat_routes, "THOUGHT_LEAK_COOLDOWN_SECONDS", 0.0), \
             patch.object(shared, "is_vertex_endpoint", return_value=True), \
             patch.object(shared, "get_vertex_access_token", return_value="tok"), \
             patch.object(shared, "API_KEY", "test-key"), \
             patch.object(shared, "API_BASE_URL", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi/chat/completions"), \
             patch.object(shared, "conversation_persistence_enabled", return_value=True), \
             patch.object(chat_routes, "_send_stream_with_429_retry", fake_send_stream), \
             patch.object(chat_routes, "_post_chat_completion_with_429_retry", fake_post_retry), \
             patch.object(chat_routes, "_mark_pending_seen", AsyncMock()), \
             patch.object(chat_routes.memory_pipeline, "process_memories_background", mem_bg_mock):
            chunks = []
            async for chunk in chat_routes._stream_and_capture_inner(
                {"Authorization": "Bearer x"},
                req_body,
                "s1",
                "帮我看看小红书",
                "google/gemini-2.5-pro",
                tool_messages=[{"role": "tool", "tool_call_id": "tc1", "content": '{"items": []}'}],
                stream_state={"terminal": False, "phase": "", "completed": False},
                req_id="t1",
            ):
                chunks.append(chunk.decode("utf-8", errors="ignore"))

        output = "".join(chunks)
        # Client NEVER received a single byte of the leaked English draft!
        self.assertNotIn("Aha!!", output)
        self.assertNotIn("Let me call browse_xiaohongshu", output)
        # Client received the rescued structured tool call!
        self.assertIn("tc_rescued", output)
        self.assertIn("data: [DONE]", output)
        # Prefix cache preserved: retry kept exact same messages and tools!
        self.assertEqual(len(captured_retry_bodies), 1)
        self.assertEqual(captured_retry_bodies[0]["messages"], req_body["messages"])
        self.assertEqual(captured_retry_bodies[0]["tools"], req_body["tools"])

    async def test_stream_thought_leak_stops_after_one_retry_and_skips_db_on_failure(self):
        stream_lines = [
            f'data: {json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": INCIDENT_1425_DRAFT}}]})}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]
        fake_stream_resp = _FakeStreamResponse(stream_lines)
        retry_calls = 0

        async def fake_send_stream(client, url, headers, body, **kwargs):
            return fake_stream_resp, 0

        async def fake_post_retry(client, url, api_key, body, **kwargs):
            nonlocal retry_calls
            retry_calls += 1
            # Retry ALSO leaks the English draft!
            return _FakeResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": INCIDENT_1425_DRAFT}, "finish_reason": "stop"}]
            })

        req_body = {
            "model": "google/gemini-2.5-pro",
            "stream": True,
            "tools": SAMPLE_TOOLS,
            "messages": [
                {"role": "user", "content": "帮我看看小红书"},
                {"role": "tool", "tool_call_id": "tc1", "content": "{}"},
            ],
        }
        mem_bg_mock = AsyncMock()
        stream_state = {"terminal": False, "phase": "", "completed": False}

        with patch.object(chat_routes, "THOUGHT_LEAK_COOLDOWN_SECONDS", 0.0), \
             patch.object(shared, "is_vertex_endpoint", return_value=True), \
             patch.object(shared, "get_vertex_access_token", return_value="tok"), \
             patch.object(shared, "API_KEY", "test-key"), \
             patch.object(shared, "API_BASE_URL", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi/chat/completions"), \
             patch.object(shared, "conversation_persistence_enabled", return_value=True), \
             patch.object(chat_routes, "_send_stream_with_429_retry", fake_send_stream), \
             patch.object(chat_routes, "_post_chat_completion_with_429_retry", fake_post_retry), \
             patch.object(chat_routes, "_mark_pending_seen", AsyncMock()), \
             patch.object(chat_routes.memory_pipeline, "process_memories_background", mem_bg_mock):
            chunks = []
            async for chunk in chat_routes._stream_and_capture_inner(
                {"Authorization": "Bearer x"},
                req_body,
                "s1",
                "帮我看看小红书",
                "google/gemini-2.5-pro",
                tool_messages=[{"role": "tool", "tool_call_id": "tc1", "content": "{}"}],
                stream_state=stream_state,
                req_id="t2",
            ):
                chunks.append(chunk.decode("utf-8", errors="ignore"))

        output = "".join(chunks)
        # Strictly capped at 1 retry (no Phase 2 cascade!)
        self.assertEqual(retry_calls, 1)
        # Never leaked to client
        self.assertNotIn("Aha!!", output)
        self.assertIn("fallback_failed", output)
        self.assertIn("data: [DONE]", output)
        # Never persisted to DB!
        mem_bg_mock.assert_not_called()
        self.assertFalse(stream_state["terminal"])

    async def test_stream_normal_chinese_and_cuplivo_ui_pass_through_immediately(self):
        stream_lines = [
            'data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "给你做了个卡片：\\n```cuplivo-ui\\n<div>Hi</div>\\n```"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]
        fake_stream_resp = _FakeStreamResponse(stream_lines)

        async def fake_send_stream(client, url, headers, body, **kwargs):
            return fake_stream_resp, 0

        retry_mock = AsyncMock()
        req_body = {
            "model": "google/gemini-2.5-pro",
            "stream": True,
            "tools": SAMPLE_TOOLS,
            "messages": [
                {"role": "user", "content": "做个卡片"},
                {"role": "tool", "tool_call_id": "tc1", "content": "{}"},
            ],
        }

        with patch.object(shared, "is_vertex_endpoint", return_value=False), \
             patch.object(shared, "conversation_persistence_enabled", return_value=False), \
             patch.object(chat_routes, "_send_stream_with_429_retry", fake_send_stream), \
             patch.object(chat_routes, "_post_chat_completion_with_429_retry", retry_mock), \
             patch.object(chat_routes, "_mark_pending_seen", AsyncMock()):
            chunks = []
            async for chunk in chat_routes._stream_and_capture_inner(
                {"Authorization": "Bearer x"},
                req_body,
                "s1",
                "做个卡片",
                "google/gemini-2.5-pro",
                tool_messages=[{"role": "tool", "tool_call_id": "tc1", "content": "{}"}],
                stream_state={"terminal": False, "phase": "", "completed": False},
                req_id="t3",
            ):
                chunks.append(chunk.decode("utf-8", errors="ignore"))

        output = "".join(chunks)
        self.assertIn("cuplivo-ui", output)
        self.assertIn("给你做了个卡片", output)
        retry_mock.assert_not_called()


class NonStreamChatThoughtLeakAsyncTests(unittest.IsolatedAsyncioTestCase):
    def _make_request(self, body: dict) -> Request:
        raw_body = json.dumps(body).encode("utf-8")

        async def mock_receive():
            return {"type": "http.request", "body": raw_body, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
        }
        return Request(scope, receive=mock_receive)

    async def test_nonstream_thought_leak_retries_once_and_returns_502_without_db_write_on_repeat(self):
        req_body = {
            "model": "google/gemini-2.5-pro",
            "stream": False,
            "tools": SAMPLE_TOOLS,
            "messages": [
                {"role": "user", "content": "帮我查小红书"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "search_memories", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "tc1", "content": '{"items": []}'},
            ],
        }
        captured_bodies = []

        async def fake_post(client, url, api_key, body, **kwargs):
            captured_bodies.append(json.loads(json.dumps(body)))
            return _FakeResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": INCIDENT_1425_DRAFT}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            })

        mem_bg_mock = AsyncMock()
        with patch.object(chat_routes, "THOUGHT_LEAK_COOLDOWN_SECONDS", 0.0), \
             patch.object(shared, "is_vertex_endpoint", return_value=True), \
             patch.object(shared, "get_vertex_access_token", return_value="tok"), \
             patch.object(shared, "API_KEY", "test-key"), \
             patch.object(shared, "API_BASE_URL", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi/chat/completions"), \
             patch.object(shared, "DATABASE_ENABLED", False), \
             patch.object(shared, "MEMORY_ENABLED", False), \
             patch.object(shared, "CACHE_PARTITION_ENABLED", False), \
             patch.object(shared, "conversation_persistence_enabled", return_value=True), \
             patch.object(shared, "get_system_prompt", AsyncMock(return_value="")), \
             patch("asyncio.sleep", AsyncMock()), \
             patch.object(chat_routes, "_post_chat_completion_with_429_retry", fake_post), \
             patch.object(chat_routes.memory_pipeline, "process_memories_background", mem_bg_mock):
            resp = await chat_routes.chat_completions(self._make_request(req_body))

        self.assertEqual(resp.status_code, 502)
        payload = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "thought_leak_intercepted")
        # 1 initial + 1 conservative retry = 2 calls, prefix byte-identical
        self.assertEqual(len(captured_bodies), 2)
        self.assertEqual(captured_bodies[0]["messages"], captured_bodies[1]["messages"])
        self.assertEqual(captured_bodies[0]["tools"], captured_bodies[1]["tools"])
        # Zero DB write!
        mem_bg_mock.assert_not_called()


class ProactiveEventPrefixCacheTests(unittest.TestCase):
    def test_b_zone_proactive_events_are_append_only_and_a_zone_strips_them(self):
        base_b_msgs = [
            {"role": "user", "content": "早安"},
            {
                "role": "assistant",
                "content": "手账1",
                "message_type": "proactive_event",
                "created_at": datetime(2026, 11, 26, 1, 0, tzinfo=timezone.utc),
                "proactive_meta": {"event_title": "第一件小事"},
            },
            {"role": "user", "content": "中午吃啥"},
            {
                "role": "assistant",
                "content": "手账2",
                "message_type": "proactive_event",
                "created_at": datetime(2026, 11, 26, 4, 0, tzinfo=timezone.utc),
                "proactive_meta": {"event_title": "第二件小事"},
            },
            {"role": "user", "content": "下午好"},
            {"role": "assistant", "content": "下午好呀"},
        ]
        before_b = partition_engine._prepare_zone_messages(
            base_b_msgs, drop_tool_calls=False, drop_empty_assistant=False, drop_proactive_events=False
        )

        # Append a 3rd proactive_event after a new user turn: earlier B-zone prefix must remain byte-identical!
        extended_b_msgs = base_b_msgs + [
            {"role": "user", "content": "晚上见"},
            {
                "role": "assistant",
                "content": "手账3",
                "message_type": "proactive_event",
                "created_at": datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc),
                "proactive_meta": {"event_title": "第三件小事"},
            },
        ]
        after_b = partition_engine._prepare_zone_messages(
            extended_b_msgs, drop_tool_calls=False, drop_empty_assistant=False, drop_proactive_events=False
        )

        self.assertEqual(after_b[:len(before_b)], before_b)
        self.assertTrue(any("第一件小事" in (m.get("content") or "") for m in after_b))
        self.assertTrue(any("第三件小事" in (m.get("content") or "") for m in after_b))

        # Once rotated into A-zone, all old proactive_event messages are cleanly stripped
        cleaned_a = partition_engine._prepare_zone_messages(
            extended_b_msgs, drop_tool_calls=True, drop_empty_assistant=True, drop_proactive_events=True
        )
        self.assertFalse(any("后台手账" in (m.get("content") or "") for m in cleaned_a))
        self.assertTrue(any("下午好呀" in (m.get("content") or "") for m in cleaned_a))


if __name__ == "__main__":
    unittest.main()
