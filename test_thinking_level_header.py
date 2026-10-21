"""Unit tests for X-Thinking-Level header overriding thinking level in main chat.

Tests:
 1 不带头 → 沿用默认档
 2 合法头覆盖默认档
 3 大小写与空白（" HIGH " → high）
 4 非法值被忽略且不报错
 5 环境变量为空 + 合法头 → 注入；环境变量为空 + 无头 → payload 不含 google 与 reasoning_effort
 6 非 Vertex 端点：reasoning_effort 被覆盖
 7 X-Skip-Conversation-Log: true → 带头也不注入（保持现状）
 8 头值不出现在发往上游的 body 与请求头里
 9 兜底阶段一仍强制 low（复用现有测试，若无则新增）
"""

import asyncio
import io
import json
import logging
import unittest
from unittest.mock import patch, MagicMock

import httpx
from fastapi import Request
from starlette.responses import JSONResponse

import routes.chat as chat_routes
import shared


def _make_mock_request(body: dict, headers: dict = None) -> Request:
    raw_body = json.dumps(body).encode("utf-8")

    async def mock_receive():
        return {
            "type": "http.request",
            "body": raw_body,
            "more_body": False,
        }

    hdrs = [(b"content-type", b"application/json")]
    if headers:
        for k, v in headers.items():
            hdrs.append((k.lower().encode("utf-8"), v.encode("utf-8")))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": hdrs,
    }
    return Request(scope, receive=mock_receive)


class _MockStreamResponse:
    def __init__(self, status_code=200, headers=None, sse_lines=None, chunk_delay=0.0):
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/event-stream"}
        self.sse_lines = sse_lines or []
        self.chunk_delay = chunk_delay
        self.closed = False

    async def aiter_bytes(self):
        for line in self.sse_lines:
            if self.chunk_delay > 0:
                await asyncio.sleep(self.chunk_delay)
            yield (line + "\n\n").encode("utf-8")

    async def aread(self):
        return b""

    async def aclose(self):
        self.closed = True


class TestThinkingLevelHeader(unittest.TestCase):
    def setUp(self):
        chat_routes._LAST_REQUEST_CACHE.clear()
        self.vertex_url = (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/"
            "publishers/google/models/gemini-2.5-flash:predict"
        )
        self.non_vertex_url = "https://api.openai.com/v1/chat/completions"

    def tearDown(self):
        chat_routes._LAST_REQUEST_CACHE.clear()

    def test_1_no_header_uses_default(self):
        """1 不带头 → 沿用默认档"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t1_001",
        }
        request = _make_mock_request(req_body)

        captured_payloads = []
        captured_headers = []

        async def mock_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            captured_headers.append(kwargs.get("headers"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch.object(shared, "REASONING_EFFORT", "medium"):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch.object(shared, "FORCE_STREAM", False):
                                        with patch("httpx.AsyncClient.post", side_effect=mock_post):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        self.assertIn("google", payload)
        self.assertEqual(
            payload["google"]["thinking_config"]["thinking_level"],
            "medium",
        )
        out = buf.getvalue()
        self.assertIn("req=t1_001", out)
        self.assertIn("default", out)

    def test_2_valid_header_overrides_default(self):
        """2 合法头覆盖默认档"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t2_002",
        }
        request = _make_mock_request(req_body, headers={"X-Thinking-Level": "high"})

        captured_payloads = []

        async def mock_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch.object(shared, "REASONING_EFFORT", "low"):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch.object(shared, "FORCE_STREAM", False):
                                        with patch("httpx.AsyncClient.post", side_effect=mock_post):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        self.assertIn("google", payload)
        self.assertEqual(
            payload["google"]["thinking_config"]["thinking_level"],
            "high",
        )
        out = buf.getvalue()
        self.assertIn("req=t2_002", out)
        self.assertIn("header", out)

    def test_3_case_and_whitespace(self):
        """3 大小写与空白（" HIGH " → high）"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t3_003",
        }
        request = _make_mock_request(req_body, headers={"X-Thinking-Level": "  HIGH  "})

        captured_payloads = []

        async def mock_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        with patch("sys.stdout", io.StringIO()):
            with patch.object(shared, "REASONING_EFFORT", "low"):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch.object(shared, "FORCE_STREAM", False):
                                        with patch("httpx.AsyncClient.post", side_effect=mock_post):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        self.assertIn("google", payload)
        self.assertEqual(
            payload["google"]["thinking_config"]["thinking_level"],
            "high",
        )

    def test_4_invalid_value_ignored_and_no_error(self):
        """4 非法值被忽略且不报错"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t4_004",
        }
        request = _make_mock_request(req_body, headers={"X-Thinking-Level": "extreme-turbo"})

        captured_payloads = []

        async def mock_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        with self.assertLogs(chat_routes.logger, level="WARNING") as cm:
            with patch("sys.stdout", io.StringIO()):
                with patch.object(shared, "REASONING_EFFORT", "medium"):
                    with patch.object(shared, "API_BASE_URL", self.vertex_url):
                        with patch.object(shared, "API_KEY", "test-key"):
                            with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                                with patch.object(shared, "DATABASE_ENABLED", False):
                                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                        with patch.object(shared, "FORCE_STREAM", False):
                                            with patch("httpx.AsyncClient.post", side_effect=mock_post):
                                                resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        self.assertIn("google", payload)
        self.assertEqual(
            payload["google"]["thinking_config"]["thinking_level"],
            "medium",
        )
        warning_output = " ".join(cm.output)
        self.assertIn("extreme-turbo", warning_output)

    def test_5_env_empty_behavior(self):
        """5 环境变量为空 + 合法头 → 注入；环境变量为空 + 无头 → payload 不含 google 与 reasoning_effort"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t5_005",
        }

        # 5a: 环境变量为空 + 合法头 → 注入
        request_with_header = _make_mock_request(req_body, headers={"X-Thinking-Level": "minimal"})
        captured_a = []

        async def mock_post_a(url, *args, **kwargs):
            captured_a.append(kwargs.get("json"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        with patch("sys.stdout", io.StringIO()):
            with patch.object(shared, "REASONING_EFFORT", ""):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch.object(shared, "FORCE_STREAM", False):
                                        with patch("httpx.AsyncClient.post", side_effect=mock_post_a):
                                            resp_a = asyncio.run(chat_routes._chat_completions_inner(request_with_header))

        self.assertEqual(resp_a.status_code, 200)
        self.assertEqual(len(captured_a), 1)
        self.assertIn("google", captured_a[0])
        self.assertEqual(
            captured_a[0]["google"]["thinking_config"]["thinking_level"],
            "minimal",
        )

        # 5b: 环境变量为空 + 无头 → payload 不含 google 与 reasoning_effort
        request_no_header = _make_mock_request(req_body)
        captured_b = []

        async def mock_post_b(url, *args, **kwargs):
            captured_b.append(kwargs.get("json"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        with patch("sys.stdout", io.StringIO()):
            with patch.object(shared, "REASONING_EFFORT", ""):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch.object(shared, "FORCE_STREAM", False):
                                        with patch("httpx.AsyncClient.post", side_effect=mock_post_b):
                                            resp_b = asyncio.run(chat_routes._chat_completions_inner(request_no_header))

        self.assertEqual(resp_b.status_code, 200)
        self.assertEqual(len(captured_b), 1)
        self.assertNotIn("google", captured_b[0])
        self.assertNotIn("reasoning_effort", captured_b[0])

    def test_6_non_vertex_endpoint(self):
        """6 非 Vertex 端点：reasoning_effort 被覆盖"""
        req_body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t6_006",
        }
        request = _make_mock_request(req_body, headers={"X-Thinking-Level": "high"})
        captured_payloads = []

        async def mock_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch.object(shared, "REASONING_EFFORT", "low"):
                with patch.object(shared, "API_BASE_URL", self.non_vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                with patch.object(shared, "FORCE_STREAM", False):
                                    with patch("httpx.AsyncClient.post", side_effect=mock_post):
                                        resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        self.assertEqual(payload.get("reasoning_effort"), "high")
        self.assertNotIn("google", payload)
        out = buf.getvalue()
        self.assertIn("req=t6_006", out)
        self.assertIn("header", out)

    def test_7_skip_conversation_log_priority(self):
        """7 X-Skip-Conversation-Log: true → 带头也不注入（保持现状）"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t7_007",
        }
        request = _make_mock_request(
            req_body,
            headers={"X-Thinking-Level": "high", "X-Skip-Conversation-Log": "true"},
        )
        captured_payloads = []

        async def mock_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        with patch("sys.stdout", io.StringIO()):
            with patch.object(shared, "REASONING_EFFORT", "low"):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch.object(shared, "FORCE_STREAM", False):
                                        with patch("httpx.AsyncClient.post", side_effect=mock_post):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        self.assertNotIn("google", payload)
        self.assertNotIn("reasoning_effort", payload)

    def test_8_header_not_leaked_upstream(self):
        """8 头值不出现在发往上游的 body 与请求头里"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
            "req_id": "t8_008",
        }
        request = _make_mock_request(req_body, headers={"X-Thinking-Level": "high"})
        captured_payloads = []
        captured_headers = []

        async def mock_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json"))
            captured_headers.append(kwargs.get("headers"))
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                request=httpx.Request("POST", url),
            )

        with patch("sys.stdout", io.StringIO()):
            with patch.object(shared, "REASONING_EFFORT", "low"):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch.object(shared, "FORCE_STREAM", False):
                                        with patch("httpx.AsyncClient.post", side_effect=mock_post):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(captured_payloads), 1)
        self.assertEqual(len(captured_headers), 1)
        p = captured_payloads[0]
        h = captured_headers[0]

        # 检查发往上游的请求头
        h_keys_lower = [k.lower() for k in h.keys()]
        self.assertNotIn("x-thinking-level", h_keys_lower)
        self.assertNotIn("x_thinking_level", h_keys_lower)
        self.assertNotIn("thinking-level", h_keys_lower)

        # 检查发往上游的请求体整个结构
        p_keys_lower = [str(k).lower() for k in p.keys()]
        self.assertNotIn("x-thinking-level", p_keys_lower)
        self.assertNotIn("x_thinking_level", p_keys_lower)
        self.assertNotIn("thinking_level", p_keys_lower)

    def test_9_fallback_phase1_still_forces_low(self):
        """9 兜底阶段一仍强制 low（复用现有测试，若无则新增）"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "测试兜底阶段一强制low"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "stream": True,
            "req_id": "t9_009",
        }
        request = _make_mock_request(req_body, headers={"X-Thinking-Level": "high"})

        malformed_sse = [
            'data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "thinking"}, "finish_reason": "malformed_function_call"}]}',
            'data: [DONE]',
        ]

        async def mock_stream_send(req, stream=True):
            return _MockStreamResponse(200, sse_lines=malformed_sse)

        captured_phase1_payloads = []

        async def mock_phase1_post(client, url, api_key, payload):
            captured_phase1_payloads.append(dict(payload))
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            return resp

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            async for _ in resp.body_iterator:
                pass

        with patch("sys.stdout", io.StringIO()):
            with patch.object(shared, "REASONING_EFFORT", "medium"):
                with patch.object(shared, "API_BASE_URL", self.vertex_url):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "get_vertex_access_token", return_value="mock-token"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    with patch("httpx.AsyncClient.send", side_effect=mock_stream_send):
                                        with patch("shared.post_chat_completion", side_effect=mock_phase1_post):
                                            asyncio.run(run_flow())

        self.assertGreaterEqual(len(captured_phase1_payloads), 1)
        phase1_payload = captured_phase1_payloads[0]
        self.assertIn("google", phase1_payload)
        self.assertEqual(
            phase1_payload["google"]["thinking_config"]["thinking_level"],
            "low",
            "兜底阶段一必须强制 low，即使请求头传入了 high",
        )


if __name__ == "__main__":
    unittest.main()
