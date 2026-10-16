"""Tests for upstream wait keepalive and timeout cap (fix/upstream-wait-keepalive).

Validates:
a. Mock upstream delay N seconds before headers: client receives ': keep-alive' then normal data.
b. Mock upstream 429 retry then success during wait: keep-alive continues, final response normal.
c. Small wait cap: timeout produces SSE error frame, upstream cancelled, no DB/memory persistence.
d. Client disconnect during wait: upstream cancelled, plug log emitted.
e. Non-streaming request: no keep-alive, behavior matches main.
f. Assert payload sent to upstream matches main verbatim (no new fields).
g. Normal fast response (<15s): no keep-alive sent, output matches main.
"""

import asyncio
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import Request
from fastapi.responses import JSONResponse
import httpx

import shared
import routes.chat as chat_routes


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


class TestUpstreamWaitKeepalive(unittest.TestCase):
    def setUp(self):
        chat_routes._LAST_REQUEST_CACHE.clear()

    def tearDown(self):
        chat_routes._LAST_REQUEST_CACHE.clear()

    def test_a_delayed_response_keepalive_then_normal(self):
        """a. mock 上游延迟 N 秒才回响应头（间隔常量改成 0.1 秒）：客户端先收到若干 ': keep-alive'，再收到正常数据。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
            "req_id": "test_a",
        }
        request = _make_mock_request(req_body)

        sse_lines = [
            'data: {"choices": [{"index": 0, "delta": {"content": "正常数据"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]

        async def mock_client_send(req, stream=True):
            await asyncio.sleep(0.25)
            return _MockStreamResponse(200, sse_lines=sse_lines)

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
            return resp, chunks

        buf = io.StringIO()
        with patch.dict("os.environ", {"UPSTREAM_KEEPALIVE_INTERVAL": "0.1"}):
            with patch("sys.stdout", buf):
                with patch("httpx.AsyncClient.send", side_effect=mock_client_send):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                resp, chunks = asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例 a 输出:\n{out}")

        keepalive_chunks = [c for c in chunks if c == b": keep-alive\n\n"]
        data_chunks = [c for c in chunks if c != b": keep-alive\n\n"]

        # 客户端先收到若干保活帧
        self.assertGreaterEqual(len(keepalive_chunks), 2)
        self.assertEqual(chunks[0], b": keep-alive\n\n")
        self.assertEqual(chunks[1], b": keep-alive\n\n")

        # 随后收到正常数据
        self.assertTrue(any(b"\xe6\xad\xa3\xe5\xb8\xb8\xe6\x95\xb0\xe6\x8d\xae" in c or b"\xe6\xad\xa3\xe5\xb8\xb8\xe6\x95\xb0\xe6\x8d\xae" in c for c in data_chunks))
        self.assertTrue(any(b"[DONE]" in c for c in data_chunks))

        # 验证诊断日志
        self.assertIn("📡 req=test_a 等上游响应头已 0.1 秒，开始发保活", out)
        self.assertRegex(out, r"📡 req=test_a 保活共 \d+ 次，响应头在 0\.\d+ 秒到达")
        self.assertNotIn("🔌", out)
        self.assertNotIn("🔥", out)

    def test_b_keepalive_during_429_retry_then_success(self):
        """b. 保活期间 mock 上游返回 429 再成功：保活持续，最终正常。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "429重试测试"}],
            "stream": True,
            "req_id": "test_b",
        }
        request = _make_mock_request(req_body)

        call_count = 0
        sse_lines = [
            'data: {"choices": [{"index": 0, "delta": {"content": "429重试后成功输出"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]

        async def mock_client_send(req, stream=True):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.06)
                return _MockStreamResponse(429, headers={"retry-after": "0.15"})
            await asyncio.sleep(0.06)
            return _MockStreamResponse(200, sse_lines=sse_lines)

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
            return resp, chunks

        buf = io.StringIO()
        with patch.dict("os.environ", {"UPSTREAM_KEEPALIVE_INTERVAL": "0.08"}):
            with patch("sys.stdout", buf):
                with patch("httpx.AsyncClient.send", side_effect=mock_client_send):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                with patch.object(shared, "is_vertex_endpoint", return_value=True):
                                    with patch.object(shared, "get_vertex_access_token", return_value="mock-token-xyz"):
                                        resp, chunks = asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例 b 输出:\n{out}")

        keepalive_chunks = [c for c in chunks if c == b": keep-alive\n\n"]
        data_chunks = [c for c in chunks if c != b": keep-alive\n\n"]

        # 保活持续
        self.assertGreaterEqual(len(keepalive_chunks), 2)
        # 最终收到正常数据
        self.assertTrue(any(b"429\xe9\x87\x8d\xe8\xaf\x95\xe5\x90\x8e\xe6\x88\x90\xe5\x8a\x9f\xe8\xbe\x93\xe5\x87\xba" in c for c in data_chunks))

        # 验证日志包含保活、429 重试和最终头到达
        self.assertIn("📡 req=test_b 等上游响应头已 0.08 秒，开始发保活", out)
        self.assertIn("⏳ req=test_b 上游 429", out)
        self.assertIn("✅ 429 重试成功", out)
        self.assertRegex(out, r"📡 req=test_b 保活共 \d+ 次，响应头在 0\.\d+ 秒到达")
        self.assertNotIn("🔌", out)
        self.assertNotIn("🔥", out)

    def test_c_wait_cap_exceeded_error_frame_and_no_persistence(self):
        """c. 上限常量改小：超限后收到错误帧、上游被取消、无落库、无记忆提取调用。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "超限测试"}],
            "stream": True,
            "req_id": "test_c",
        }
        request = _make_mock_request(req_body)

        upstream_cancelled = False

        async def mock_hanging_send(req, stream=True):
            nonlocal upstream_cancelled
            try:
                await asyncio.sleep(5.0)
                return _MockStreamResponse(200)
            except asyncio.CancelledError:
                upstream_cancelled = True
                raise

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
            return resp, chunks

        buf = io.StringIO()
        mock_save_tokens = AsyncMock()
        mock_process_memories = AsyncMock()

        with patch.dict("os.environ", {"UPSTREAM_WAIT_CAP": "0.15", "UPSTREAM_KEEPALIVE_INTERVAL": "0.06"}):
            with patch("sys.stdout", buf):
                with patch("httpx.AsyncClient.send", side_effect=mock_hanging_send):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "DATABASE_ENABLED", True):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                with patch.object(shared, "MEMORY_ENABLED", True):
                                    with patch("db.conversations.save_token_usage", mock_save_tokens):
                                        with patch("memory_pipeline.process_memories_background", mock_process_memories):
                                            resp, chunks = asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例 c 输出:\n{out}")

        # 上游被取消
        self.assertTrue(upstream_cancelled)

        # 收到错误帧与 [DONE]
        raw_text = b"".join(chunks).decode("utf-8")
        self.assertIn("data: [DONE]", raw_text)
        err_lines = [c for c in chunks if b"fallback_failed" in c]
        self.assertEqual(len(err_lines), 1)
        err_line_str = err_lines[0].decode("utf-8").strip()
        self.assertTrue(err_line_str.startswith("data: "))
        err_json = json.loads(err_line_str[6:])
        self.assertEqual(err_json["error"]["message"], "上游 0.15 秒无响应")
        self.assertEqual(err_json["error"]["type"], "fallback_failed")
        self.assertEqual(err_json["error"]["code"], 502)

        # 超限日志
        self.assertIn("⏱ req=test_c 等待超上限 0.15 秒，已取消", out)
        self.assertNotIn("🔌", out)
        self.assertNotIn("🔥", out)

        # 无落库、无记忆提取
        mock_save_tokens.assert_not_called()
        mock_process_memories.assert_not_called()

    def test_d_client_disconnect_during_wait(self):
        """d. 等待期客户端断开：上游被取消，🔌 日志出现。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "断开测试"}],
            "stream": True,
            "req_id": "test_d",
        }
        request = _make_mock_request(req_body)

        upstream_cancelled = False

        async def mock_hanging_send(req, stream=True):
            nonlocal upstream_cancelled
            try:
                await asyncio.sleep(5.0)
                return _MockStreamResponse(200)
            except asyncio.CancelledError:
                upstream_cancelled = True
                raise

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            iterator = resp.body_iterator
            # 读出一个 chunk (keep-alive) 后模拟客户端断开
            async for chunk in iterator:
                break
            await iterator.aclose()

        buf = io.StringIO()
        with patch.dict("os.environ", {"UPSTREAM_KEEPALIVE_INTERVAL": "0.04"}):
            with patch("sys.stdout", buf):
                with patch("httpx.AsyncClient.send", side_effect=mock_hanging_send):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例 d 输出:\n{out}")

        # 上游被取消
        self.assertTrue(upstream_cancelled)
        # 🔌 日志正常输出且阶段为等上游响应头
        self.assertRegex(out, r"🔌 req=test_d 客户端断开: 已用时 \d+\.\d+ 秒, 阶段=等上游响应头")
        self.assertNotIn("🔥", out)

    def test_e_non_streaming_behavior_unaffected(self):
        """e. 非流式请求：无保活，行为与 main 一致。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "非流式测试"}],
            "stream": False,
            "req_id": "test_e",
        }
        request = _make_mock_request(req_body)

        fake_resp_data = {
            "choices": [{"message": {"role": "assistant", "content": "非流式回复"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        class MockHttpxResponse:
            def __init__(self):
                self.status_code = 200
                self.text = json.dumps(fake_resp_data)

            def json(self):
                return fake_resp_data

        async def mock_client_post(*args, **kwargs):
            return MockHttpxResponse()

        buf = io.StringIO()
        with patch.dict("os.environ", {"UPSTREAM_KEEPALIVE_INTERVAL": "0.05"}):
            with patch("sys.stdout", buf):
                with patch("httpx.AsyncClient.post", side_effect=mock_client_post):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                resp = asyncio.run(chat_routes._chat_completions_inner(request))

        out = buf.getvalue()
        print(f"用例 e 输出:\n{out}")

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 200)
        resp_content = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(resp_content["choices"][0]["message"]["content"], "非流式回复")
        self.assertNotIn("keep-alive", out)
        self.assertNotIn("开始发保活", out)

    def test_f_payload_to_upstream_verbatim_identical(self):
        """f. 断言实际发往上游的 payload 与 main 逐字相同（不含任何新字段）。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "测试payload一致性"},
            ],
            "stream": True,
            "temperature": 0.7,
            "req_id": "test_f",
        }
        request = _make_mock_request(req_body)

        captured_payloads = []
        sse_lines = [
            'data: {"choices": [{"index": 0, "delta": {"content": "ok"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]

        async def mock_client_send(req, *args, **kwargs):
            captured_payloads.append(json.loads(req.content.decode("utf-8")))
            return _MockStreamResponse(200, sse_lines=sse_lines)

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            async for _ in resp.body_iterator:
                pass

        with patch("httpx.AsyncClient.send", side_effect=mock_client_send):
            with patch.object(shared, "API_KEY", "test-key"):
                with patch.object(shared, "DATABASE_ENABLED", False):
                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                        asyncio.run(run_flow())

        self.assertEqual(len(captured_payloads), 1)
        actual_payload = captured_payloads[0]

        # 必须不含任何新增字段
        forbidden_keys = {
            "keepalive", "keep_alive", "wait_cap", "upstream_wait",
            "timeout", "UPSTREAM_WAIT_CAP", "UPSTREAM_KEEPALIVE_INTERVAL",
        }
        for k in forbidden_keys:
            self.assertNotIn(k, actual_payload)

        # 验证与 main 的期望结构逐字段严格一致
        expected_keys = {"model", "messages", "stream", "temperature"}
        self.assertEqual(set(actual_payload.keys()), expected_keys)
        self.assertEqual(actual_payload["model"], "google/gemini-2.5-flash")
        self.assertEqual(actual_payload["messages"], [{"role": "user", "content": "测试payload一致性"}])
        self.assertEqual(actual_payload["stream"], True)
        self.assertEqual(actual_payload["temperature"], 0.7)

    def test_g_fast_response_no_keepalive(self):
        """g. 正常快速响应（<15 秒）：不发保活，输出与 main 一致。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "快速响应"}],
            "stream": True,
            "req_id": "test_g",
        }
        request = _make_mock_request(req_body)

        sse_lines = [
            'data: {"choices": [{"index": 0, "delta": {"content": "秒回正文"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]

        async def mock_client_send(req, stream=True):
            await asyncio.sleep(0.01)  # 远小于默认 15s 间隔
            return _MockStreamResponse(200, sse_lines=sse_lines)

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
            return resp, chunks

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch("httpx.AsyncClient.send", side_effect=mock_client_send):
                with patch.object(shared, "API_KEY", "test-key"):
                    with patch.object(shared, "DATABASE_ENABLED", False):
                        with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                            resp, chunks = asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例 g 输出:\n{out}")

        keepalive_chunks = [c for c in chunks if c == b": keep-alive\n\n"]
        # 不发保活
        self.assertEqual(len(keepalive_chunks), 0)

        # 正常接收正文与 [DONE]
        raw_text = b"".join(chunks).decode("utf-8")
        self.assertIn("秒回正文", raw_text)
        self.assertIn("data: [DONE]", raw_text)

        # 日志中不得出现保活相关日志
        self.assertNotIn("开始发保活", out)
        self.assertNotIn("保活共", out)
        self.assertNotIn("等待超上限", out)
        self.assertNotIn("🔌", out)
        self.assertNotIn("🔥", out)


if __name__ == "__main__":
    unittest.main()
