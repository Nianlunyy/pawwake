"""Unit tests for non-streaming 429 retry mechanism (fix/nonstream-429-retry).

Verifies:
a. mock upstream returns 429 twice then 200: non-streaming request returns 200, log contains "429 重试成功", upstream called 3 times.
b. mock upstream returns 429 three times: returns 429, body is upstream verbatim, upstream called 3 times.
c. non-Vertex URL: 429 does not retry, upstream called only 1 time.
d. actual payload sent to upstream is character-by-character identical to main (comparing entire body).
e. non-streaming request with last message role=tool: 1.5s buffer preserved, 429 retry works normally.
"""

import asyncio
import io
import json
import unittest
from unittest.mock import patch

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


class TestNonstream429Retry(unittest.TestCase):
    def setUp(self):
        chat_routes._LAST_REQUEST_CACHE.clear()

    def tearDown(self):
        chat_routes._LAST_REQUEST_CACHE.clear()

    def test_a_mock_upstream_429_twice_then_200(self):
        """a. mock 上游先两次 429 再 200：非流式请求最终返回 200，日志出现 '429 重试成功'，上游被调用 3 次。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好，测试非流式429重试"}],
            "stream": False,
            "req_id": "test_a",
        }
        request = _make_mock_request(req_body)

        call_count = 0
        recorded_sleeps = []
        real_sleep = asyncio.sleep

        async def mock_sleep(delay, *args, **kwargs):
            recorded_sleeps.append(delay)
            await real_sleep(0.001)

        fake_success_data = {
            "choices": [{"message": {"role": "assistant", "content": "重试成功后的非流式回复"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }

        async def mock_post(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(
                    status_code=429,
                    headers={"content-type": "application/json", "retry-after": "0.1"},
                    json={"error": {"code": 429, "message": "Resource exhausted"}},
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json=fake_success_data,
                request=httpx.Request("POST", url),
            )

        buf = io.StringIO()
        vertex_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini-2.5-flash:predict"
        with patch("sys.stdout", buf):
            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("httpx.AsyncClient.post", side_effect=mock_post):
                    with patch.object(shared, "API_BASE_URL", vertex_url):
                        with patch.object(shared, "API_KEY", "test-vertex-key"):
                            with patch.object(shared, "get_vertex_access_token", return_value="mock-token-123"):
                                with patch.object(shared, "DATABASE_ENABLED", False):
                                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                        with patch.object(shared, "FORCE_STREAM", False):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        out = buf.getvalue()
        print(f"用例 a 输出:\n{out}")

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 200)
        resp_data = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(resp_data["choices"][0]["message"]["content"], "重试成功后的非流式回复")
        self.assertEqual(call_count, 3)
        self.assertIn("429 重试成功", out)
        self.assertIn("[非流式] 429 重试成功（第 2 次）", out)

    def test_b_mock_upstream_429_three_times(self):
        """b. mock 上游三次都 429：返回 429，body 为上游原样，上游被调用 3 次。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好，测试三次429耗尽"}],
            "stream": False,
            "req_id": "test_b",
        }
        request = _make_mock_request(req_body)

        call_count = 0
        real_sleep = asyncio.sleep

        async def mock_sleep(delay, *args, **kwargs):
            await real_sleep(0.001)

        upstream_error_body = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
            }
        }

        async def mock_post(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                status_code=429,
                headers={"content-type": "application/json", "retry-after": "0.1"},
                json=upstream_error_body,
                request=httpx.Request("POST", url),
            )

        buf = io.StringIO()
        vertex_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini-2.5-flash:predict"
        with patch("sys.stdout", buf):
            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("httpx.AsyncClient.post", side_effect=mock_post):
                    with patch.object(shared, "API_BASE_URL", vertex_url):
                        with patch.object(shared, "API_KEY", "test-vertex-key"):
                            with patch.object(shared, "get_vertex_access_token", return_value="mock-token-123"):
                                with patch.object(shared, "DATABASE_ENABLED", False):
                                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                        with patch.object(shared, "FORCE_STREAM", False):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        out = buf.getvalue()
        print(f"用例 b 输出:\n{out}")

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 429)
        resp_data = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(resp_data, upstream_error_body)
        self.assertEqual(call_count, 3)
        self.assertNotIn("429 重试成功", out)

    def test_c_non_vertex_url_429_no_retry(self):
        """c. 非 Vertex 的 URL：429 不重试，上游只调用 1 次。"""
        req_body = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "测试非Vertex无需重试"}],
            "stream": False,
            "req_id": "test_c",
        }
        request = _make_mock_request(req_body)

        call_count = 0
        upstream_error_body = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "requests",
                "code": 429,
            }
        }

        async def mock_post(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                status_code=429,
                headers={"content-type": "application/json"},
                json=upstream_error_body,
                request=httpx.Request("POST", url),
            )

        buf = io.StringIO()
        non_vertex_url = "https://openrouter.ai/api/v1/chat/completions"
        with patch("sys.stdout", buf):
            with patch("httpx.AsyncClient.post", side_effect=mock_post):
                with patch.object(shared, "API_BASE_URL", non_vertex_url):
                    with patch.object(shared, "API_KEY", "test-openrouter-key"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                with patch.object(shared, "FORCE_STREAM", False):
                                    resp = asyncio.run(chat_routes._chat_completions_inner(request))

        out = buf.getvalue()
        print(f"用例 c 输出:\n{out}")

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 429)
        resp_data = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(resp_data, upstream_error_body)
        self.assertEqual(call_count, 1)
        self.assertNotIn("429 重试成功", out)
        self.assertNotIn("次重试", out)

    def test_d_payload_to_upstream_verbatim_identical(self):
        """d. 实际发往上游的 payload 与 main 逐字相同（比较整个 body，不是只比键集合）。"""
        req_body = {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "你好，这是一条完整的测试请求体"},
            ],
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
            "req_id": "test_d",
        }
        request = _make_mock_request(req_body)

        captured_kwargs = {}

        fake_resp_data = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

        async def mock_post(url, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json=fake_resp_data,
                request=httpx.Request("POST", url),
            )

        vertex_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini-2.5-flash:predict"
        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            with patch.object(shared, "API_BASE_URL", vertex_url):
                with patch.object(shared, "API_KEY", "test-vertex-key"):
                    with patch.object(shared, "get_vertex_access_token", return_value="mock-token-123"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                with patch.object(shared, "FORCE_STREAM", False):
                                    with patch.object(shared, "REASONING_EFFORT", None):
                                        resp = asyncio.run(chat_routes._chat_completions_inner(request))

        self.assertIn("json", captured_kwargs)
        actual_payload = captured_kwargs["json"]

        # 手写推演 main 在相同环境下组装并发送给 client.post 的预期 body：
        # 1. req_id 从 body 中 pop 移除
        # 2. messages 经由单轮直发逻辑原样保留
        # 3. is_vertex_endpoint 为 True，model 前缀补齐为 google/gemini-2.5-flash
        # 4. 其他字段 stream, temperature, top_p, max_tokens 原样保留
        expected_main_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "你好，这是一条完整的测试请求体"},
            ],
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
        }

        # 逐字比较整个 body（结构与键值完全相等）
        self.assertEqual(actual_payload, expected_main_body)
        # 序列化为 JSON 字符串对比，确保字典内容与各字段值完全逐字一致
        self.assertEqual(
            json.dumps(actual_payload, ensure_ascii=False, sort_keys=True),
            json.dumps(expected_main_body, ensure_ascii=False, sort_keys=True),
        )

        # 严格验证禁止包含任何内部或重试参数
        forbidden_keys = {"req_id", "tag", "max_retries", "retry", "attempt"}
        for k in forbidden_keys:
            self.assertNotIn(k, actual_payload)

    def test_e_last_message_role_tool_buffer_and_retry(self):
        """e. 最后一条消息 role=tool 的非流式请求：1.5 秒缓冲仍在，429 重试正常。"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "调用工具测试"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"result": "success"}'},
            ],
            "stream": False,
            "req_id": "test_e",
        }
        request = _make_mock_request(req_body)

        call_count = 0
        recorded_sleeps = []
        real_sleep = asyncio.sleep

        async def mock_sleep(delay, *args, **kwargs):
            recorded_sleeps.append(delay)
            await real_sleep(0.001)

        fake_success_data = {
            "choices": [{"message": {"role": "assistant", "content": "工具返回后的回复"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        }

        async def mock_post(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    status_code=429,
                    headers={"content-type": "application/json", "retry-after": "0.1"},
                    json={"error": {"code": 429, "message": "Resource exhausted"}},
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                json=fake_success_data,
                request=httpx.Request("POST", url),
            )

        buf = io.StringIO()
        vertex_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini-2.5-flash:predict"
        with patch("sys.stdout", buf):
            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("httpx.AsyncClient.post", side_effect=mock_post):
                    with patch.object(shared, "API_BASE_URL", vertex_url):
                        with patch.object(shared, "API_KEY", "test-vertex-key"):
                            with patch.object(shared, "get_vertex_access_token", return_value="mock-token-123"):
                                with patch.object(shared, "DATABASE_ENABLED", False):
                                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                        with patch.object(shared, "FORCE_STREAM", False):
                                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        out = buf.getvalue()
        print(f"用例 e 输出:\n{out}")

        self.assertIsInstance(resp, JSONResponse)
        self.assertEqual(resp.status_code, 200)
        resp_data = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(resp_data["choices"][0]["message"]["content"], "工具返回后的回复")
        self.assertEqual(call_count, 2)

        # 验证 1.5 秒缓冲仍在
        self.assertIn("Gemini/Vertex tool 返回缓冲: sleep 1.5s 防 429", out)
        self.assertIn(1.5, recorded_sleeps)

        # 验证 429 重试成功
        self.assertIn("429 重试成功", out)
        self.assertIn("[非流式] 429 重试成功（第 1 次）", out)


if __name__ == "__main__":
    unittest.main()
