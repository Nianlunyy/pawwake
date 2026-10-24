"""
test_log_convergence.py: 验证日志收敛（非流式80字预览、记忆300字上限、Vertex令牌前缀移除、启动异常脱敏）
"""

import asyncio
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import Request
from starlette.datastructures import Headers

import memory_extractor
import routes.chat as chat_routes
import shared
from main import sanitize_exception_message


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data) if isinstance(data, (dict, list)) else str(data)

    def json(self):
        return self._data


def _make_mock_request(body: dict, headers: dict = None) -> Request:
    header_list = [
        (b"host", b"testserver"),
        (b"content-type", b"application/json"),
    ]
    if headers:
        for k, v in headers.items():
            header_list.append((k.lower().encode("latin1"), v.encode("latin1")))

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "headers": header_list,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    request = Request(scope)
    request._body = json.dumps(body).encode("utf-8")
    return request


class TestLogConvergence(unittest.TestCase):
    def test_vertex_token_log_convergence(self):
        """验证 Vertex 令牌日志只打印已注入和长度，严格不打印任何前缀"""
        token = "ya29.secret_token_1234567890"
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        request = _make_mock_request(req_body)

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            with patch.object(shared, "API_KEY", "test-key"):
                with patch.object(shared, "DATABASE_ENABLED", False):
                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                        with patch.object(shared, "is_vertex_endpoint", return_value=True):
                            with patch.object(shared, "get_vertex_access_token", return_value=token):
                                with patch("routes.chat.stream_and_capture", new_callable=AsyncMock):
                                    resp = asyncio.run(chat_routes.chat_completions(request))

        out = captured_out.getvalue()
        self.assertIn(f"🔑 Vertex Token 已注入, 长度: {len(token)}", out)
        self.assertNotIn("前缀:", out)
        self.assertNotIn(token[:10], out)

    def test_nonstream_long_reply_preview_convergence(self):
        """验证超长非流式回复只打印字符数 + 前 80 字预览"""
        long_content = "A" * 5000
        resp_data = {
            "choices": [{"message": {"role": "assistant", "content": long_content}}],
            "usage": {},
        }
        mock_resp = FakeResponse(resp_data, status_code=200)

        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
        }
        request = _make_mock_request(req_body)

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            with patch.object(shared, "API_KEY", "test-key"):
                with patch.object(shared, "DATABASE_ENABLED", False):
                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        out = captured_out.getvalue()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("🔬 原始响应完整结构（共 5036 字符）:", out)
        # 验证预览内容截断（前 80 字加 ...）
        self.assertIn("...", out)
        # 完整 5000 字符绝不能出现在日志中
        self.assertNotIn("A" * 100, out)

    def test_memory_long_malformed_json_preview_convergence(self):
        """验证超长畸形记忆返回预览上限 300 字符，并保留错误行及上下文"""
        long_junk = "{" + ("x" * 2000) + ": invalid_json"
        resp_data = {
            "choices": [
                {
                    "message": {"content": long_junk},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 100},
        }
        mock_resp = FakeResponse(resp_data, status_code=200)

        sample_messages = [{"role": "user", "content": "你好", "_source_message_id": 1}]

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            with patch.object(memory_extractor, "MEMORY_MODEL", "mock-model"):
                with patch.object(memory_extractor, "MEMORY_API_KEY", "test-mem-key"):
                    with patch("shared.post_chat_completion", new_callable=AsyncMock, return_value=mock_resp):
                        result = asyncio.run(memory_extractor.extract_memories(sample_messages))

        out = captured_out.getvalue()
        self.assertEqual(result, [])
        self.assertIn("⚠️  记忆提取结果中未找到完整 JSON 数组", out)
        self.assertIn("📝 原始返回（共 2015 字符）:", out)
        self.assertIn("…(已截断，共 2015 字)", out)
        self.assertNotIn("x" * 350, out)

    def test_sanitize_exception_message(self):
        """验证异常脱敏函数对三种输入（带密码URL、不带密码URL、普通文本）的处理"""
        # 1. 带密码的 postgres URL
        url_with_pass = "failed to connect to postgresql://myuser:super_secret_pw@db.neon.tech:5432/main_db?sslmode=require"
        sanitized_1 = sanitize_exception_message(url_with_pass)
        self.assertEqual(sanitized_1, "failed to connect to postgresql://***@db.neon.tech:5432/main_db?sslmode=require")
        self.assertNotIn("super_secret_pw", sanitized_1)
        self.assertNotIn("myuser", sanitized_1)

        # 2. 不带密码的 postgres URL
        url_no_pass = "failed to connect to postgresql://localhost:5432/main_db"
        sanitized_2 = sanitize_exception_message(url_no_pass)
        self.assertEqual(sanitized_2, "failed to connect to postgresql://localhost:5432/main_db")

        # 3. 普通文本
        plain_text = "Connection refused by peer (errno 111)"
        sanitized_3 = sanitize_exception_message(plain_text)
        self.assertEqual(sanitized_3, "Connection refused by peer (errno 111)")


if __name__ == "__main__":
    unittest.main()
