"""
test_cache_diag.py: 验证前缀对比诊断日志与缓存 Usage 诊断日志
包含用例：
① 连续两次请求仅末尾新增 → 共同前缀应等于上一次全部
② 改动中间一条 → 首个差异 idx 正确
③ tools 变化 → tools同=N，tools 不变 → tools同=Y
④ 异常输入不抛出、请求照常发出
⑤ Usage 各种结构与缺省字段测试
⑥ 端到端调用生产函数 _chat_completions_inner，确认诊断日志正常打出且异常被安全隔离
"""

import asyncio
import io
import json
import re
import sys
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import Request
from starlette.datastructures import Headers

import routes.chat as chat_routes
import shared


class TestCacheDiag(unittest.TestCase):
    def setUp(self):
        chat_routes._LAST_REQUEST_CACHE.clear()

    def tearDown(self):
        chat_routes._LAST_REQUEST_CACHE.clear()

    def test_case_1_continuous_requests_append_tail(self):
        """用例①：连续两次请求仅末尾新增 → 共同前缀应等于上一次全部"""
        sid = "test-session-1"
        body_1 = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "你是一个贴心助手"},
                {"role": "user", "content": "你好，今天天气怎么样？"},
                {"role": "assistant", "content": "今天天气晴朗，气温25摄氏度。"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }

        # 捕获第一次请求输出
        buf_1 = io.StringIO()
        with patch("sys.stdout", buf_1):
            res_1 = chat_routes.diagnose_cache_prefix(sid, body_1)

        out_1 = buf_1.getvalue().strip()
        print(f"用例1 请求1 日志: {out_1}")
        self.assertTrue(res_1["is_first"])
        self.assertEqual(res_1["round"], 1)
        self.assertIn("无上一请求", out_1)
        self.assertIn("round=1", out_1)

        # 第二次请求：仅在末尾新增一条 user 消息
        body_2 = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "你是一个贴心助手"},
                {"role": "user", "content": "你好，今天天气怎么样？"},
                {"role": "assistant", "content": "今天天气晴朗，气温25摄氏度。"},
                {"role": "user", "content": "那明天会下雨吗？"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }

        buf_2 = io.StringIO()
        with patch("sys.stdout", buf_2):
            res_2 = chat_routes.diagnose_cache_prefix(sid, body_2)

        out_2 = buf_2.getvalue().strip()
        print(f"用例1 请求2 日志: {out_2}")

        self.assertFalse(res_2["is_first"])
        self.assertEqual(res_2["round"], 2)
        # 共同前缀应等于上一次全部 (3 条)
        self.assertEqual(res_2["common_prefix_count"], len(body_1["messages"]))
        self.assertEqual(res_2["common_prefix_count"], 3)
        self.assertEqual(res_2["total_messages"], 4)
        self.assertTrue(res_2["tools_same"])
        # 首个差异在 index 3
        self.assertEqual(res_2["diff_idx"], 3)
        self.assertEqual(res_2["diff_role"], "user")
        self.assertEqual(res_2["diff_snippet"], "那明天会下雨吗？")

        # 检查日志格式
        self.assertIn("round=2", out_2)
        self.assertIn("tools同=Y", out_2)
        self.assertIn("共同前缀 3/4 条", out_2)
        self.assertIn("首个差异 idx=3, role=user, 片段=那明天会下雨吗？", out_2)

    def test_case_2_modify_middle_message(self):
        """用例②：改动中间一条 → 首个差异 idx 正确"""
        sid = "test-session-2"
        body_1 = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "系统人设"},
                {"role": "user", "content": "原始提问一"},
                {"role": "assistant", "content": "原始回复一"},
                {"role": "user", "content": "后续提问二"},
            ],
        }

        with patch("sys.stdout", io.StringIO()):
            chat_routes.diagnose_cache_prefix(sid, body_1)

        # 改动 index 1 (中间一条)
        body_2 = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "系统人设"},
                {"role": "user", "content": "修改后的提问一（中间被改动）"},
                {"role": "assistant", "content": "原始回复一"},
                {"role": "user", "content": "后续提问二"},
            ],
        }

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            res_2 = chat_routes.diagnose_cache_prefix(sid, body_2)

        out = buf.getvalue().strip()
        print(f"用例2 改动中间 日志: {out}")

        self.assertEqual(res_2["common_prefix_count"], 1)
        self.assertEqual(res_2["diff_idx"], 1)
        self.assertEqual(res_2["diff_role"], "user")
        self.assertEqual(res_2["diff_snippet"], "修改后的提问一（中间被改动）")
        self.assertIn("共同前缀 1/4 条", out)
        self.assertIn("首个差异 idx=1, role=user, 片段=修改后的提问一（中间被改动）", out)

    def test_case_3_tools_change(self):
        """用例③：tools 变化 → tools同=N；tools 不变 → tools同=Y"""
        sid = "test-session-3"
        body_1 = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "查股票"}],
            "tools": [{"type": "function", "function": {"name": "query_stock"}}],
        }

        with patch("sys.stdout", io.StringIO()):
            chat_routes.diagnose_cache_prefix(sid, body_1)

        # tools 改变
        body_2 = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "查股票"},
                {"role": "assistant", "content": "请输入股票代码"},
            ],
            "tools": [{"type": "function", "function": {"name": "query_crypto"}}],
        }

        buf_2 = io.StringIO()
        with patch("sys.stdout", buf_2):
            res_2 = chat_routes.diagnose_cache_prefix(sid, body_2)

        out_2 = buf_2.getvalue().strip()
        print(f"用例3 tools变化 日志: {out_2}")
        self.assertFalse(res_2["tools_same"])
        self.assertIn("tools同=N", out_2)

        # tools 保持不变
        body_3 = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "查股票"},
                {"role": "assistant", "content": "请输入股票代码"},
                {"role": "user", "content": "BTC"},
            ],
            "tools": [{"type": "function", "function": {"name": "query_crypto"}}],
        }

        buf_3 = io.StringIO()
        with patch("sys.stdout", buf_3):
            res_3 = chat_routes.diagnose_cache_prefix(sid, body_3)

        out_3 = buf_3.getvalue().strip()
        print(f"用例3 tools不变 日志: {out_3}")
        self.assertTrue(res_3["tools_same"])
        self.assertIn("tools同=Y", out_3)

    def test_case_4_abnormal_input_tolerance(self):
        """用例④：异常输入不抛出、诊断日志安全降级"""
        # 测试各类畸形入参，绝不抛出异常
        bad_inputs = [
            (None, None),
            ("sid", None),
            ("sid", {}),
            ("sid", {"messages": "not a list"}),
            ("sid", {"messages": [None, 123, {"bad": True}]}),
            ("sid", {"messages": [], "tools": object()}),
            ("sid", {"messages": [{"role": "user", "content": [{"bad_type": 1}]}]}),
        ]

        for sid, body in bad_inputs:
            try:
                res = chat_routes.diagnose_cache_prefix(sid, body)
                self.assertIsInstance(res, dict)
            except Exception as e:
                self.fail(f"diagnose_cache_prefix 抛出了异常: {e}，输入: {body}")

        # Usage 日志异常输入容错
        bad_usages = [
            None,
            "not a dict",
            {},
            {"prompt_tokens": None},
            {"prompt_tokens": "abc"},
            {"prompt_tokens": -10},
            {"cached_tokens": object()},
        ]

        for u in bad_usages:
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                try:
                    chat_routes.log_cache_usage(u)
                    out = buf.getvalue().strip()
                    self.assertIn("🧮 缓存: cached=", out)
                except Exception as e:
                    self.fail(f"log_cache_usage 抛出了异常: {e}，输入: {u}")

    def test_case_5_usage_formats(self):
        """用例⑤：验证 usage 缓存字段各服务商格式与缺省按 0 逻辑"""
        test_cases = [
            # OpenAI / Vertex 嵌套 prompt_tokens_details
            ({"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 800}}, 800, 1000, "80%"),
            # OpenRouter / Claude cache_read_input_tokens
            ({"prompt_tokens": 500, "cache_read_input_tokens": 250}, 250, 500, "50%"),
            # 直接 cached_tokens
            ({"prompt_tokens": 300, "cached_tokens": 100}, 100, 300, "33.3%"),
            # 无任何缓存字段，缺省按 0
            ({"prompt_tokens": 200}, 0, 200, "0%"),
            # prompt_tokens 为 0 / 缺省
            ({}, 0, 0, "0%"),
        ]

        for u, expected_c, expected_p, expected_pct in test_cases:
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                chat_routes.log_cache_usage(u)
            out = buf.getvalue().strip()
            print(f"Usage 测试: 输入={u} -> 输出={out}")
            expected_line = f"🧮 缓存: cached={expected_c} / prompt={expected_p} ({expected_pct})"
            self.assertEqual(out, expected_line)

    def test_case_6_production_chat_completion_integration(self):
        """用例⑥：端到端调用生产函数 _chat_completions_inner，验证日志正常打出且异常被安全拦截"""
        req_body = {
            "model": "test-chat-model",
            "messages": [
                {"role": "user", "content": "你好，这是一条生产测试消息"}
            ],
            "stream": False,
        }

        # 构造 FastAPI Request 对象
        async def mock_receive():
            return {
                "type": "http.request",
                "body": json.dumps(req_body).encode("utf-8"),
                "more_body": False,
            }

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
        }
        request = Request(scope, receive=mock_receive)

        fake_resp = {
            "choices": [{"message": {"role": "assistant", "content": "你好！我是AI助手。"}}],
            "usage": {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 60}},
        }

        class MockHttpxResponse:
            def __init__(self):
                self.status_code = 200
                self.text = json.dumps(fake_resp)

            def json(self):
                return fake_resp

        async def mock_client_post(*args, **kwargs):
            return MockHttpxResponse()

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch("httpx.AsyncClient.post", side_effect=mock_client_post):
                with patch.object(shared, "API_KEY", "test-key"):
                    with patch.object(shared, "DATABASE_ENABLED", False):
                        with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                            resp = asyncio.run(chat_routes._chat_completions_inner(request))

        out = buf.getvalue()
        print(f"端到端生产函数调用输出:\n{out}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("🧩 前缀对比:", out)
        self.assertIn("无上一请求", out)
        self.assertIn("🧮 缓存: cached=60 / prompt=120 (50%)", out)

        # 验证即使诊断逻辑内部抛异常，主请求绝对不受影响
        async def mock_receive_2():
            return {
                "type": "http.request",
                "body": json.dumps(req_body).encode("utf-8"),
                "more_body": False,
            }
        req_2 = Request(scope, receive=mock_receive_2)

        with patch("sys.stdout", io.StringIO()):
            with patch("routes.chat.diagnose_cache_prefix", side_effect=RuntimeError("模拟前缀诊断爆炸")):
                with patch("routes.chat.log_cache_usage", side_effect=RuntimeError("模拟缓存用量诊断爆炸")):
                    with patch("httpx.AsyncClient.post", side_effect=mock_client_post):
                        with patch.object(shared, "API_KEY", "test-key"):
                            with patch.object(shared, "DATABASE_ENABLED", False):
                                with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                    resp_2 = asyncio.run(chat_routes._chat_completions_inner(req_2))

        # 请求仍顺利返回 200，绝不因日志崩溃
        self.assertEqual(resp_2.status_code, 200)
        print("✅ 诊断异常拦截验证通过：内部爆炸时生产请求照常返回 200")


if __name__ == "__main__":
    unittest.main()
