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
        self.assertRegex(out, r"📡 req=\w{6} 请求:")
        self.assertRegex(out, r"🧩 req=\w{6} 前缀对比:")
        self.assertIn("无上一请求", out)
        self.assertRegex(out, r"🧮 req=\w{6} 缓存: cached=60 / prompt=120 \(50%\)")

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

    def test_case_7_stream_immediate_200(self):
        """用例⑦（阶段a）：流式立即200，验证📡/⏱尝试#1/📨/⏱首字节/🏁/🧮/📊均正常输出且无🔌"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "stream": True,
            "req_id": "test01",
        }
        request = _make_mock_request(req_body)

        sse_lines = [
            'data: {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}',
            'data: {"choices": [{"index": 0, "delta": {"content": "你好呀！"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "cached_tokens": 50, "completion_tokens": 10, "total_tokens": 110}}',
            'data: [DONE]',
        ]

        async def mock_client_send(req, stream=True):
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
                    with patch.object(shared, "DATABASE_ENABLED", True):
                        with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                            with patch("db.conversations.save_token_usage", new_callable=AsyncMock):
                                resp, chunks = asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例7 立即200 输出:\n{out}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("📡 req=test01 请求:", out)
        self.assertRegex(out, r"⏱ req=test01 尝试#1: 发出→响应头 \d+\.\d+s status=200")
        self.assertIn("📨 req=test01 上游响应: status=200", out)
        self.assertRegex(out, r"⏱ req=test01 首字节 \d+\.\d+s")
        self.assertIn("🏁 req=test01 finish_reason: stop", out)
        self.assertIn("🧮 req=test01 缓存: cached=50 / prompt=100 (50%)", out)
        self.assertIn("📊 req=test01 Stream Token: 100 + 10 = 110", out)
        self.assertNotIn("🔌 req=test01 客户端断开", out)

    def test_case_8_stream_429_then_200(self):
        """用例⑧（阶段b）：先429再200重试，验证尝试#1(429) -> ⏳上游429 -> 尝试#2(200) -> 响应头/首字节正常"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "测试重试"}],
            "stream": True,
            "req_id": "test02",
        }
        request = _make_mock_request(req_body)

        call_count = 0
        sse_lines = [
            'data: {"choices": [{"index": 0, "delta": {"content": "重试成功输出"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 80, "cached_tokens": 0, "completion_tokens": 5, "total_tokens": 85}}',
            'data: [DONE]',
        ]

        async def mock_client_send(req, stream=True):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockStreamResponse(429, headers={"retry-after": "0.01"})
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
                            with patch.object(shared, "is_vertex_endpoint", return_value=True):
                                with patch.object(shared, "get_vertex_access_token", return_value="mock-token-xyz"):
                                    resp, chunks = asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例8 先429再200 输出:\n{out}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(call_count, 2)
        self.assertRegex(out, r"⏱ req=test02 尝试#1: 发出→响应头 \d+\.\d+s status=429")
        self.assertRegex(out, r"⏳ req=test02 上游 429，\d+\.\d+s 后第 1 次重试")
        self.assertRegex(out, r"⏱ req=test02 尝试#2: 发出→响应头 \d+\.\d+s status=200")
        self.assertIn("📨 req=test02 上游响应: status=200", out)
        self.assertRegex(out, r"⏱ req=test02 首字节 \d+\.\d+s")
        self.assertNotIn("🔌 req=test02 客户端断开", out)

    def test_case_9_stream_hanging_client_disconnect(self):
        """用例⑨（阶段c）：上游一直不返回，客户端取消生成器，验证必须有 🔌 行且阶段=等上游响应头"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "挂起测试"}],
            "stream": True,
            "req_id": "test03",
        }
        request = _make_mock_request(req_body)

        async def mock_hanging_send(req, stream=True):
            await asyncio.sleep(100.0)

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            async def consume():
                async for _ in resp.body_iterator:
                    pass
            task = asyncio.create_task(consume())
            await asyncio.sleep(0.04)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch("httpx.AsyncClient.send", side_effect=mock_hanging_send):
                with patch.object(shared, "API_KEY", "test-key"):
                    with patch.object(shared, "DATABASE_ENABLED", False):
                        with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                            asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例9 客户端断开输出:\n{out}")
        self.assertIn("📡 req=test03 请求:", out)
        self.assertRegex(out, r"🔌 req=test03 客户端断开: 已用时 \d+\.\d+ 秒, 阶段=等上游响应头")
        self.assertNotIn("阶段=接收中", out)
        self.assertNotIn("阶段=兜底", out)

    def test_case_10_slow_warning_60s(self):
        """用例⑩：单次尝试等待超时打 60s 告警日志，但不中止请求"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "慢响应测试"}],
            "stream": True,
            "req_id": "test04",
        }
        request = _make_mock_request(req_body)

        sse_lines = [
            'data: {"choices": [{"index": 0, "delta": {"content": "迟到的正文"}}]}',
            'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
            'data: [DONE]',
        ]

        async def mock_slow_send(req, stream=True):
            await asyncio.sleep(0.06)  # 超过 0.03s 阈值
            return _MockStreamResponse(200, sse_lines=sse_lines)

        async def run_flow():
            resp = await chat_routes._chat_completions_inner(request)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)
            return resp, chunks

        buf = io.StringIO()
        with patch.dict("os.environ", {"UPSTREAM_SLOW_WARN_TIMEOUT": "0.03"}):
            with patch("sys.stdout", buf):
                with patch("httpx.AsyncClient.send", side_effect=mock_slow_send):
                    with patch.object(shared, "API_KEY", "test-key"):
                        with patch.object(shared, "DATABASE_ENABLED", False):
                            with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                                resp, chunks = asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例10 慢响应告警输出:\n{out}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("⏱ req=test04 尝试#1 已等待 60s 仍无响应", out)
        self.assertRegex(out, r"⏱ req=test04 尝试#1: 发出→响应头 \d+\.\d+s status=200")
        self.assertRegex(out, r"⏱ req=test04 首字节 \d+\.\d+s")
        self.assertNotIn("🔌 req=test04 客户端断开", out)

    def test_case_11_disconnect_during_receiving(self):
        """用例⑪：接收中阶段客户端断开，验证阶段=接收中"""
        req_body = {
            "model": "google/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "接收中断开测试"}],
            "stream": True,
            "req_id": "test05",
        }
        request = _make_mock_request(req_body)

        async def run_flow():
            sse_lines = [
                'data: {"choices": [{"index": 0, "delta": {"content": "第一块"}}]}',
                'data: {"choices": [{"index": 0, "delta": {"content": "第二块"}}]}',
            ]
            resp_mock = _MockStreamResponse(200, sse_lines=sse_lines, chunk_delay=0.08)

            async def mock_send(req, stream=True):
                return resp_mock

            with patch("httpx.AsyncClient.send", side_effect=mock_send):
                resp = await chat_routes._chat_completions_inner(request)
                iterator = resp.body_iterator
                # 读出首字节后客户端断开连接 (aclose)
                async for chunk in iterator:
                    break
                await iterator.aclose()

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            with patch.object(shared, "API_KEY", "test-key"):
                with patch.object(shared, "DATABASE_ENABLED", False):
                    with patch.object(shared, "CACHE_PARTITION_ENABLED", False):
                        asyncio.run(run_flow())

        out = buf.getvalue()
        print(f"用例11 接收中断开输出:\n{out}")
        self.assertIn("📡 req=test05 请求:", out)
        self.assertRegex(out, r"⏱ req=test05 首字节 \d+\.\d+s")
        self.assertRegex(out, r"🔌 req=test05 客户端断开: 已用时 \d+\.\d+ 秒, 阶段=接收中")


def _make_mock_request(body_dict, headers_dict=None):
    hdrs = [(b"content-type", b"application/json")]
    if headers_dict:
        for k, v in headers_dict.items():
            hdrs.append((k.lower().encode("utf-8"), v.encode("utf-8")))

    async def mock_receive():
        return {
            "type": "http.request",
            "body": json.dumps(body_dict).encode("utf-8"),
            "more_body": False,
        }

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


if __name__ == "__main__":
    unittest.main()
