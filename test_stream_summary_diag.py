"""
test_stream_summary_diag.py:
验证 routes/chat.py:
1. _streaming_response 响应头包含 X-Accel-Buffering: no 与 Cache-Control: no-cache, no-transform
2. _stream_and_capture 的 finally 在正常完成、客户端断开(aclose / GeneratorExit)、上游异常时都能 100% 打印汇总日志
"""

import asyncio
import io
import json
import unittest
from unittest.mock import patch

import httpx

import routes.chat as chat_routes
import shared


class _FakeStreamResponse:
    def __init__(self, chunks, status_code=200, error_after=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = {"content-type": "text/event-stream"}
        self._error_after = error_after
        self.closed = False

    async def aiter_bytes(self):
        for idx, chunk in enumerate(self._chunks):
            if self._error_after is not None and idx >= self._error_after:
                raise httpx.ReadTimeout("simulated upstream read timeout")
            yield chunk
            await asyncio.sleep(0)

    async def aclose(self):
        self.closed = True


class TestStreamSummaryDiag(unittest.TestCase):
    def test_streaming_response_anti_buffering_headers(self):
        async def _empty():
            if False:
                yield b""

        resp = chat_routes._streaming_response(_empty(), [])
        self.assertEqual(resp.headers.get("cache-control"), "no-cache, no-transform")
        self.assertEqual(resp.headers.get("x-accel-buffering"), "no")

    def test_summary_logged_on_normal_completion(self):
        chunks = [
            b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        fake_resp = _FakeStreamResponse(chunks)

        async def _fake_retry(*args, **kwargs):
            return fake_resp, 0

        async def _run():
            gen = chat_routes.stream_and_capture(
                headers={},
                body={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                session_id="s1",
                user_message="hi",
                model="test-model",
                skip_conversation_log=True,
                req_id="norm01",
            )
            received = []
            async for item in gen:
                received.append(item)
            return b"".join(received)

        buf = io.StringIO()
        with patch.object(shared, "is_vertex_endpoint", return_value=False), \
             patch.object(shared, "conversation_persistence_enabled", return_value=False), \
             patch.object(chat_routes, "_send_stream_with_429_retry", side_effect=_fake_retry), \
             patch("sys.stdout", buf):
            total_out = asyncio.run(_run())

        out = buf.getvalue()
        self.assertIn("📦 req=norm01 下游发送汇总:", out)
        self.assertIn("yielded_done=True", out)
        self.assertIn("exit=normal", out)
        self.assertIn(f"bytes={len(total_out)}", out)

    def test_summary_logged_on_client_disconnect(self):
        chunks = [
            b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"body"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        fake_resp = _FakeStreamResponse(chunks)

        async def _fake_retry(*args, **kwargs):
            return fake_resp, 0

        async def _run():
            gen = chat_routes.stream_and_capture(
                headers={},
                body={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                session_id="s2",
                user_message="hi",
                model="test-model",
                skip_conversation_log=True,
                req_id="disc02",
            )
            first = await gen.__anext__()
            await gen.aclose()
            return first

        buf = io.StringIO()
        with patch.object(shared, "is_vertex_endpoint", return_value=False), \
             patch.object(shared, "conversation_persistence_enabled", return_value=False), \
             patch.object(chat_routes, "_send_stream_with_429_retry", side_effect=_fake_retry), \
             patch("sys.stdout", buf):
            asyncio.run(_run())

        out = buf.getvalue()
        self.assertIn("📦 req=disc02 下游发送汇总:", out)
        self.assertIn("yielded_done=False", out)
        self.assertIn("exit=GeneratorExit", out)
        self.assertIn("🔌 req=disc02 客户端断开", out)

    def test_summary_logged_on_upstream_error(self):
        chunks = [
            b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"body"}}]}\n\n',
        ]
        fake_resp = _FakeStreamResponse(chunks, error_after=1)

        async def _fake_retry(*args, **kwargs):
            return fake_resp, 0

        async def _run():
            gen = chat_routes.stream_and_capture(
                headers={},
                body={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                session_id="s3",
                user_message="hi",
                model="test-model",
                skip_conversation_log=True,
                req_id="uerr03",
            )
            async for _ in gen:
                pass

        buf = io.StringIO()
        with patch.object(shared, "is_vertex_endpoint", return_value=False), \
             patch.object(shared, "conversation_persistence_enabled", return_value=False), \
             patch.object(chat_routes, "_send_stream_with_429_retry", side_effect=_fake_retry), \
             patch("sys.stdout", buf):
            with self.assertRaises(httpx.ReadTimeout):
                asyncio.run(_run())

        out = buf.getvalue()
        self.assertIn("📦 req=uerr03 下游发送汇总:", out)
        self.assertIn("yielded_done=False", out)
        self.assertIn("exit=ReadTimeout", out)
        self.assertIn("🔥 req=uerr03 网关异常终止: type=ReadTimeout", out)


if __name__ == "__main__":
    unittest.main()
