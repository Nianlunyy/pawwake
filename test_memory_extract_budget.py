"""
test_memory_extract_budget.py: 验证记忆提取超时配置、耗时统计与异常格式
"""

import asyncio
import contextlib
import importlib
import io
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

import httpx

import memory_extractor
import shared


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def json(self):
        return self._data


class TestMemoryExtractBudget(unittest.TestCase):
    def setUp(self):
        self.original_model = memory_extractor.MEMORY_MODEL
        self.original_key = memory_extractor.MEMORY_API_KEY
        memory_extractor.MEMORY_MODEL = "test-memory-model"
        memory_extractor.MEMORY_API_KEY = "test-memory-key"
        self.sample_messages = [
            {"role": "user", "content": "我喜欢喝红茶", "_source_message_id": 101},
            {"role": "assistant", "content": "记下了，红茶很好喝", "_source_message_id": 102},
        ]

    def tearDown(self):
        memory_extractor.MEMORY_MODEL = self.original_model
        memory_extractor.MEMORY_API_KEY = self.original_key

    def test_timeout_exception_handling(self):
        """1. mock shared.post_chat_completion 抛 httpx.ReadTimeout("") -> extract_memories 返回 []，日志含 ReadTimeout 和超时秒数"""
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            with patch("shared.post_chat_completion", side_effect=httpx.ReadTimeout("")):
                result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))

        output = f.getvalue()
        self.assertEqual(result, [])
        self.assertIn("ReadTimeout", output)
        self.assertIn(str(int(memory_extractor.MEMORY_TIMEOUT)), output)
        self.assertIn("记忆提取超时", output)
        print(f"✅ 用例1(超时捕获): 成功拦截 ReadTimeout，返回空列表，日志含类名与超时秒数:\n   {output.strip()}")

    def test_generic_exception_empty_str(self):
        """2. 通用异常且 str(e) 为空 -> extract_memories 返回 []，日志含类名和 repr"""
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            with patch("shared.post_chat_completion", side_effect=RuntimeError("")):
                result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))

        output = f.getvalue()
        self.assertEqual(result, [])
        self.assertIn("RuntimeError", output)
        self.assertIn("RuntimeError('')", output)
        self.assertIn("记忆提取出错", output)
        print(f"✅ 用例2(空str异常): 成功记录类名与 repr:\n   {output.strip()}")

    def test_reload_memory_timeout_env(self):
        """3. 设置环境变量 MEMORY_TIMEOUT=7 后重新加载模块，确认 httpx.AsyncClient 收到的 timeout 为 7"""
        old_env = os.environ.get("MEMORY_TIMEOUT")
        try:
            os.environ["MEMORY_TIMEOUT"] = "7"
            importlib.reload(memory_extractor)
            self.assertEqual(memory_extractor.MEMORY_TIMEOUT, 7.0)

            captured_timeouts = []
            real_async_client = httpx.AsyncClient

            def mock_client_init(*args, **kwargs):
                if "timeout" in kwargs:
                    captured_timeouts.append(kwargs["timeout"])
                elif len(args) > 1:
                    captured_timeouts.append(args[1])
                return real_async_client(*args, **kwargs)

            memory_extractor.MEMORY_MODEL = "test-memory-model"
            memory_extractor.MEMORY_API_KEY = "test-memory-key"

            # mock post_chat_completion 直接抛异常中断，只要 AsyncClient 实例化成功并捕获参数即可
            with patch("httpx.AsyncClient", side_effect=mock_client_init):
                with patch("shared.post_chat_completion", side_effect=httpx.ReadTimeout("")):
                    asyncio.run(memory_extractor.extract_memories(self.sample_messages))

            self.assertIn(7.0, captured_timeouts)
            print(f"✅ 用例3(超时环境变量重载): MEMORY_TIMEOUT=7 生效，AsyncClient 收到 timeout={captured_timeouts}")
        finally:
            if old_env is None:
                os.environ.pop("MEMORY_TIMEOUT", None)
            else:
                os.environ["MEMORY_TIMEOUT"] = old_env
            importlib.reload(memory_extractor)
            memory_extractor.MEMORY_MODEL = self.original_model
            memory_extractor.MEMORY_API_KEY = self.original_key

    def test_normal_response_and_timing_log(self):
        """4. 正常返回的 mock -> ⏱ 行出现，解析结果不变"""
        resp_data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "content": "用户喜欢喝红茶",
                                "importance": 5,
                                "action": "new",
                                "source_refs": [1],
                            }
                        ])
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 42},
        }
        mock_response = FakeResponse(resp_data, status_code=200)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            with patch("shared.post_chat_completion", new_callable=AsyncMock, return_value=mock_response):
                result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))

        output = f.getvalue()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "用户喜欢喝红茶")
        self.assertEqual(result[0]["source_message_ids"], [101])
        self.assertIn("⏱ 提取耗时", output)
        self.assertIn("秒", output)
        print(f"✅ 用例4(正常提取与计时): ⏱ 计时日志正常打印，解析结果正确: {result}")


if __name__ == "__main__":
    unittest.main()
