"""
test_memory_thinking_effort.py: 验证记忆提取推理档 (MEMORY_REASONING_EFFORT) 请求体透传与截断诊断
"""

import asyncio
import io
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import memory_extractor
import shared


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data)

    def json(self):
        return self._data


class TestMemoryThinkingEffort(unittest.TestCase):
    def setUp(self):
        self.orig_model = memory_extractor.MEMORY_MODEL
        self.orig_key = memory_extractor.MEMORY_API_KEY
        self.orig_effort = memory_extractor.MEMORY_REASONING_EFFORT
        memory_extractor.MEMORY_MODEL = "test-memory-model"
        memory_extractor.MEMORY_API_KEY = "test-memory-key"
        self.sample_messages = [
            {"role": "user", "content": "我最喜欢喝大吉岭红茶", "_source_message_id": 101},
            {"role": "assistant", "content": "收到，记下你喜欢大吉岭红茶", "_source_message_id": 102},
        ]

    def tearDown(self):
        memory_extractor.MEMORY_MODEL = self.orig_model
        memory_extractor.MEMORY_API_KEY = self.orig_key
        memory_extractor.MEMORY_REASONING_EFFORT = self.orig_effort

    def test_case_a_empty_env_omitted_verbatim(self):
        """a. 环境变量为空 -> 请求体不含 reasoning_effort，且与改动前逐字相同"""
        memory_extractor.MEMORY_REASONING_EFFORT = ""
        captured_payloads = []

        async def mock_post(client, url, key, payload):
            captured_payloads.append(payload)
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps([
                                {
                                    "content": "用户喜欢大吉岭红茶",
                                    "importance": 5,
                                    "action": "new",
                                    "source_refs": [1],
                                }
                            ])
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 50},
            }
            return FakeResponse(resp_data, status_code=200)

        with patch("shared.post_chat_completion", side_effect=mock_post), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))

        out = mock_stdout.getvalue()
        self.assertEqual(len(captured_payloads), 1)
        sent_body = captured_payloads[0]

        # 逐字结构核对：必须仅有 model, max_tokens, messages 三个键
        self.assertEqual(list(sent_body.keys()), ["model", "max_tokens", "messages"])
        self.assertNotIn("reasoning_effort", sent_body)
        self.assertNotIn("thinking_config", sent_body)
        self.assertEqual(sent_body["model"], "test-memory-model")
        self.assertEqual(sent_body["max_tokens"], memory_extractor.MEMORY_MAX_TOKENS)

        # 检查日志输出
        self.assertIn("🧠 记忆提取推理档: 未设置", out)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "用户喜欢大吉岭红茶")
        print(f"✅ 用例a(推理档为空): 请求体键集合与改动前完全一致: {list(sent_body.keys())}，日志确认打出 '未设置'")

    def test_case_b_reasoning_effort_low(self):
        """b. MEMORY_REASONING_EFFORT=low -> 请求体 reasoning_effort == 'low'，且不含 thinking_config"""
        memory_extractor.MEMORY_REASONING_EFFORT = "low"
        captured_payloads = []

        async def mock_post(client, url, key, payload):
            captured_payloads.append(payload)
            resp_data = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps([
                                {
                                    "content": "用户喜欢大吉岭红茶",
                                    "importance": 5,
                                    "action": "new",
                                    "source_refs": [1],
                                }
                            ])
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 50},
            }
            return FakeResponse(resp_data, status_code=200)

        with patch("shared.post_chat_completion", side_effect=mock_post), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))

        out = mock_stdout.getvalue()
        self.assertEqual(len(captured_payloads), 1)
        sent_body = captured_payloads[0]

        self.assertEqual(sent_body.get("reasoning_effort"), "low")
        self.assertNotIn("thinking_config", sent_body)
        self.assertIn("🧠 记忆提取推理档: low", out)
        self.assertEqual(len(result), 1)
        print(f"✅ 用例b(reasoning_effort=low): 请求体成功注入 reasoning_effort=low 且无 thinking_config")

    def test_case_c_truncation_after_source_refs_bracket(self):
        """c. 返回 finish_reason=length 且 JSON 在 source_refs 的 ']' 后截断 -> 返回 []，日志含截断提示"""
        memory_extractor.MEMORY_REASONING_EFFORT = "low"

        # 模拟第 834 轮截断场景：在 source_refs 的 ] 之后截断，缺少 } 和数组结尾 ]
        truncated_json = (
            "[\n"
            '  {\n'
            '    "content": "阿狸最喜欢喝大吉岭红茶",\n'
            '    "importance": 8,\n'
            '    "action": "new",\n'
            '    "candidate_id": null,\n'
            '    "remind_at": null,\n'
            '    "source_refs": [1]'
        )

        resp_data = {
            "choices": [
                {
                    "message": {"content": truncated_json},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "completion_tokens": 1308,
                "completion_tokens_details": {"reasoning_tokens": 6688},
            },
        }

        async def mock_post(client, url, key, payload):
            return FakeResponse(resp_data, status_code=200)

        with patch("shared.post_chat_completion", side_effect=mock_post), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))

        out = mock_stdout.getvalue()

        # 预期：返回空列表，并且输出中包含截断关键警告
        self.assertEqual(result, [])
        self.assertIn("⚠️  记忆提取结果解析失败", out)
        self.assertIn(
            "⚠️ 本次输出被 max_tokens 截断（finish_reason=length），解析失败多半源于截断，推理 token 占用见上方原始返回行",
            out,
        )
        self.assertIn("完整原始返回", out)
        print(f"✅ 用例c(source_refs截断诊断): 成功拦截截断异常，返回 [] 并打印截断诊断告警")

    def test_case_d_normal_response_with_various_efforts(self):
        """d. 正常返回仍能解析；测试不同档位透传"""
        test_efforts = ["medium", "high", "none", "minimal"]
        for effort in test_efforts:
            memory_extractor.MEMORY_REASONING_EFFORT = effort
            captured_payloads = []

            async def mock_post(client, url, key, payload):
                captured_payloads.append(payload)
                resp_data = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps([
                                    {
                                        "content": f"阿狸在档位{effort}下的记忆",
                                        "importance": 7,
                                        "action": "new",
                                        "source_refs": [1],
                                    }
                                ])
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 60},
                }
                return FakeResponse(resp_data, status_code=200)

            with patch("shared.post_chat_completion", side_effect=mock_post), \
                 patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))

            out = mock_stdout.getvalue()
            self.assertEqual(len(captured_payloads), 1)
            self.assertEqual(captured_payloads[0].get("reasoning_effort"), effort)
            self.assertIn(f"🧠 记忆提取推理档: {effort}", out)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["content"], f"阿狸在档位{effort}下的记忆")
            self.assertEqual(result[0]["importance"], 7)

        print(f"✅ 用例d(正常解析与多档位支持): 正常返回成功解析，多档位透传校验通过")

    def test_case_e_response_json_decode_error(self):
        """e. mock response.json() 抛 JSONDecodeError -> extract_memories 返回 []，打印响应体而非输入消息"""
        sentinel = "测试输入哨兵句_唯一特征标识_98765"
        sample_messages = [
            {"role": "user", "content": sentinel, "_source_message_id": 101},
        ]

        class BadJsonResponse:
            status_code = 200
            text = "<html>502 Bad Gateway</html>"

            def json(self):
                raise json.JSONDecodeError("Expecting value", self.text, 0)

        async def mock_post(client, url, key, payload):
            return BadJsonResponse()

        with patch("shared.post_chat_completion", side_effect=mock_post), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            try:
                result = asyncio.run(memory_extractor.extract_memories(sample_messages))
            except Exception as err:
                self.fail(f"触发了非预期异常: {err}")

        out = mock_stdout.getvalue()
        self.assertEqual(result, [])
        self.assertIn("⚠️  记忆提取结果解析失败", out)
        self.assertIn("<html>502", out)
        self.assertIn("响应体前 2000 字符", out)
        self.assertNotIn(sentinel, out)
        self.assertNotIn("本次输出被 max_tokens 截断", out)
        print(f"✅ 用例e(response.json异常拦截): 成功截获 502 HTML 响应体，日志含 <html>502 且排除哨兵输入句")

    def test_case_f_response_none_json_decode_error(self):
        """f. mock post_chat_completion 在 response 赋值前抛 JSONDecodeError -> response 为 None 时不抛异常，打印响应体不可用"""
        async def mock_post_raise(*args, **kwargs):
            raise json.JSONDecodeError("mock decode error before response assignment", "", 0)

        with patch("shared.post_chat_completion", side_effect=mock_post_raise), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            try:
                result = asyncio.run(memory_extractor.extract_memories(self.sample_messages))
            except Exception as err:
                self.fail(f"response 为 None 时抛出了未预期异常: {err}")

        out = mock_stdout.getvalue()
        self.assertEqual(result, [])
        self.assertIn("响应体不可用", out)
        self.assertNotIn("本次输出被 max_tokens 截断", out)
        print(f"✅ 用例f(response为None异常拦截): response 为 None 时平稳捕获，打印 '响应体不可用' 且返回 []")


if __name__ == "__main__":
    unittest.main()

