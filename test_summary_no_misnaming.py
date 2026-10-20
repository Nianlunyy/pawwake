"""Unit test for summary prompt rules avoiding misnaming (fix/summary-no-misnaming).

Verifies:
a) generate_summary 与 fold_summary_parts 构造出的实际提示词均含新增噪声过滤规则：
   "称呼说错、认错人，以及由此产生的道歉与澄清，属于噪声，整件事不写入摘要。"
b) generate_summary 构造出的实际提示词含正向封闭的人称规则：
   "摘要中指代她只写\"阿狸\"或\"狸宝\"，自称一律写\"我\"。"
c) 提示词全文严格不含特定误称与口误事件词。
"""

import asyncio
import unittest
from unittest.mock import patch

import shared
import partition_engine


def _capture_generate_summary_prompt() -> str:
    msgs = [
        {"role": "user", "content": "阿狸，今天过得开心吗？", "created_at": "2026-10-20T10:00:00Z"},
        {"role": "assistant", "content": "很开心呀，阿澈！", "created_at": "2026-10-20T10:01:00Z"},
    ]
    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "【状态与约定】\n模拟生成的摘要正文"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-summary-model"):
            res = asyncio.run(partition_engine.generate_summary(msgs, "sess_test"))

    if not captured_prompt:
        raise RuntimeError("generate_summary 未成功构造 prompt")
    return captured_prompt


def _capture_fold_summary_prompt() -> str:
    old_parts = [
        "【记录时段：2026-10-18 10:00 至 2026-10-18 12:00（UTC+8）】\n旧摘要段落一" * 10,
        "【记录时段：2026-10-18 14:00 至 2026-10-18 16:00（UTC+8）】\n旧摘要段落二" * 10,
    ]
    recent_parts = [
        "【记录时段：2026-10-19 10:00 至 2026-10-19 12:00（UTC+8）】\n近期保护段落一",
        "【记录时段：2026-10-19 14:00 至 2026-10-19 16:00（UTC+8）】\n近期保护段落二",
    ]
    captured_prompt = None

    async def mock_call(prompt, label="摘要折叠"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "合并后的基础摘要正文"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-summary-model"):
            with patch.object(shared, "CACHE_SUMMARY_BUDGET_CHARS", 50):
                res = asyncio.run(
                    partition_engine.fold_summary_parts(old_parts + recent_parts, protected_tail=0)
                )

    if not captured_prompt:
        raise RuntimeError("fold_summary_parts 未成功构造 prompt")
    return captured_prompt


class TestSummaryNoMisnaming(unittest.TestCase):
    def test_summary_and_fold_prompts_no_misnaming(self):
        """调用生产函数构造出实际使用的摘要提示词与折叠提示词，断言 a、b、c。"""
        gen_prompt = _capture_generate_summary_prompt()
        fold_prompt = _capture_fold_summary_prompt()

        rule_noise = "称呼说错、认错人，以及由此产生的道歉与澄清，属于噪声，整件事不写入摘要。"
        rule_person = '摘要中指代她只写"阿狸"或"狸宝"，自称一律写"我"。'

        # 断言 a（摘要）: generate_summary 提示词含新增噪声过滤规则
        with self.subTest("断言 a（摘要）: generate_summary 含新增噪声过滤规则"):
            self.assertIn(
                rule_noise,
                gen_prompt,
                "断言 a（摘要）失败: generate_summary 提示词未包含新增的噪声过滤规则",
            )

        # 断言 a（折叠）: fold_summary_parts 提示词含新增噪声过滤规则
        with self.subTest("断言 a（折叠）: fold_summary_parts 含新增噪声过滤规则"):
            self.assertIn(
                rule_noise,
                fold_prompt,
                "断言 a（折叠）失败: fold_summary_parts 提示词未包含新增的噪声过滤规则",
            )

        # 断言 b: 含正向封闭的人称规则
        with self.subTest("断言 b: 含正向封闭的人称规则"):
            self.assertIn(
                rule_person,
                gen_prompt,
                "断言 b 失败: generate_summary 提示词未包含正向封闭的人称映射规则",
            )

        # 断言 c: 整段提示词不含特定误称与口误事件词
        with self.subTest("断言 c: 整段提示词不含特定误称与口误词"):
            forbidden_words = ["\u9648\u5de5", "\u4e4c\u9f99"]
            for word in forbidden_words:
                self.assertNotIn(
                    word,
                    gen_prompt,
                    f"断言 c 失败: generate_summary 提示词违规包含禁词 '{word}'",
                )
                self.assertNotIn(
                    word,
                    fold_prompt,
                    f"断言 c 失败: fold_summary_parts 提示词违规包含禁词 '{word}'",
                )


if __name__ == "__main__":
    unittest.main()
