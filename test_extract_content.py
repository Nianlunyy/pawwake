"""
test_extract_content.py: 验证记忆提取输入的纯文本转换与长度保护
"""

import difflib
import unittest
from memory_extractor import (
    _content_to_text,
    _count_image_blocks,
    _format_conversation,
    EXTRACT_MAX_CHARS_PER_MESSAGE,
    EXTRACT_MAX_CHARS_TOTAL,
)


class TestExtractContent(unittest.TestCase):
    def test_regression_pure_text(self):
        """回归：纯文字消息样本，改动前后 conversation_text 逐字相同"""
        sample_messages = [
            {"role": "user", "content": "你好，我是阿狸。", "_source_message_id": 101},
            {"role": "assistant", "content": "你好阿狸！很高兴见到你。", "_source_message_id": 102},
            {"role": "user", "content": "我最喜欢的食物是草莓蛋糕。\n今天天气也很好。", "_source_message_id": 103},
            {"role": "assistant", "content": "记下了，你喜欢草莓蛋糕！", "_source_message_id": 104},
            {"role": "user", "content": "再加一条无证据ID的消息"},
            {"role": "assistant", "content": "收到无证据回复。"},
            {"role": "user", "content": "复用证据ID的消息", "_source_message_id": 103},
        ]

        # 改动前的旧逻辑
        old_conversation_text = ""
        old_source_ref_by_id = {}
        for msg in sample_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            source_id = msg.get("_source_message_id")
            if isinstance(source_id, int) and not isinstance(source_id, bool):
                source_ref = old_source_ref_by_id.setdefault(source_id, len(old_source_ref_by_id) + 1)
                source_prefix = f"[证据#{source_ref}] "
            else:
                source_prefix = "[证据不可用] "
            if role == "user":
                old_conversation_text += f"{source_prefix}阿狸: {content}\n"
            elif role == "assistant":
                old_conversation_text += f"{source_prefix}阿澈: {content}\n"

        # 改动后的新逻辑
        processed_messages = []
        for msg in sample_messages:
            role = msg.get("role", "unknown")
            if role not in {"user", "assistant"}:
                continue
            raw_content = msg.get("content", "")
            text = _content_to_text(raw_content)
            img_count = _count_image_blocks(raw_content)
            truncated = False
            if len(text) > EXTRACT_MAX_CHARS_PER_MESSAGE:
                text = text[:EXTRACT_MAX_CHARS_PER_MESSAGE] + "…（内容过长，已截断）"
                truncated = True
            processed_messages.append({
                "role": role,
                "content": text,
                "source_id": msg.get("_source_message_id"),
                "img_count": img_count,
                "truncated": truncated,
            })

        active_indices = list(range(len(processed_messages)))
        while True:
            candidate_items = [processed_messages[i] for i in active_indices]
            new_conversation_text, _ = _format_conversation(candidate_items)
            if len(new_conversation_text) <= EXTRACT_MAX_CHARS_TOTAL:
                break
            active_indices.pop(0)

        # 逐字对比
        diff = list(
            difflib.unified_diff(
                old_conversation_text.splitlines(keepends=True),
                new_conversation_text.splitlines(keepends=True),
                fromfile="old_conversation_text",
                tofile="new_conversation_text",
            )
        )
        diff_str = "".join(diff)
        print("=== 回归测试 diff 开始 ===")
        print(diff_str, end="")
        print("=== 回归测试 diff 结束（上面应为空）===")
        self.assertEqual(diff_str, "")
        self.assertEqual(old_conversation_text, new_conversation_text)

    def test_multimodal_base64_image(self):
        """构造带 base64 大图与文字的列表 content，验证不含 base64，保留 [图片] 与文字，总长 < 60000"""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5_000_000}},
            {"type": "text", "text": "测试文字"},
        ]
        text = _content_to_text(content)
        img_count = _count_image_blocks(content)
        self.assertEqual(img_count, 1)

        # 构造 prompt
        prompt = f"请从以下对话中提取新的记忆：\n\n[证据不可用] 阿狸: {text}\n"

        self.assertNotIn("base64", prompt)
        self.assertNotIn("data:image", prompt)
        self.assertIn("[图片]", prompt)
        self.assertIn("测试文字", prompt)
        self.assertLess(len(prompt), 60000)
        self.assertEqual(text, "[图片]\n测试文字")

    def test_single_message_truncation_20000(self):
        """单条 20000 字符文本截断行为"""
        raw_text = "测" * 20000
        text = _content_to_text(raw_text)
        truncated = False
        if len(text) > EXTRACT_MAX_CHARS_PER_MESSAGE:
            text = text[:EXTRACT_MAX_CHARS_PER_MESSAGE] + "…（内容过长，已截断）"
            truncated = True

        self.assertTrue(truncated)
        self.assertEqual(len(text), 8000 + len("…（内容过长，已截断）"))
        self.assertTrue(text.startswith("测" * 8000))
        self.assertTrue(text.endswith("…（内容过长，已截断）"))

    def test_twelve_messages_discard_and_retention(self):
        """12 条各 8000 字符：丢弃旧消息直到不超 60000，且最后一条 user 与 assistant 保留"""
        messages = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "role": role,
                "content": f"第{i}条_" + ("A" * (8000 - 10)),
                "_source_message_id": 1000 + i,
            })

        processed_messages = []
        for msg in messages:
            role = msg.get("role", "unknown")
            text = _content_to_text(msg.get("content", ""))
            truncated = False
            if len(text) > EXTRACT_MAX_CHARS_PER_MESSAGE:
                text = text[:EXTRACT_MAX_CHARS_PER_MESSAGE] + "…（内容过长，已截断）"
                truncated = True
            processed_messages.append({
                "role": role,
                "content": text,
                "source_id": msg.get("_source_message_id"),
                "img_count": _count_image_blocks(msg.get("content", "")),
                "truncated": truncated,
            })

        last_user_idx = max(i for i, m in enumerate(processed_messages) if m["role"] == "user")
        last_assistant_idx = max(i for i, m in enumerate(processed_messages) if m["role"] == "assistant")
        protected_indices = {last_user_idx, last_assistant_idx}

        self.assertEqual(last_user_idx, 10)
        self.assertEqual(last_assistant_idx, 11)

        active_indices = list(range(len(processed_messages)))
        while True:
            candidate_items = [processed_messages[i] for i in active_indices]
            conversation_text, _ = _format_conversation(candidate_items)
            if len(conversation_text) <= EXTRACT_MAX_CHARS_TOTAL:
                break
            discard_pos = None
            for pos, idx in enumerate(active_indices):
                if idx not in protected_indices:
                    discard_pos = pos
                    break
            if discard_pos is None:
                break
            active_indices.pop(discard_pos)

        retained_items = [processed_messages[i] for i in active_indices]
        discarded_count = len(processed_messages) - len(retained_items)

        # 初始 12 条 * ~8000 字 > 96000 字；上限 60000 字
        self.assertLessEqual(len(conversation_text), EXTRACT_MAX_CHARS_TOTAL)
        self.assertGreater(discarded_count, 0)
        # 最后一条 user (第10条) 和最后一条 assistant (第11条) 必须保留
        self.assertIn(10, active_indices)
        self.assertIn(11, active_indices)
        self.assertTrue(any(item["content"].startswith("第10条_") for item in retained_items))
        self.assertTrue(any(item["content"].startswith("第11条_") for item in retained_items))

    def test_extract_memories_logging_and_call(self):
        """测试 extract_memories 实际调用时的日志输出与输入防泄漏"""
        import asyncio
        import io
        import unittest.mock as mock
        import memory_extractor

        sample_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 10000}},
                    {"type": "text", "text": "带图提问"},
                ],
                "_source_message_id": 55,
            },
            {
                "role": "assistant",
                "content": "回答完毕",
                "_source_message_id": 56,
            },
        ]

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "choices": [{"message": {"content": '[{"content": "用户喜欢发图", "importance": 6, "action": "new", "source_refs": [1]}]'}}],
            "usage": {"completion_tokens": 50},
        }

        captured_calls = []

        async def fake_post(client, url, key, payload):
            captured_calls.append(payload)
            return fake_resp

        with mock.patch.object(memory_extractor, "MEMORY_MODEL", "mock-model"), \
             mock.patch.object(memory_extractor, "MEMORY_API_KEY", "mock-key"), \
             mock.patch("shared.post_chat_completion", side_effect=fake_post), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = asyncio.run(memory_extractor.extract_memories(sample_messages))

        output = mock_stdout.getvalue()
        # 断言日志输出格式（只打数字，不打内容）
        self.assertIn("📏 提取输入: 2 条, 共", output)
        self.assertIn("图片块 1 个", output)
        self.assertIn("截断 0 处", output)
        self.assertIn("丢弃旧消息 0 条", output)
        self.assertNotIn("base64", output)
        self.assertNotIn("A" * 100, output)

        # 断言发给上游的请求内容
        self.assertEqual(len(captured_calls), 1)
        sent_body = captured_calls[0]
        sent_user_msg = sent_body["messages"][1]["content"]
        self.assertNotIn("base64", sent_user_msg)
        self.assertIn("[图片]", sent_user_msg)
        self.assertIn("带图提问", sent_user_msg)

        # 断言解析结果
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "用户喜欢发图")
        self.assertEqual(result[0]["source_message_ids"], [55])


if __name__ == "__main__":
    unittest.main()

