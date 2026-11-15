import unittest
from unittest.mock import AsyncMock, patch

import memory_pipeline
import partition_engine


CARD_1 = """好的，给你做了一个小组件：
```cuplivo-ui
<!-- title: 每日饮水打卡 -->
<div>喝水 +1</div>
```
记得按时喝水哦。"""

CARD_2 = """又更新了一版卡片：
```html-preview
<title>晚安小夜灯</title>
<button>关灯</button>
```"""

CARD_UNTITLED = """```cuplivo-ui
<p>无标题</p>
```"""


class InteractiveUiP0Tests(unittest.TestCase):
    def test_fold_historical_keeps_only_last_card_and_preserves_prefix(self):
        msgs_turn_4 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "做个喝水打卡"},
            {"role": "assistant", "content": CARD_1},
            {"role": "user", "content": "再做个夜灯"},
            {"role": "assistant", "content": CARD_2},
            {"role": "user", "content": "今天辛苦啦"},
            {"role": "assistant", "content": "不客气～"},
            {"role": "user", "content": "晚安"},
        ]

        folded_turn_4 = partition_engine.fold_historical_interactive_ui_messages(msgs_turn_4)

        # 第一张卡被折叠并提取了注释标题；最后一张卡完整保留；原列表未被就地修改
        self.assertIn("[曾生成互动界面：每日饮水打卡（源码已折叠）]", folded_turn_4[2]["content"])
        self.assertNotIn("<div>喝水 +1</div>", folded_turn_4[2]["content"])
        self.assertEqual(folded_turn_4[4]["content"], CARD_2)
        self.assertEqual(msgs_turn_4[2]["content"], CARD_1)

        # 继续聊两轮未生成新卡：历史前缀字节完全不变（保护 Prefix Cache）
        msgs_turn_6 = msgs_turn_4 + [
            {"role": "assistant", "content": "晚安好梦"},
            {"role": "user", "content": "早安"},
        ]
        folded_turn_6 = partition_engine.fold_historical_interactive_ui_messages(msgs_turn_6)
        self.assertEqual(folded_turn_6[:8], folded_turn_4[:8])

    def test_fold_historical_supports_multimodal_blocks_and_untitled_fallback(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": CARD_UNTITLED, "cache_control": {"type": "ephemeral"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": CARD_2},
                ],
            },
        ]
        folded = partition_engine.fold_historical_interactive_ui_messages(msgs)
        self.assertEqual(
            folded[0]["content"][0]["text"],
            "[曾生成互动界面：未命名卡片（源码已折叠）]",
        )
        self.assertEqual(folded[0]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(folded[1]["content"][0]["text"], CARD_2)

    def test_fold_historical_handles_inline_backticks_inside_script(self):
        card_with_inline_backticks = (
            "```cuplivo-ui\n"
            "<!-- title: 代码演示卡 -->\n"
            "<script>const tpl = '```inline```';</script>\n"
            "<div>完整闭合</div>\n"
            "```"
        )
        msgs = [
            {"role": "assistant", "content": card_with_inline_backticks},
            {"role": "assistant", "content": CARD_2},
        ]
        folded = partition_engine.fold_historical_interactive_ui_messages(msgs)
        self.assertEqual(
            folded[0]["content"],
            "[曾生成互动界面：代码演示卡（源码已折叠）]",
        )
        self.assertNotIn("<div>完整闭合</div>", folded[0]["content"])
        self.assertEqual(folded[1]["content"], CARD_2)

    def test_strip_ui_state_suffix_requires_literal_brackets(self):
        # 正常剥离末尾状态元数据（含多行 JSON 或仅编号）
        raw_json = '继续剧情\n\n[ui-state #0: {"hp": 80,\n "mood": "calm"}]'
        self.assertEqual(partition_engine.strip_ui_state_suffix(raw_json), "继续剧情")

        raw_no_colon = "点了一下\n[ui-state #2]"
        self.assertEqual(partition_engine.strip_ui_state_suffix(raw_no_colon), "点了一下")

        # 若正则未转义 `[`，旧字符类会在末尾遇到字母/数字时误匹配；确保普通字母末尾不被误删
        normal_text = "我们今天去公园玩了一圈 i"
        self.assertEqual(partition_engine.strip_ui_state_suffix(normal_text), normal_text)

        middle_tag = "[ui-state #0: {}] 是刚才的状态，但后面还有正文"
        self.assertEqual(partition_engine.strip_ui_state_suffix(middle_tag), middle_tag)


class InteractiveUiP0AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_summary_folds_all_ui_cards_and_strips_ui_state(self):
        captured = {}

        async def fake_call_summary(prompt, max_tokens=2500):
            captured["prompt"] = prompt
            return "摘要内容"

        messages = [
            {"role": "user", "content": '点亮夜灯\n\n[ui-state #0: {"lamp": "on"}]'},
            {"role": "assistant", "content": CARD_2},
        ]
        with patch.object(partition_engine.shared, "CACHE_SUMMARY_MODEL", "test-summary-model"), \
             patch.object(partition_engine, "_call_summary_model", side_effect=fake_call_summary):
            summary = await partition_engine.generate_summary(messages, session_id="s1")

        self.assertEqual(summary, "摘要内容")
        prompt = captured["prompt"]
        self.assertNotIn("ui-state", prompt)
        self.assertNotIn("<button>关灯</button>", prompt)
        self.assertIn("[曾生成互动界面：晚安小夜灯（源码已折叠）]", prompt)
        self.assertIn("点亮夜灯", prompt)

    async def test_memory_pipeline_reroll_before_strip_and_clean_extraction(self):
        user_click_2 = '确认选择\n\n[ui-state #0: {"choice": "B"}]'
        last_user_click_1 = '确认选择\n\n[ui-state #0: {"choice": "A"}]'
        captured = {}

        async def fake_get_candidates(query):
            captured["query"] = query
            return []

        async def fake_extract(msgs, existing_memories=None, **kwargs):
            captured["extract_msgs"] = msgs
            return []

        with patch.object(memory_pipeline.shared, "MEMORY_ENABLED", True), \
             patch.object(memory_pipeline.shared, "MEMORY_EXTRACT_INTERVAL", 1), \
             patch.object(memory_pipeline.memory_extractor, "MEMORY_MODEL", "test-model"), \
             patch.object(memory_pipeline.db_conversations, "get_last_user_content", new=AsyncMock(return_value=last_user_click_1)), \
             patch.object(memory_pipeline.db_conversations, "update_last_assistant_message", new=AsyncMock()) as mock_reroll_update, \
             patch.object(memory_pipeline.db_conversations, "save_message", new=AsyncMock(side_effect=[101, 102])), \
             patch.object(memory_pipeline.db_memories, "get_extraction_candidates", side_effect=fake_get_candidates), \
             patch.object(memory_pipeline, "extract_memories", side_effect=fake_extract):
            await memory_pipeline.process_memories_background(
                session_id="s1",
                user_msg=user_click_2,
                assistant_msg=CARD_1,
                model="test-model",
                context_messages=None,
            )

        # 1) 文案相同但 [ui-state] 不同：绝不能误判为 re-roll
        mock_reroll_update.assert_not_awaited()
        # 2) 送入候选检索与记忆提取前必须剥离 [ui-state] 并折叠互动卡片源码
        self.assertNotIn("ui-state", captured["query"])
        self.assertNotIn("<div>喝水 +1</div>", captured["query"])
        self.assertEqual(captured["extract_msgs"][0]["content"], "确认选择")
        self.assertIn("[曾生成互动界面：每日饮水打卡（源码已折叠）]", captured["extract_msgs"][1]["content"])
        self.assertNotIn("<div>喝水 +1</div>", captured["extract_msgs"][1]["content"])

    async def test_memory_pipeline_exact_same_ui_state_triggers_reroll(self):
        same_click = '确认选择\n\n[ui-state #0: {"choice": "A"}]'
        with patch.object(memory_pipeline.shared, "MEMORY_ENABLED", True), \
             patch.object(memory_pipeline.shared, "MEMORY_EXTRACT_INTERVAL", 1), \
             patch.object(memory_pipeline.memory_extractor, "MEMORY_MODEL", "test-model"), \
             patch.object(memory_pipeline.db_conversations, "get_last_user_content", new=AsyncMock(return_value=same_click)), \
             patch.object(memory_pipeline.db_conversations, "update_last_assistant_message", new=AsyncMock(return_value=202)) as mock_reroll_update, \
             patch.object(memory_pipeline.db_conversations, "save_message", new=AsyncMock()) as mock_save, \
             patch.object(memory_pipeline.db_memories, "get_extraction_candidates", new=AsyncMock(return_value=[])), \
             patch.object(memory_pipeline, "extract_memories", new=AsyncMock(return_value=[])) as mock_extract:
            await memory_pipeline.process_memories_background(
                session_id="s1",
                user_msg=same_click,
                assistant_msg="新回复",
                model="test-model",
                context_messages=None,
            )

        mock_reroll_update.assert_awaited_once_with("s1", "新回复", "test-model")
        mock_save.assert_not_awaited()
        mock_extract.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
