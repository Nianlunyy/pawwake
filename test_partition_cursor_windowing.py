"""Unit tests for cursor windowing and virtual round padding."""

import asyncio
from datetime import datetime, timezone, timedelta
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import shared
import partition_engine
import memory_pipeline
from db import conversations as db_conversations


class TestPartitionCursorWindowing(unittest.TestCase):

    def test_group_by_rounds_with_virtual_padding(self):
        """测试虚拟占位轮次与真实消息组装后的逻辑轮分组对齐。"""
        a_start_round = 20
        cutoff_dt = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
        cutoff_id = 199

        # 构造 a_start_round 个虚拟占位轮次
        padded_virtual_rounds = [
            {"role": "user", "content": "", "_virtual": True}
        ] * (a_start_round - 1) + [
            {
                "role": "user",
                "content": "",
                "id": cutoff_id,
                "created_at": cutoff_dt,
                "_virtual": True,
            }
        ]

        # 构造 Zone A (10 轮) 与 Zone B (5 轮) 的真实消息
        active_msgs = []
        msg_id_counter = 200
        for r in range(15):
            t_user = cutoff_dt + timedelta(minutes=r * 10 + 1)
            t_asst = cutoff_dt + timedelta(minutes=r * 10 + 2)
            active_msgs.append({
                "id": msg_id_counter,
                "role": "user",
                "content": f"用户问题 round {a_start_round + r}",
                "created_at": t_user,
            })
            msg_id_counter += 1
            active_msgs.append({
                "id": msg_id_counter,
                "role": "assistant",
                "content": f"AI回复 round {a_start_round + r}",
                "created_at": t_asst,
            })
            msg_id_counter += 1

        all_db_msgs = padded_virtual_rounds + active_msgs
        rounds = partition_engine.group_by_rounds(all_db_msgs)

        # 验证总轮数 = 20 (虚拟) + 15 (真实) = 35
        self.assertEqual(len(rounds), 35)

        # 验证前 20 轮全为虚拟占位轮次
        for r_idx in range(a_start_round):
            self.assertEqual(len(rounds[r_idx]), 1)
            self.assertTrue(rounds[r_idx][0].get("_virtual"))

        # 验证第 a_start_round 轮开始为真实 Zone A 消息
        first_real_round = rounds[a_start_round]
        self.assertEqual(len(first_real_round), 2)
        self.assertEqual(first_real_round[0]["content"], "用户问题 round 20")
        self.assertFalse(first_real_round[0].get("_virtual", False))

    def test_out_of_context_prefix_cutoff_anchor_reading(self):
        """验证 _out_of_context_prefix_cutoff 能精准提取虚拟末位的真实 cutoff 锚点，不报错无丢失。"""
        a_start_round = 5
        cutoff_dt = datetime(2026, 5, 1, 18, 30, 0, tzinfo=timezone.utc)
        cutoff_id = 88

        padded_virtual_rounds = [
            {"role": "user", "content": "", "_virtual": True}
        ] * (a_start_round - 1) + [
            {
                "role": "user",
                "content": "",
                "id": cutoff_id,
                "created_at": cutoff_dt,
                "_virtual": True,
            }
        ]

        # 真实活跃消息
        active_msgs = [
            {"id": 89, "role": "user", "content": "问题5", "created_at": cutoff_dt + timedelta(minutes=1)},
            {"id": 90, "role": "assistant", "content": "回答5", "created_at": cutoff_dt + timedelta(minutes=2)},
        ]

        recall_history = padded_virtual_rounds + active_msgs

        # 调用 memory_pipeline 中的 cutoff 提取函数
        cutoff = memory_pipeline._out_of_context_prefix_cutoff(recall_history, a_start_round)
        self.assertIsNotNone(cutoff)
        self.assertEqual(cutoff[0], cutoff_dt)
        self.assertEqual(cutoff[1], cutoff_id)

    def test_build_partitioned_messages_ignores_virtual_rounds(self):
        """验证 build_partitioned_messages 绝对切片只纳入真实 A/B 区，虚拟轮次完全被抛弃。"""
        a_start_round = 2
        cutoff_dt = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
        padded_virtual_rounds = [
            {"role": "user", "content": "", "_virtual": True},
            {"role": "user", "content": "", "id": 10, "created_at": cutoff_dt, "_virtual": True},
        ]

        # 真实 A 区 (X=2) + B 区 (1轮)
        active_msgs = [
            # Round 2 (Zone A - 0)
            {"id": 11, "role": "user", "content": "A1问题", "created_at": cutoff_dt + timedelta(minutes=1)},
            {"id": 12, "role": "assistant", "content": "A1回答", "created_at": cutoff_dt + timedelta(minutes=2)},
            # Round 3 (Zone A - 1)
            {"id": 13, "role": "user", "content": "A2问题", "created_at": cutoff_dt + timedelta(minutes=3)},
            {"id": 14, "role": "assistant", "content": "A2回答", "created_at": cutoff_dt + timedelta(minutes=4)},
            # Round 4 (Zone B - 0)
            {"id": 15, "role": "user", "content": "B1问题", "created_at": cutoff_dt + timedelta(minutes=5)},
            {"id": 16, "role": "assistant", "content": "B1回答", "created_at": cutoff_dt + timedelta(minutes=6)},
        ]

        client_new = [{"role": "user", "content": "当前用户新问题"}]
        all_msgs = padded_virtual_rounds + active_msgs + client_new

        mock_state = {
            "summary_parts": ["历史对话基础摘要"],
            "a_start_round": a_start_round,
        }

        with patch.object(shared, "CACHE_PARTITION_X", 2):
            with patch.object(db_conversations, "get_session_cache_state", new=AsyncMock(return_value=mock_state)):
                messages = asyncio.run(
                    partition_engine.build_partitioned_messages(
                        "sess_test",
                        all_msgs,
                        base_prompt="System Prompt",
                        user_message="当前用户新问题",
                    )
                )

        # 检查输出消息列表中绝对不包含虚拟占位消息
        contents = [str(m.get("content")) for m in messages]
        for c in contents:
            self.assertNotIn("_virtual", c)

        # 检查输出中包含摘要、A区真实消息、B区真实消息、当前user
        self.assertTrue(any("历史对话基础摘要" in c for c in contents))
        self.assertTrue(any("A1问题" in c for c in contents))
        self.assertTrue(any("A1回答" in c for c in contents))
        self.assertTrue(any("A2问题" in c for c in contents))
        self.assertTrue(any("B1问题" in c for c in contents))
        self.assertTrue(any("当前用户新问题" in c for c in contents))

    def test_memory_extraction_messages_never_polluted_by_virtual_rounds(self):
        """验证 db_extraction_msgs 仅含真实活跃消息，_build_memory_extraction_messages 绝对只切真实上下文。"""
        real_active_msgs = [
            {"role": "user", "content": "近期用户提问", "_source_message_id": 101},
            {"role": "assistant", "content": "近期AI回复", "_source_message_id": 102},
        ]
        # db_extraction_msgs 保持只有真实消息
        db_extraction_msgs = list(real_active_msgs)

        extracted_msgs, round_count = partition_engine._build_memory_extraction_messages(
            db_extraction_msgs,
            assistant_msg="当前最终AI回复",
            interval=1,
            assistant_source_id=103,
        )

        self.assertEqual(round_count, 1)
        self.assertEqual(len(extracted_msgs), 3)  # user(101), asst(102), asst(103)
        self.assertEqual(extracted_msgs[0]["_source_message_id"], 101)
        self.assertEqual(extracted_msgs[1]["_source_message_id"], 102)
        self.assertEqual(extracted_msgs[2]["_source_message_id"], 103)

    def test_db_get_partition_conversation_messages_fallback(self):
        """测试越界或异常时安全降级到全量查询。"""
        mock_pool = MagicMock()
        mock_conn = AsyncMock()

        # 模拟 start_row 为 None（越界）
        mock_conn.fetchrow.return_value = None
        mock_conn.fetch.return_value = [
            {"id": 1, "role": "user", "content": "fallback msg", "metadata": None, "created_at": datetime.now(timezone.utc)}
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with patch("db.core.get_pool", new=AsyncMock(return_value=mock_pool)):
            rows, cutoff, a_start, is_sliced = asyncio.run(
                db_conversations.get_partition_conversation_messages("test_sid", a_start_round=999)
            )

        self.assertFalse(is_sliced)
        self.assertIsNone(cutoff)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "fallback msg")


if __name__ == "__main__":
    unittest.main()
