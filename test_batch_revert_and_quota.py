"""Unit tests for batch/all revert-merge, pinned memory quota decoupling, and consolidation prompt."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import memory_consolidator
import memory_pipeline
import shared
from db import memories as db_memories
from routes.memories import api_batch_revert_merge


class TestMemoryPipelineQuota(unittest.IsolatedAsyncioTestCase):
    async def test_build_system_prompt_pinned_does_not_deduct_quota(self):
        """测试 build_system_prompt_with_memories 中 pinned 记忆不再从 MAX_MEMORIES_INJECT 中扣减名额。"""
        with patch.object(shared, "MEMORY_ENABLED", True), \
             patch.object(shared, "MAX_MEMORIES_INJECT", 2):
            # 存在 2 条常驻记忆
            pinned_memories = [
                {"id": 1, "content": "常驻1", "created_at": "2026-11-01"},
                {"id": 2, "content": "常驻2", "created_at": "2026-11-01"},
            ]
            with patch("memory_pipeline._claim_due_reminders", new_callable=AsyncMock, return_value=[]), \
                 patch("memory_pipeline._get_pinned_memories_or_empty", new_callable=AsyncMock, return_value=pinned_memories), \
                 patch("memory_pipeline._search_memories_or_empty", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = [{"id": 3, "content": "动态记忆3", "created_at": "2026-11-02"}]

                result = await memory_pipeline.build_system_prompt_with_memories("用户问句", "基础人设")

                # limit 应该依然是 2，而不是 2 - 2 = 0
                mock_search.assert_called_once()
                args, _ = mock_search.call_args
                self.assertEqual(args[0], "用户问句")
                self.assertEqual(args[1], 2)
                self.assertIn("【常驻记忆】", result)
                self.assertIn("【从过往对话中检索到的相关记忆】", result)
                self.assertIn("常驻1", result)
                self.assertIn("动态记忆3", result)

    async def test_build_memory_text_pinned_does_not_deduct_quota(self):
        """测试 build_memory_text 中 pinned 记忆不再从 MAX_MEMORIES_INJECT 中扣减名额。"""
        with patch.object(shared, "MEMORY_ENABLED", True), \
             patch.object(shared, "MAX_MEMORIES_INJECT", 2):
            pinned_memories = [
                {"id": 1, "content": "常驻1", "created_at": "2026-11-01"},
                {"id": 2, "content": "常驻2", "created_at": "2026-11-01"},
            ]
            with patch("memory_pipeline._claim_due_reminders", new_callable=AsyncMock, return_value=[]), \
                 patch("memory_pipeline._get_pinned_memories_or_empty", new_callable=AsyncMock, return_value=pinned_memories), \
                 patch("db.conversations.get_active_seen_memory_ids", new_callable=AsyncMock, return_value=[]), \
                 patch("memory_pipeline._search_memories_for_injection", new_callable=AsyncMock) as mock_search:
                mock_search.return_value = [{"id": 3, "content": "动态记忆3", "created_at": "2026-11-02"}]

                result = await memory_pipeline.build_memory_text("用户问句", session_id="s1")

                mock_search.assert_called_once()
                args, _ = mock_search.call_args
                self.assertEqual(args[0], "用户问句")
                self.assertEqual(args[1], 2)
                self.assertIn("<pinned_memories>", result)
                self.assertIn("<retrieved_memories>", result)
                self.assertIn("常驻1", result)
                self.assertIn("动态记忆3", result)


class TestConsolidationPrompt(unittest.TestCase):
    def test_prompt_anti_mashup_instructions(self):
        """验证自动整理 Prompt 中包含禁止跨时段、跨话题强行拼凑的硬性约束。"""
        prompt = memory_consolidator.ORGANIZE_DRAFT_PROMPT
        self.assertIn("严格按「单一具体事件/同一具体话题」分组", prompt)
        self.assertIn("严禁将同一天内不同话题", prompt)
        self.assertIn("绝对禁止为了凑数而跨话题合并", prompt)


class TestRevertMergeBatch(unittest.IsolatedAsyncioTestCase):
    def _create_mock_conn(self):
        mock_conn = MagicMock()
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)
        mock_conn.transaction = MagicMock(return_value=mock_tx)
        return mock_conn

    async def test_batch_revert_specified_ids(self):
        """测试指定 ID 列表的批量撤回合并：成功恢复碎片并删除事件记忆，跳过常驻记忆。"""
        mock_conn = self._create_mock_conn()

        # 模拟 fetch 结果：101 是普通事件（merged_from=[1, 2]），102 是常驻事件（is_pinned=True）
        rows_result = [
            {
                "id": 101, "layer": 2, "is_active": True, "is_pinned": False,
                "merged_from": [1, 2], "has_pending_reminder": False, "protected_by_supersession": False
            },
            {
                "id": 102, "layer": 2, "is_active": True, "is_pinned": True,
                "merged_from": [3, 4], "has_pending_reminder": False, "protected_by_supersession": False
            },
        ]
        # 针对 101 的 sources 检查：[1, 2] 均存在
        sources_101 = [{"id": 1}, {"id": 2}]

        async def mock_fetch(query, *params):
            if "WHERE memory.id = ANY($1::int[])" in query:
                return rows_result
            if "WHERE id = ANY($1::int[])" in query:
                return sources_101
            return []

        mock_conn.fetch = mock_fetch
        mock_conn.execute = AsyncMock(return_value="UPDATE 2")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch("db.core.get_pool", return_value=mock_pool):
            result = await db_memories.revert_merge_batch(memory_ids=[101, 102], revert_all=False)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["reverted"], 1)   # 101 被撤回
            self.assertEqual(result["restored"], 2)   # 恢复碎片 1, 2
            self.assertEqual(result["skipped"], 1)    # 102 因常驻被跳过

    async def test_batch_revert_all_nested(self):
        """测试 revert_all=True 时的多轮解包能力（事件 200 合并了事件 100，解开后事件 100 进一步解开为碎片 1, 2）。"""
        mock_conn = self._create_mock_conn()

        pass_counter = 0

        async def mock_fetch(query, *params):
            nonlocal pass_counter
            if "WHERE memory.layer = 2" in query:
                pass_counter += 1
                if pass_counter == 1:
                    # 第一轮：存在事件 200 (merged_from=[100, 3])
                    return [{
                        "id": 200, "layer": 2, "is_active": True, "is_pinned": False,
                        "merged_from": [100, 3], "has_pending_reminder": False, "protected_by_supersession": False
                    }]
                elif pass_counter == 2:
                    # 第二轮：事件 100 恢复为 active，被扫描到 (merged_from=[1, 2])
                    return [{
                        "id": 100, "layer": 2, "is_active": True, "is_pinned": False,
                        "merged_from": [1, 2], "has_pending_reminder": False, "protected_by_supersession": False
                    }]
                else:
                    return []
            if "WHERE id = ANY($1::int[])" in query:
                ids = params[0]
                return [{"id": i} for i in ids]
            return []

        mock_conn.fetch = mock_fetch
        mock_conn.execute = AsyncMock(return_value="OK")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch("db.core.get_pool", return_value=mock_pool):
            result = await db_memories.revert_merge_batch(revert_all=True)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["reverted"], 2)  # 200 和 100 均被撤回
            self.assertEqual(result["restored"], 3)  # 最终恢复了 1, 2, 3 (100 在第二轮被移出 restored)
            self.assertEqual(result["skipped"], 0)

    async def test_batch_revert_skips_incomplete_sources(self):
        """测试来源碎片已被物理清理的事件记忆自动安全跳过。"""
        mock_conn = self._create_mock_conn()

        # 记忆 103 的 merged_from=[1, 2]，但数据库里只剩 1
        rows_result = [{
            "id": 103, "layer": 2, "is_active": True, "is_pinned": False,
            "merged_from": [1, 2], "has_pending_reminder": False, "protected_by_supersession": False
        }]
        sources_result = [{"id": 1}]

        async def mock_fetch(query, *params):
            if "WHERE memory.id = ANY($1::int[])" in query:
                return rows_result
            if "WHERE id = ANY($1::int[])" in query:
                return sources_result
            return []

        mock_conn.fetch = mock_fetch
        mock_conn.execute = AsyncMock()

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch("db.core.get_pool", return_value=mock_pool):
            result = await db_memories.revert_merge_batch(memory_ids=[103], revert_all=False)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["reverted"], 0)
            self.assertEqual(result["restored"], 0)
            self.assertEqual(result["skipped"], 1)


class TestBatchRevertRoute(unittest.IsolatedAsyncioTestCase):
    async def test_route_validation(self):
        """测试路由层对非法入参的校验。"""
        mock_req = MagicMock()

        with patch.object(shared, "MEMORY_ENABLED", False):
            res = await api_batch_revert_merge(mock_req)
            self.assertEqual(res, {"error": "记忆系统未启用"})

        with patch.object(shared, "MEMORY_ENABLED", True):
            # 空 IDs 且 all=False
            mock_req.json = AsyncMock(return_value={"ids": []})
            res = await api_batch_revert_merge(mock_req)
            self.assertEqual(res.status_code, 400)

            # 非整数 IDs
            mock_req.json = AsyncMock(return_value={"ids": [True, "abc"]})
            res = await api_batch_revert_merge(mock_req)
            self.assertEqual(res.status_code, 400)

    async def test_route_success(self):
        """测试路由成功转发到 db_memories.revert_merge_batch。"""
        mock_req = MagicMock()
        mock_req.json = AsyncMock(return_value={"all": True})

        with patch.object(shared, "MEMORY_ENABLED", True), \
             patch("db.memories.revert_merge_batch", new_callable=AsyncMock) as mock_batch:
            mock_batch.return_value = {"status": "ok", "reverted": 5, "restored": 12, "skipped": 0}
            res = await api_batch_revert_merge(mock_req)
            mock_batch.assert_called_once_with(memory_ids=None, revert_all=True)
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["reverted"], 5)


if __name__ == "__main__":
    unittest.main()
