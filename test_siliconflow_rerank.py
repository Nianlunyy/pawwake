"""Unit tests for SiliconFlow / Qwen3-Reranker integration and single-candidate normalization fix."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import recall_rerank
import shared
from db.memories import _anchored_semantic_normalize


class TestSiliconflowRerank(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orig_rerank_base = shared.RERANK_BASE_URL
        self.orig_rerank_key = shared.RERANK_API_KEY
        self.orig_rerank_model = shared.RERANK_MODEL
        self.orig_emb_base = shared.EMBEDDING_BASE_URL
        self.orig_emb_key = shared.EMBEDDING_API_KEY
        self.orig_api_base = shared.API_BASE_URL
        self.orig_api_key = shared.API_KEY

    def tearDown(self):
        shared.RERANK_BASE_URL = self.orig_rerank_base
        shared.RERANK_API_KEY = self.orig_rerank_key
        shared.RERANK_MODEL = self.orig_rerank_model
        shared.EMBEDDING_BASE_URL = self.orig_emb_base
        shared.EMBEDDING_API_KEY = self.orig_emb_key
        shared.API_BASE_URL = self.orig_api_base
        shared.API_KEY = self.orig_api_key

    def test_resolve_rerank_config(self):
        # 1. 专用地址与 Key 齐全时优先使用
        shared.RERANK_BASE_URL = "https://custom.rerank/v1"
        shared.RERANK_API_KEY = "custom-key"
        shared.EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
        shared.EMBEDDING_API_KEY = "emb-key"
        b, k = recall_rerank.resolve_rerank_config()
        self.assertEqual(b, "https://custom.rerank/v1")
        self.assertEqual(k, "custom-key")

        # 2. 专用只填了一个 -> 返回 (None, None) 错误
        shared.RERANK_BASE_URL = "https://custom.rerank/v1"
        shared.RERANK_API_KEY = ""
        b, k = recall_rerank.resolve_rerank_config()
        self.assertIsNone(b)
        self.assertIsNone(k)

        # 3. 专用都留空 -> 优先复用 Embedding 配置
        shared.RERANK_BASE_URL = ""
        shared.RERANK_API_KEY = ""
        shared.EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
        shared.EMBEDDING_API_KEY = "emb-key"
        b, k = recall_rerank.resolve_rerank_config()
        self.assertEqual(b, "https://api.siliconflow.cn/v1")
        self.assertEqual(k, "emb-key")

        # 4. 专用与 Embedding 都留空 -> 回退到 API 基础连接
        shared.EMBEDDING_BASE_URL = ""
        shared.EMBEDDING_API_KEY = ""
        shared.API_BASE_URL = "https://api.fallback.com/v1"
        shared.API_KEY = "api-key"
        b, k = recall_rerank.resolve_rerank_config()
        self.assertEqual(b, "https://api.fallback.com/v1")
        self.assertEqual(k, "api-key")

    def test_build_standard_rerank_url(self):
        self.assertEqual(
            recall_rerank.build_standard_rerank_url("https://api.siliconflow.cn/v1"),
            "https://api.siliconflow.cn/v1/rerank",
        )
        self.assertEqual(
            recall_rerank.build_standard_rerank_url("https://api.siliconflow.cn/v1/"),
            "https://api.siliconflow.cn/v1/rerank",
        )
        self.assertEqual(
            recall_rerank.build_standard_rerank_url("https://api.siliconflow.cn/v1/embeddings"),
            "https://api.siliconflow.cn/v1/rerank",
        )
        self.assertEqual(
            recall_rerank.build_standard_rerank_url("https://api.siliconflow.cn/v1/rerank"),
            "https://api.siliconflow.cn/v1/rerank",
        )

    def test_qwen_request_body_and_pick_standard(self):
        candidates = [
            {"id": 1, "content": "阿狸周四有普拉提课", "event_date": "2026-10-29"},
            {"id": 2, "content": "阿狸爱吃老三样", "created_at": "2026-10-21"},
            {"id": 3, "content": "无关闲聊文字", "created_at": "2026-10-15"},
        ]
        # 1. 验证千问模型包含 instruction
        body_qwen = recall_rerank.standard_request_body(
            "Qwen/Qwen3-Reranker-8B", "周四有什么课", candidates, "memory", include_instruction=True
        )
        self.assertIn("instruction", body_qwen)
        self.assertEqual(len(body_qwen["documents"]), 3)
        self.assertEqual(body_qwen["documents"][0], "[2026-10-29] 阿狸周四有普拉提课")

        # 2. 验证非千问模型不带 instruction
        body_bge = recall_rerank.standard_request_body(
            "BAAI/bge-reranker-v2-m3", "周四有什么课", candidates, "memory", include_instruction=True
        )
        self.assertNotIn("instruction", body_bge)

        # 3. 验证 pick_standard 过滤低分候选项
        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.65},
                {"index": 2, "relevance_score": 0.18},  # 应当被过滤，不凑数
            ]
        }
        picked, gate = recall_rerank.pick_standard(mock_response, len(candidates))
        self.assertEqual(gate, 0.95)
        self.assertEqual(picked, [0, 1])  # 只有 index 0 和 1 入选

        # 4. 全低分时总闸门触发，全部剔除 (gate < 0.5)
        mock_all_low = {
            "results": [
                {"index": 0, "relevance_score": 0.35},
                {"index": 1, "relevance_score": 0.22},
                {"index": 2, "relevance_score": 0.10},
            ]
        }
        picked_low, gate_low = recall_rerank.pick_standard(mock_all_low, len(candidates))
        self.assertEqual(gate_low, 0.35)
        self.assertEqual(picked_low, [])

    @patch("httpx.AsyncClient.post")
    async def test_rerank_standard_flow(self, mock_post):
        shared.RERANK_BASE_URL = ""
        shared.RERANK_API_KEY = ""
        shared.EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
        shared.EMBEDDING_API_KEY = "sk-test-key"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = {
            "model": "Qwen/Qwen3-Reranker-8B",
            "results": [
                {"index": 0, "relevance_score": 0.88},
                {"index": 1, "relevance_score": 0.32},
            ],
        }
        mock_post.return_value = mock_resp

        candidates = [
            {"id": 10, "content": "相关记忆"},
            {"id": 11, "content": "低分不相关"},
        ]
        res = await recall_rerank.rerank(
            "Qwen/Qwen3-Reranker-8B", "查询测试", candidates, "memory"
        )
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], 10)

    @patch("httpx.AsyncClient.post")
    async def test_rerank_instruction_400_fallback(self, mock_post):
        """测试某些厂商不支持 instruction 返回 400 时自动剥离重试"""
        shared.RERANK_BASE_URL = "https://api.siliconflow.cn/v1"
        shared.RERANK_API_KEY = "sk-test"

        resp_400 = MagicMock()
        resp_400.status_code = 400
        resp_400.is_success = False

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.is_success = True
        resp_200.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.91}]
        }

        mock_post.side_effect = [resp_400, resp_200]

        candidates = [{"id": 1, "content": "普拉提"}]
        res = await recall_rerank.rerank("Qwen/Qwen3-Reranker-8B", "普拉提", candidates, "memory")
        self.assertEqual(len(res), 1)
        self.assertEqual(mock_post.call_count, 2)
        # 第二次重试应该去掉了 instruction
        second_call_json = mock_post.call_args_list[1].kwargs.get("json", {})
        self.assertNotIn("instruction", second_call_json)

    def test_single_candidate_normalization_fix(self):
        """测试单候选记忆在归一化时不被自己减自己归零"""
        # 1. 之前 Bug：只有一条老三样 (0.412)，因 0.412 - 0.412 归零
        scores = {1: 0.412}
        norm = _anchored_semantic_normalize(scores, 0.70)
        expected = 0.412 / 0.70
        self.assertAlmostEqual(norm[1], expected, places=4)
        self.assertGreater(norm[1], 0.58)

        # 2. 多候选池：下界仍取批内最低，维持原有防抬分机制
        multi_scores = {1: 0.60, 2: 0.20}
        norm_multi = _anchored_semantic_normalize(multi_scores, 0.70)
        # min_v = 0.20, max_v = 0.70, spread = 0.50
        # id 1: (0.60 - 0.20) / 0.50 = 0.80
        # id 2: (0.20 - 0.20) / 0.50 = 0.00
        self.assertAlmostEqual(norm_multi[1], 0.80, places=4)
        self.assertAlmostEqual(norm_multi[2], 0.00, places=4)

    async def test_dashboard_settings_validation(self):
        """测试 Dashboard 保存重排设置放开域名限制并支持复用 Embedding"""
        from starlette.requests import Request
        from routes.dashboard import save_settings

        shared.EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"

        # 模拟 Request（不传 embedding 重建参数，测试重排字段校验）
        req_data = {
            "MEMORY_RERANK_ENABLED": True,
            "RERANK_MODEL": "Qwen/Qwen3-Reranker-8B",
            "RERANK_BASE_URL": "",
            "RERANK_API_KEY": "",
        }
        scope = {
            "type": "http",
            "method": "PUT",
            "path": "/api/settings",
            "headers": [(b"content-type", b"application/json")],
        }
        req = Request(scope)
        req._json = req_data

        with patch("db.core.set_gateway_config", new=AsyncMock()):
            res = await save_settings(req)
            # 确保不会返回 "重排只支持 OpenRouter、Zenmux 和 TypeSafe" 错误
            if isinstance(res, dict) and "error" in res:
                self.fail(f"save_settings returned unexpected error: {res['error']}")
            self.assertEqual(res.get("status"), "ok")


if __name__ == "__main__":
    unittest.main()
