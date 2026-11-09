"""
test_oauth_mcp.py: 针对 OAuth 2.1 握手层、/mcp/ 鉴权保护与拦截、PKCE 及 MCP 工具升级的单元测试
"""

import asyncio
import base64
import contextlib
import hashlib
import os
import time
import unittest
import urllib.parse
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import auth
import mcp_server
import routes.oauth_mcp as oauth_mcp
import shared
from db import memories as db_memories
from main import app


@contextlib.contextmanager
def mcp_test_client():
    """Helper context manager to run MCPServer session manager cleanly in tests."""
    sm = mcp_server.server.session_manager
    sm._has_started = False
    sm._task_group = None
    with TestClient(app, follow_redirects=False) as c:
        yield c


class TestOAuthMCPHandshake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PUBLIC_BASE_URL"] = "https://pawwake.example.com"
        os.environ["OAUTH_ALLOWED_REDIRECT_HOSTS"] = (
            "gemini.google.com,victim.example.com,attacker.example.com,allowed.example.com"
        )
        cls.client = TestClient(app, follow_redirects=False)

    def setUp(self):
        self.secret = "test-gateway-secret-12345"
        self.redirect_uri = "https://gemini.google.com/oauth/callback"
        self.state = "test-state-abc"
        # PKCE S256 pair
        self.verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        digest = hashlib.sha256(self.verifier.encode("ascii")).digest()
        self.challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

        oauth_mcp._authorize_failures.clear()
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            self.client_id = oauth_mcp.create_stateless_client_id([self.redirect_uri])
        oauth_mcp._registered_clients[self.client_id] = {
            "client_id": self.client_id,
            "client_name": "Google Gemini",
            "redirect_uris": [self.redirect_uri],
        }

    # ============================================================
    # 1. 资源与服务发现端点 (RFC 8414 / RFC 9728)
    # ============================================================

    def test_protected_resource_metadata(self):
        """测试 /.well-known/oauth-protected-resource 与 /mcp 返回标准元数据"""
        for path in [
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
        ]:
            resp = self.client.get(path, headers={"Host": "pawwake.example.com"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["resource"], "https://pawwake.example.com/mcp/")
            self.assertIn("https://pawwake.example.com", data["authorization_servers"])

    def test_authorization_server_metadata(self):
        """测试 /.well-known/oauth-authorization-server 与 openid-configuration 元数据"""
        for path in [
            "/.well-known/oauth-authorization-server",
            "/.well-known/openid-configuration",
        ]:
            resp = self.client.get(path, headers={"Host": "pawwake.example.com"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["issuer"], "https://pawwake.example.com")
            self.assertEqual(data["authorization_endpoint"], "https://pawwake.example.com/oauth/authorize")
            self.assertEqual(data["token_endpoint"], "https://pawwake.example.com/oauth/token")
            self.assertEqual(data["registration_endpoint"], "https://pawwake.example.com/oauth/register")
            self.assertEqual(data["revocation_endpoint"], "https://pawwake.example.com/oauth/revoke")
            self.assertIn("code", data["response_types_supported"])
            self.assertIn("authorization_code", data["grant_types_supported"])
            self.assertIn("refresh_token", data["grant_types_supported"])
            self.assertEqual(data["code_challenge_methods_supported"], ["S256"])
            self.assertNotIn("plain", data["code_challenge_methods_supported"])
            self.assertIn("mcp", data["scopes_supported"])

    # ============================================================
    # 2. 动态客户端注册 (DCR, RFC 7591)
    # ============================================================

    def test_dynamic_client_registration(self):
        """测试 POST /oauth/register 返回 201 Created 与每次随机生成的 client_id"""
        resp = self.client.post(
            "/oauth/register",
            json={
                "client_name": "Google Gemini",
                "redirect_uris": [self.redirect_uri],
            },
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertTrue(data["client_id"])
        self.assertNotEqual(data["client_id"], "pawwake-gemini-client")
        self.assertEqual(data["client_name"], "Google Gemini")
        self.assertEqual(data["redirect_uris"], [self.redirect_uri])
        self.assertIn("authorization_code", data["grant_types"])

    # ============================================================
    # 3. 授权确认页与接口 (GATEWAY_SECRET 常量时间比对 + 一次性 code)
    # ============================================================

    def test_oauth_authorize_get(self):
        """测试 GET /oauth/authorize 渲染授权表单"""
        resp = self.client.get(
            "/oauth/authorize",
            params={
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "state": self.state,
                "code_challenge": self.challenge,
                "code_challenge_method": "S256",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("GATEWAY_SECRET", resp.text)
        self.assertIn(self.client_id, resp.text)

    def test_oauth_authorize_post_wrong_secret(self):
        """测试 POST /oauth/authorize 密码错误返回 401 且绝不签发 code"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            resp = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": "wrong-secret",
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "state": self.state,
                    "code_challenge": self.challenge,
                    "code_challenge_method": "S256",
                },
            )
            self.assertEqual(resp.status_code, 401)
            self.assertIn("校验失败", resp.text)

    def test_oauth_authorize_post_success(self):
        """测试 POST /oauth/authorize 密码正确返回 302 重定向并携带 code"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            resp = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "state": self.state,
                    "code_challenge": self.challenge,
                    "code_challenge_method": "S256",
                },
            )
            self.assertEqual(resp.status_code, 302)
            location = resp.headers["location"]
            self.assertTrue(location.startswith(self.redirect_uri))
            self.assertIn("code=", location)
            self.assertIn(f"state={self.state}", location)

    # ============================================================
    # 4. 令牌交换与刷新接口 (PKCE + 密钥校验)
    # ============================================================

    def _get_auth_code(self):
        """Helper: 获取有效的授权 code"""
        resp = self.client.post(
            "/oauth/authorize",
            data={
                "gateway_secret": self.secret,
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "state": self.state,
                "code_challenge": self.challenge,
                "code_challenge_method": "S256",
            },
        )
        location = resp.headers["location"]
        import urllib.parse
        parsed = urllib.parse.urlparse(location)
        q = dict(urllib.parse.parse_qsl(parsed.query))
        return q["code"]

    def test_oauth_token_pkce_mismatch(self):
        """测试 PKCE code_verifier 不匹配直接返回 400"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            code = self._get_auth_code()
            resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": "wrong-verifier-12345678901234567890",
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertEqual(resp.json()["error"], "invalid_grant")

    def test_oauth_token_success_and_replay_prevention(self):
        """测试 PKCE 成功换取派生令牌，且同一 code 再次使用被销毁防重放"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            code = self._get_auth_code()

            # 第一次交换成功
            resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": self.verifier,
                },
            )
            self.assertEqual(resp.status_code, 200)
            token_data = resp.json()
            self.assertTrue(token_data["access_token"].startswith("pwa_"))
            self.assertTrue(auth.verify_derived_token(token_data["access_token"]))
            self.assertEqual(token_data["token_type"], "Bearer")
            self.assertTrue(token_data["refresh_token"].startswith("pwr_"))
            self.assertEqual(token_data["expires_in"], 2592000)
            self.assertEqual(token_data["scope"], "mcp")

            # 重放该 code 应立即失败
            resp_replay = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": self.verifier,
                },
            )
            self.assertEqual(resp_replay.status_code, 400)

    def test_oauth_token_json_body_supported(self):
        """测试 /oauth/token 同样支持 application/json 请求体"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            code = self._get_auth_code()
            resp = self.client.post(
                "/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": self.verifier,
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["access_token"].startswith("pwa_"))
            self.assertTrue(auth.verify_derived_token(resp.json()["access_token"]))

    def test_oauth_token_client_secret_check(self):
        """测试携带非法 client_secret 时返回 401"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            code = self._get_auth_code()
            resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": "invalid-secret",
                    "code_verifier": self.verifier,
                },
            )
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(resp.json()["error"], "invalid_client")

    def test_oauth_token_refresh(self):
        """测试 refresh_token 刷新令牌与防伪造"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            code = self._get_auth_code()
            token_resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": self.verifier,
                },
            )
            refresh_token = token_resp.json()["refresh_token"]

            # 正常刷新
            refresh_resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            self.assertEqual(refresh_resp.status_code, 200)
            new_data = refresh_resp.json()
            self.assertTrue(new_data["access_token"].startswith("pwa_"))
            self.assertTrue(auth.verify_derived_token(new_data["access_token"]))
            self.assertTrue(new_data["refresh_token"].startswith("pwr_"))

            # 伪造 refresh token
            forged = "pwr_fake_random_bytes_invalid_signature"
            bad_resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": forged,
                },
            )
            self.assertEqual(bad_resp.status_code, 400)
            self.assertEqual(bad_resp.json()["error"], "invalid_grant")

    def test_oauth_revoke(self):
        """测试 POST /oauth/revoke 返回 200 OK {}"""
        resp = self.client.post("/oauth/revoke")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {})

    # ============================================================
    # 5. /mcp 门卫拦截与 WWW-Authenticate 响应头及防 307 重定向
    # ============================================================

    def test_mcp_unauthorized_returns_401_with_www_authenticate(self):
        """测试未带令牌访问 /mcp 与 /mcp/ 均返回 401 并附带标准 WWW-Authenticate 头"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            for path in ["/mcp", "/mcp/"]:
                resp = self.client.get(path, headers={"Host": "pawwake.example.com"})
                self.assertEqual(resp.status_code, 401)
                www_auth = resp.headers.get("www-authenticate", "")
                self.assertIn("Bearer", www_auth)
                self.assertIn(
                    'resource_metadata="https://pawwake.example.com/.well-known/oauth-protected-resource"',
                    www_auth,
                )

    def test_mcp_dual_path_passthrough_no_307(self):
        """测试 /mcp 和 /mcp/ 双路径直通，绝不发生 307 POST 重定向"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            resp1 = self.client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            )
            self.assertEqual(resp1.status_code, 401)
            self.assertNotEqual(resp1.status_code, 307)

            resp2 = self.client.post(
                "/mcp/",
                json={"jsonrpc": "2.0", "method": "ping", "id": 2},
            )
            self.assertEqual(resp2.status_code, 401)
            self.assertNotEqual(resp2.status_code, 307)


class TestMCPTools(unittest.TestCase):
    """测试 mcp_server 中新增的工具与防重逻辑"""

    def test_get_memory_summary(self):
        """测试 get_memory_summary 工具读取分区摘要"""
        with patch.object(shared, "DATABASE_ENABLED", True):
            with patch.object(shared, "CACHE_PARTITION_ENABLED", True):
                with patch("db.conversations.get_session_cache_state", new_callable=AsyncMock) as mock_state:
                    mock_state.return_value = {
                        "summary_parts": ["第1段摘要", "第2段摘要"],
                        "a_start_round": 10,
                        "updated_at": None,
                    }
                    res = asyncio.run(mcp_server.get_memory_summary(session_id="test-session"))
                    self.assertEqual(res["status"], "ok")
                    self.assertEqual(res["session_id"], "test-session")
                    self.assertEqual(res["summary"], "第1段摘要\n\n第2段摘要")
                    self.assertEqual(res["summary_count"], 2)
                    self.assertEqual(res["a_start_round"], 10)
                    self.assertNotIn("summary_parts", res)

    def test_recall_conversation_snippets(self):
        """测试 recall_conversation_snippets 工具召回历史对话切片"""
        with patch.object(shared, "DATABASE_ENABLED", True):
            with patch.object(shared, "CONVERSATION_RECALL_ENABLED", True):
                with patch("db.search.search_chat_fragments", new_callable=AsyncMock) as mock_frag:
                    mock_frag.return_value = (
                        [
                            {
                                "session_id": "sess-1",
                                "fragments": ["[2026-05-02] user: 你好啊\n[2026-05-02] assistant: 你好！"],
                            }
                        ],
                        1,
                    )
                    res = asyncio.run(mcp_server.recall_conversation_snippets(query="你好", limit=3))
                    self.assertEqual(res["status"], "ok")
                    self.assertEqual(res["count"], 1)
                    self.assertEqual(res["snippets"][0]["session_id"], "sess-1")
                    self.assertIn("你好", res["snippets"][0]["snippet"])

    def test_add_memory_duplicate_protection(self):
        """测试 add_memory 在未传 external_id 时的防重保护"""
        with patch.object(shared, "DATABASE_ENABLED", True):
            with patch.object(shared, "MEMORY_ENABLED", True):
                with patch("db.memories.check_duplicate_memory", new_callable=AsyncMock) as mock_dup:
                    mock_dup.return_value = {
                        "is_duplicate": True,
                        "reason": "exact",
                        "matched_id": 99,
                        "similarity": 1.0,
                    }
                    with patch("db.memories.get_memory_by_id", new_callable=AsyncMock) as mock_get:
                        mock_get.return_value = {
                            "id": 99,
                            "content": "已知记忆",
                            "importance": 5,
                            "title": None,
                            "layer": 1,
                            "event_date": None,
                            "external_id": None,
                            "is_active": True,
                            "is_pinned": False,
                        }
                        res = asyncio.run(mcp_server.add_memory(content="已知记忆"))
                        self.assertEqual(res["status"], "exists")
                        self.assertEqual(res["memory"]["id"], 99)
                        self.assertEqual(res["duplicate_reason"], "exact")


class TestReverseValidations(unittest.TestCase):
    """专门针对反向验证的用例：PKCE 不匹配、redirect_uri 不匹配、错误密钥、用 OAuth 令牌访问 /v1/models"""

    @classmethod
    def setUpClass(cls):
        os.environ["PUBLIC_BASE_URL"] = "https://pawwake.example.com"
        cls.client = TestClient(app, follow_redirects=False)

    def setUp(self):
        self.secret = "test-gateway-secret-12345"
        self.client_id = "pawwake-gemini-client"
        self.redirect_uri = "https://gemini.google.com/oauth/callback"
        self.state = "state-rev"
        self.verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        digest = hashlib.sha256(self.verifier.encode("ascii")).digest()
        self.challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

        oauth_mcp._authorize_failures.clear()
        oauth_mcp._registered_clients[self.client_id] = {
            "client_id": self.client_id,
            "client_name": "Google Gemini",
            "redirect_uris": [self.redirect_uri],
        }

    def _get_code(self):
        resp = self.client.post(
            "/oauth/authorize",
            data={
                "gateway_secret": self.secret,
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "state": self.state,
                "code_challenge": self.challenge,
                "code_challenge_method": "S256",
            },
        )
        import urllib.parse
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(resp.headers["location"]).query))
        return q["code"]

    def test_reverse_pkce(self):
        """反向验证 1：PKCE 正向匹配得 200，反向不匹配得 400"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 反向：错误 code_verifier -> 400
            code_neg = self._get_code()
            r_neg = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code_neg,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": "wrong-verifier",
                },
            )
            self.assertEqual(r_neg.status_code, 400)
            self.assertEqual(r_neg.json()["error"], "invalid_grant")

            # 正向：正确 code_verifier -> 200
            code_pos = self._get_code()
            r_pos = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code_pos,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": self.verifier,
                },
            )
            self.assertEqual(r_pos.status_code, 200)

    def test_reverse_redirect_uri(self):
        """反向验证 2：redirect_uri 正向一致得 200，反向不匹配得 400"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 反向：token 交换时 redirect_uri 不匹配 -> 400
            code_neg = self._get_code()
            r_neg = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code_neg,
                    "redirect_uri": "https://evil.com/callback",
                    "client_id": self.client_id,
                    "code_verifier": self.verifier,
                },
            )
            self.assertEqual(r_neg.status_code, 400)
            self.assertEqual(r_neg.json()["error"], "invalid_grant")

            # 正向：redirect_uri 匹配 -> 200
            code_pos = self._get_code()
            r_pos = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code_pos,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": self.verifier,
                },
            )
            self.assertEqual(r_pos.status_code, 200)

    def test_reverse_secret_validation(self):
        """反向验证 3：密钥正确得 302，反向错误密钥得 401"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            oauth_mcp._authorize_failures.clear()
            # 反向：错误密码 -> 401
            r_neg = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": "bad-password",
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "state": self.state,
                    "code_challenge": self.challenge,
                    "code_challenge_method": "S256",
                },
            )
            self.assertEqual(r_neg.status_code, 401)

            oauth_mcp._authorize_failures.clear()
            # 正向：正确密码 -> 302
            r_pos = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "state": self.state,
                    "code_challenge": self.challenge,
                    "code_challenge_method": "S256",
                },
            )
            self.assertEqual(r_pos.status_code, 302)

    def test_reverse_oauth_token_on_v1_models(self):
        """反向验证 4：X-Gateway-Key 访问 /v1/models 得 200，反向用 OAuth Bearer 访问 /v1/models 必须失败返回 401"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 正向：带 X-Gateway-Key 放行
            r_pos = self.client.get("/v1/models", headers={"X-Gateway-Key": self.secret})
            self.assertEqual(r_pos.status_code, 200)

            # 反向：带派生 OAuth Bearer 令牌访问 /v1/models 被拒
            token = auth.create_derived_token()
            r_neg = self.client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(r_neg.status_code, 401)
            self.assertEqual(r_neg.json(), {"error": "Unauthorized. Provide X-Gateway-Key header."})


# ============================================================
# 8 项新增核心测试用例
# ============================================================

class TestNewRequirements(unittest.TestCase):
    """测试用例新增覆盖：
    1. 后注册覆盖回调地址的攻击
    2. 未登记 client_id
    3. plain 被拒
    4. 伪造 Host 不影响元数据与未配置 PUBLIC_BASE_URL 返回 503
    5. 全局失败限流
    6. 派生令牌在 /v1 /api 得 401 在 /mcp 得 200
    7. 过期令牌与换 Key 后旧令牌失效
    8. add_memory 包含关系不覆盖
    """

    @classmethod
    def setUpClass(cls):
        os.environ["PUBLIC_BASE_URL"] = "https://pawwake.example.com"
        os.environ["OAUTH_ALLOWED_REDIRECT_HOSTS"] = (
            "gemini.google.com,victim.example.com,attacker.example.com,allowed.example.com"
        )
        cls.client = TestClient(app, follow_redirects=False)

    def setUp(self):
        self.secret = "test-gateway-secret-12345"
        self.redirect_uri = "https://gemini.google.com/oauth/callback"
        oauth_mcp._authorize_failures.clear()
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            self.client_id = oauth_mcp.create_stateless_client_id([self.redirect_uri])
        oauth_mcp._registered_clients[self.client_id] = {
            "client_id": self.client_id,
            "client_name": "Google Gemini",
            "redirect_uris": [self.redirect_uri],
        }

    def test_post_registration_overwrite_callback_attack(self):
        """1. 后注册覆盖回调地址的攻击：DCR 随机生成 client_id 且绝不覆盖已有登记"""
        # 受害者正常注册
        resp = self.client.post(
            "/oauth/register",
            json={
                "client_name": "Victim Client",
                "redirect_uris": ["https://victim.example.com/callback"],
            },
        )
        self.assertEqual(resp.status_code, 201)
        victim_client_id = resp.json()["client_id"]
        self.assertEqual(
            oauth_mcp._registered_clients[victim_client_id]["redirect_uris"],
            ["https://victim.example.com/callback"],
        )

        # 攻击者尝试带受害者的 client_id 重新注册并偷换回调地址
        resp_attack = self.client.post(
            "/oauth/register",
            json={
                "client_id": victim_client_id,
                "client_name": "Attacker Client",
                "redirect_uris": ["https://attacker.example.com/callback"],
            },
        )
        self.assertEqual(resp_attack.status_code, 400)
        self.assertEqual(
            oauth_mcp._registered_clients[victim_client_id]["redirect_uris"],
            ["https://victim.example.com/callback"],
        )

        # 攻击者即使不传 client_id 再次注册，也只能得到独立的新 client_id，受害者回调绝不被篡改
        resp_attack2 = self.client.post(
            "/oauth/register",
            json={
                "client_name": "Attacker Client 2",
                "redirect_uris": ["https://attacker.example.com/callback"],
            },
        )
        self.assertEqual(resp_attack2.status_code, 201)
        attacker_client_id = resp_attack2.json()["client_id"]
        self.assertNotEqual(attacker_client_id, victim_client_id)
        self.assertEqual(
            oauth_mcp._registered_clients[victim_client_id]["redirect_uris"],
            ["https://victim.example.com/callback"],
        )

    def test_unregistered_client_id_rejected(self):
        """2. 未登记 client_id：/oauth/authorize（GET 与 POST）和 /oauth/token 均拒绝"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            bad_id = "unregistered-client-id-xyz"

            # GET /oauth/authorize 拒绝
            resp_get = self.client.get(
                "/oauth/authorize",
                params={
                    "client_id": bad_id,
                    "redirect_uri": self.redirect_uri,
                    "response_type": "code",
                },
            )
            self.assertEqual(resp_get.status_code, 400)
            self.assertEqual(resp_get.json()["error"], "unauthorized_client")

            # POST /oauth/authorize 拒绝
            resp_post = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": bad_id,
                    "redirect_uri": self.redirect_uri,
                },
            )
            self.assertEqual(resp_post.status_code, 400)
            self.assertEqual(resp_post.json()["error"], "unauthorized_client")

            # POST /oauth/token 拒绝未登记 client_id
            resp_token = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": "non-existent-code",
                    "redirect_uri": self.redirect_uri,
                    "client_id": bad_id,
                    "code_verifier": "verifier",
                },
            )
            self.assertEqual(resp_token.status_code, 400)

    def test_plain_code_challenge_rejected(self):
        """3. plain 被拒：元数据去掉 plain，实际校验只留 S256，请求 plain 直接被拒"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 检查 Authorization Server 元数据
            resp_meta = self.client.get("/.well-known/oauth-authorization-server")
            self.assertEqual(resp_meta.status_code, 200)
            self.assertEqual(resp_meta.json()["code_challenge_methods_supported"], ["S256"])
            self.assertNotIn("plain", resp_meta.json()["code_challenge_methods_supported"])

            # GET /oauth/authorize 使用 plain 被拒
            resp_get = self.client.get(
                "/oauth/authorize",
                params={
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "response_type": "code",
                    "code_challenge": "plain-challenge",
                    "code_challenge_method": "plain",
                },
            )
            self.assertEqual(resp_get.status_code, 400)

            # POST /oauth/authorize 使用 plain 被拒
            resp_post = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "code_challenge": "plain-challenge",
                    "code_challenge_method": "plain",
                },
            )
            self.assertEqual(resp_post.status_code, 400)

    def test_forged_host_ignored_and_missing_public_base_url_503(self):
        """4. 伪造 Host 不影响元数据，且未设置 PUBLIC_BASE_URL 时 OAuth 端点返回 503"""
        # 伪造 Host 和 X-Forwarded-Host 头，元数据地址始终严格读取 PUBLIC_BASE_URL
        forged_headers = {
            "Host": "evil.attacker.com",
            "X-Forwarded-Host": "evil.attacker.com",
        }
        resp = self.client.get("/.well-known/oauth-authorization-server", headers=forged_headers)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["issuer"], "https://pawwake.example.com")
        self.assertNotIn("evil.attacker.com", resp.text)

        resp_res = self.client.get("/.well-known/oauth-protected-resource", headers=forged_headers)
        self.assertEqual(resp_res.status_code, 200)
        self.assertEqual(resp_res.json()["resource"], "https://pawwake.example.com/mcp/")
        self.assertNotIn("evil.attacker.com", resp_res.text)

        # 未设置 PUBLIC_BASE_URL 时，全部 OAuth 端点返回 503
        with patch.dict(os.environ, {"PUBLIC_BASE_URL": ""}):
            self.assertEqual(
                self.client.get("/.well-known/oauth-authorization-server").status_code, 503
            )
            self.assertEqual(
                self.client.get("/.well-known/oauth-protected-resource").status_code, 503
            )
            self.assertEqual(
                self.client.post("/oauth/register", json={"redirect_uris": ["https://foo.com/cb"]}).status_code, 503
            )
            self.assertEqual(
                self.client.get(f"/oauth/authorize?client_id={self.client_id}&redirect_uri={self.redirect_uri}").status_code, 503
            )
            self.assertEqual(
                self.client.post("/oauth/authorize", data={"client_id": self.client_id}).status_code, 503
            )
            self.assertEqual(
                self.client.post("/oauth/token", data={"grant_type": "authorization_code"}).status_code, 503
            )
            self.assertEqual(
                self.client.post("/oauth/revoke").status_code, 503
            )

    def test_authorize_rate_limiting(self):
        """5. POST /oauth/authorize 全局失败限流：每分钟最多 5 次失败，第 6 次返回 429"""
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            oauth_mcp._authorize_failures.clear()

            # 前 5 次失败（错误密码返回 401）
            for i in range(5):
                resp = self.client.post(
                    "/oauth/authorize",
                    data={
                        "gateway_secret": f"wrong-pass-{i}",
                        "client_id": self.client_id,
                        "redirect_uri": self.redirect_uri,
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    },
                )
                self.assertEqual(resp.status_code, 401)

            # 第 6 次超过阈值，返回 429
            resp_6 = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                },
            )
            self.assertEqual(resp_6.status_code, 429)
            self.assertEqual(resp_6.json()["error"], "slow_down")

    def test_derived_token_access_control(self):
        """6. 派生令牌在 /v1 /api 返回 401，在 /mcp 返回 200；主密钥在 /mcp 老方式可用"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret), \
             patch.object(shared, "DASHBOARD_PASSWORD", "test-dashboard-password"), \
             patch.object(shared, "SESSION_SECRET", "s" * 32):
            token = auth.create_derived_token()

            # /v1 端点拒绝派生令牌
            resp_v1 = self.client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(resp_v1.status_code, 401)

            # /api 端点拒绝派生令牌
            resp_api = self.client.get("/api/memories", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(resp_api.status_code, 401)

            # /mcp 路径接受派生令牌（返回 200）
            with mcp_test_client() as lifespan_client:
                resp_mcp = lifespan_client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                )
                self.assertEqual(resp_mcp.status_code, 200)

                # 主密钥直接用于 /mcp 的老方式保持可用
                resp_old = lifespan_client.post(
                    "/mcp",
                    headers={"X-Gateway-Key": self.secret},
                    json={"jsonrpc": "2.0", "method": "ping", "id": 2},
                )
                self.assertEqual(resp_old.status_code, 200)

    def test_expired_and_rotated_tokens_rejected(self):
        """7. 过期令牌被拒；换 Key 后旧令牌自动失效"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 生成已过期的派生令牌 (expires_in = -10 秒)
            expired_token = auth.create_derived_token(expires_in=-10)
            self.assertFalse(auth.verify_derived_token(expired_token))

            with mcp_test_client() as lifespan_client:
                resp_exp = lifespan_client.post(
                    "/mcp",
                    headers={"Authorization": f"Bearer {expired_token}"},
                    json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                )
                self.assertEqual(resp_exp.status_code, 401)

            # 测试轮换密钥：旧密钥签发的令牌在换 Key 后失效
            valid_token = auth.create_derived_token(expires_in=3600)
            self.assertTrue(auth.verify_derived_token(valid_token))

            # 轮换密钥为 new-secret-key
            with patch.object(shared, "GATEWAY_SECRET", "new-secret-key"):
                self.assertFalse(auth.verify_derived_token(valid_token))
                with mcp_test_client() as lifespan_client:
                    resp_rotated = lifespan_client.post(
                        "/mcp",
                        headers={"Authorization": f"Bearer {valid_token}"},
                        json={"jsonrpc": "2.0", "method": "ping", "id": 2},
                    )
                    self.assertEqual(resp_rotated.status_code, 401)

    def test_add_memory_containment_does_not_overwrite(self):
        """8. add_memory 包含关系不覆盖：去掉包含关系的覆盖与丢弃，新记忆正常创建"""
        with patch.object(shared, "DATABASE_ENABLED", True):
            with patch.object(shared, "MEMORY_ENABLED", True):
                # 验证底层 check_duplicate_memory 对包含关系不再判定为 duplicate
                mock_conn = AsyncMock()
                mock_conn.fetch.return_value = [{"id": 10, "content": "小明在北京"}]
                mock_pool = MagicMock()
                mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

                with patch("db.core.get_pool", return_value=mock_pool):
                    # 包含旧内容的新内容
                    res_sup = asyncio.run(db_memories.check_duplicate_memory("小明在北京买了苹果"))
                    self.assertFalse(res_sup["is_duplicate"])

                    # 被旧内容包含的新内容
                    res_sub = asyncio.run(db_memories.check_duplicate_memory("北京"))
                    self.assertFalse(res_sub["is_duplicate"])

                # 验证 mcp_server.add_memory 对包含关系不会调用 update 而是直接 save 创建
                with patch("db.memories.check_duplicate_memory", new_callable=AsyncMock) as mock_dup:
                    mock_dup.return_value = {"is_duplicate": False, "reason": None, "matched_id": None}
                    with patch("db.memories.save_memory", new_callable=AsyncMock) as mock_save:
                        mock_save.return_value = 101
                        res = asyncio.run(mcp_server.add_memory(content="小明在北京买了苹果"))
                        self.assertEqual(res["status"], "created")
                        self.assertEqual(res["memory"]["id"], 101)
                        self.assertNotEqual(res["status"], "updated")


class TestOAuthRound2Requirements(unittest.TestCase):
    """涵盖任务第二阶段所有新要求的测试用例：
    1. 畸形令牌（含 é）在 /mcp 和 /oauth/token 得 401/400 而非 500
    2. 名单外与空名单被拒（403 且打印指定日志）
    3. 清空内存登记表后刷新仍成功
    4. 缺 code_challenge 被拒（GET 与 POST）
    5. 过期刷新令牌被拒
    6. Dashboard 段 git diff origin/main..HEAD -- auth.py 在该段为空
    7. 授权页显示"授权码将发送到：<redirect_uri 的主机>"，去掉写死的 "Google Gemini"
    """

    @classmethod
    def setUpClass(cls):
        os.environ["PUBLIC_BASE_URL"] = "https://pawwake.example.com"
        os.environ["OAUTH_ALLOWED_REDIRECT_HOSTS"] = "allowed.example.com,gemini.google.com"
        cls.client = TestClient(app, follow_redirects=False)

    def setUp(self):
        self.secret = "test-gateway-secret-round2"
        self.redirect_uri = "https://allowed.example.com/oauth/callback"
        oauth_mcp._authorize_failures.clear()
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            self.client_id = oauth_mcp.create_stateless_client_id([self.redirect_uri])
        oauth_mcp._registered_clients[self.client_id] = {
            "client_id": self.client_id,
            "client_name": "Test Client",
            "redirect_uris": [self.redirect_uri],
        }

    def test_malformed_token_returns_401_or_400_not_500(self):
        """1. 畸形令牌（含 é）在 /mcp 和 /oauth/token 得 401/400 而非 500，且生产校验函数不抛异常"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 生产校验函数调用测试：畸形令牌（含非 ASCII）均返回 False，绝不抛异常
            malformed_tokens = [
                "é",
                "pwa_malformed_é",
                "pwa_notanint_signature",
                "pwa_12345_wrongsig_é",
                "",
                None,
                12345,
            ]
            for t in malformed_tokens:
                self.assertFalse(auth.verify_derived_token(t))

            malformed_refresh = [
                "é",
                "pwr_malformed_é",
                "pwr_notanint_raw_sig",
                "pwr_12345_raw_badsig_é",
                "",
                None,
                12345,
            ]
            for r in malformed_refresh:
                self.assertFalse(oauth_mcp.verify_refresh_token(r))

            malformed_client_ids = [
                "é",
                "pwc_malformed_é",
                "pwc_badbase64_sig",
                "",
                None,
            ]
            for c in malformed_client_ids:
                is_valid, uris = oauth_mcp.verify_client_id(c)
                self.assertFalse(is_valid)
                self.assertEqual(uris, [])

            # HTTP 路径调用测试：/mcp 接受畸形令牌（含 é）返回 401，绝不抛 500
            with mcp_test_client() as lifespan_client:
                resp_mcp1 = lifespan_client.post(
                    "/mcp",
                    headers=[(b"authorization", b"Bearer pwa_123_invalid_\xc3\xa9")],
                    json={"jsonrpc": "2.0", "method": "ping", "id": 1},
                )
                self.assertEqual(resp_mcp1.status_code, 401)

                resp_mcp2 = lifespan_client.post(
                    "/mcp",
                    headers=[(b"authorization", b"Bearer \xc3\xa9")],
                    json={"jsonrpc": "2.0", "method": "ping", "id": 2},
                )
                self.assertEqual(resp_mcp2.status_code, 401)

                resp_mcp3 = lifespan_client.post(
                    "/mcp",
                    headers=[(b"x-gateway-key", b"key_\xc3\xa9")],
                    json={"jsonrpc": "2.0", "method": "ping", "id": 3},
                )
                self.assertEqual(resp_mcp3.status_code, 401)

            # HTTP 路径调用测试：/oauth/token 接受畸形 refresh_token（含 é）返回 400，绝不抛 500
            resp_tok = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": "pwr_123_raw_é",
                },
            )
            self.assertEqual(resp_tok.status_code, 400)
            self.assertEqual(resp_tok.json()["error"], "invalid_grant")

    def test_redirect_hosts_whitelist_and_empty_rejected(self):
        """2. 名单外与空名单被拒（403 且打印指定日志）"""
        import io

        # 名单外被拒
        with patch.dict(os.environ, {"OAUTH_ALLOWED_REDIRECT_HOSTS": "allowed.example.com"}):
            stdout_buf = io.StringIO()
            with patch("sys.stdout", stdout_buf):
                resp = self.client.post(
                    "/oauth/register",
                    json={"redirect_uris": ["https://evil.unauthorized.com/callback"]},
                )
            self.assertEqual(resp.status_code, 403)
            self.assertIn("🚫 OAuth 注册被拒 host=evil.unauthorized.com", stdout_buf.getvalue())

        # 空名单被拒
        with patch.dict(os.environ, {"OAUTH_ALLOWED_REDIRECT_HOSTS": ""}):
            stdout_buf = io.StringIO()
            with patch("sys.stdout", stdout_buf):
                resp = self.client.post(
                    "/oauth/register",
                    json={"redirect_uris": ["https://allowed.example.com/callback"]},
                )
            self.assertEqual(resp.status_code, 403)
            self.assertIn("🚫 OAuth 注册被拒 host=allowed.example.com", stdout_buf.getvalue())

    def test_refresh_token_succeeds_after_memory_cleared(self):
        """3. 清空内存登记表后刷新仍成功"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 获取有效授权码并换取令牌
            verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
            digest = hashlib.sha256(verifier.encode("ascii")).digest()
            challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

            resp_auth = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                },
            )
            self.assertEqual(resp_auth.status_code, 302)
            parsed = urllib.parse.urlparse(resp_auth.headers["location"])
            code = dict(urllib.parse.parse_qsl(parsed.query))["code"]

            token_resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": verifier,
                },
            )
            self.assertEqual(token_resp.status_code, 200)
            refresh_token = token_resp.json()["refresh_token"]

            # 清空内存登记表
            oauth_mcp._registered_clients.clear()
            self.assertEqual(len(oauth_mcp._registered_clients), 0)

            # 刷新令牌依然成功（带 stateless client_id）
            ref_resp1 = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.client_id,
                },
            )
            self.assertEqual(ref_resp1.status_code, 200)
            new_refresh = ref_resp1.json()["refresh_token"]

            # 不带 client_id 刷新也依然成功
            ref_resp2 = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": new_refresh,
                },
            )
            self.assertEqual(ref_resp2.status_code, 200)

    def test_missing_or_invalid_code_challenge_rejected(self):
        """4. 缺 code_challenge 或非 S256 被拒（GET 与 POST 均返回 400）"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # GET 缺 code_challenge
            r_get1 = self.client.get(
                "/oauth/authorize",
                params={
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "response_type": "code",
                },
            )
            self.assertEqual(r_get1.status_code, 400)
            self.assertEqual(r_get1.json()["error"], "invalid_request")

            # GET 方法为 plain
            r_get2 = self.client.get(
                "/oauth/authorize",
                params={
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "response_type": "code",
                    "code_challenge": "challenge",
                    "code_challenge_method": "plain",
                },
            )
            self.assertEqual(r_get2.status_code, 400)

            # POST 缺 code_challenge
            r_post1 = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                },
            )
            self.assertEqual(r_post1.status_code, 400)

            # POST 方法为 plain
            r_post2 = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "code_challenge": "challenge",
                    "code_challenge_method": "plain",
                },
            )
            self.assertEqual(r_post2.status_code, 400)

    def test_expired_refresh_token_rejected(self):
        """5. 过期刷新令牌被拒"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            # 生成 10 秒前已过期的刷新令牌
            expired = oauth_mcp.create_refresh_token(expires_in=-10)
            self.assertFalse(oauth_mcp.verify_refresh_token(expired))

            # 请求 /oauth/token 交换
            resp = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": expired,
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertEqual(resp.json()["error"], "invalid_grant")

    def test_dashboard_auth_diff_against_origin_main_is_empty(self):
        """6. Dashboard 段 git diff origin/main..HEAD -- auth.py 在该段为空"""
        import subprocess
        diff = subprocess.check_output(
            ["git", "diff", "origin/main", "--", "auth.py"],
            text=True,
        )
        self.assertNotIn("is_dashboard_path", diff)
        self.assertNotIn("Dashboard login", diff)

    def test_authorize_page_shows_redirect_host_and_removes_hardcoded_gemini(self):
        """7. 授权页显示"授权码将发送到：<redirect_uri 的主机>"，去掉写死的 "Google Gemini" """
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        resp = self.client.get(
            "/oauth/authorize",
            params={
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("授权码将发送到：allowed.example.com", resp.text)
        self.assertNotIn("Google Gemini 正在请求连接", resp.text)

    def test_code_verifier_with_non_ascii_returns_400_not_500(self):
        """8. code_verifier 含 é 在 /oauth/token 交换时得 400 而非 500"""
        with patch.object(shared, "GATEWAY_SECRET", self.secret):
            verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
            digest = hashlib.sha256(verifier.encode("ascii")).digest()
            challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

            resp_auth = self.client.post(
                "/oauth/authorize",
                data={
                    "gateway_secret": self.secret,
                    "client_id": self.client_id,
                    "redirect_uri": self.redirect_uri,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                },
            )
            self.assertEqual(resp_auth.status_code, 302)
            parsed = urllib.parse.urlparse(resp_auth.headers["location"])
            code = dict(urllib.parse.parse_qsl(parsed.query))["code"]

            # 使用包含非 ASCII 字符 é 的 code_verifier 进行换票
            resp_tok = self.client.post(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "code_verifier": "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk_é",
                },
            )
            self.assertEqual(resp_tok.status_code, 400)
            self.assertEqual(resp_tok.json()["error"], "invalid_grant")


if __name__ == "__main__":
    unittest.main()
