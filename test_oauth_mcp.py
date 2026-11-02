"""
test_oauth_mcp.py: 针对 OAuth 2.1 握手层、/mcp/ 鉴权保护与拦截、PKCE 及 MCP 工具升级的单元测试
"""

import asyncio
import base64
import hashlib
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import auth
import mcp_server
import routes.oauth_mcp as oauth_mcp
import shared
from main import app


class TestOAuthMCPHandshake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, follow_redirects=False)

    def setUp(self):
        self.secret = "test-gateway-secret-12345"
        self.client_id = "pawwake-gemini-client"
        self.redirect_uri = "https://gemini.google.com/oauth/callback"
        self.state = "test-state-abc"
        # PKCE S256 pair
        self.verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        digest = hashlib.sha256(self.verifier.encode("ascii")).digest()
        self.challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

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
            self.assertIn("S256", data["code_challenge_methods_supported"])
            self.assertIn("mcp", data["scopes_supported"])

    # ============================================================
    # 2. 动态客户端注册 (DCR, RFC 7591)
    # ============================================================

    def test_dynamic_client_registration(self):
        """测试 POST /oauth/register 返回 201 Created 与已注册 client_id"""
        resp = self.client.post(
            "/oauth/register",
            json={
                "client_name": "Google Gemini",
                "redirect_uris": [self.redirect_uri],
            },
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["client_id"], "pawwake-gemini-client")
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
        """测试 PKCE 成功换取令牌，且同一 code 再次使用被销毁防重放"""
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
            self.assertEqual(token_data["access_token"], self.secret)
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
            self.assertEqual(resp.json()["access_token"], self.secret)

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
            self.assertEqual(new_data["access_token"], self.secret)
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
            # 未授权请求访问 /mcp 不应该发生 307，而是直接 401
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


if __name__ == "__main__":
    unittest.main()
