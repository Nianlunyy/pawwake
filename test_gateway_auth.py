"""
test_gateway_auth.py: 验证网关鉴权多 Key 支持、未设置密钥一律 401 及 scope['path'] 防回归
"""

import asyncio
import unittest
from unittest.mock import patch

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import URL

import auth
import shared


def _make_request(path: str, method: str = "GET", headers: dict = None, cookies: dict = None) -> Request:
    header_list = []
    if headers:
        for k, v in headers.items():
            header_list.append((k.lower().encode("latin1"), v.encode("latin1")))
    cookie_str = ""
    if cookies:
        cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
        header_list.append((b"cookie", cookie_str.encode("latin1")))

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "headers": header_list,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    return Request(scope)


class MismatchedRequest(Request):
    """用于测试 ⑤：故意让 url.path 与 scope['path'] 不一致的假请求对象"""
    def __init__(self, scope_path: str, url_path: str, method: str = "GET", headers: dict = None):
        header_list = []
        if headers:
            for k, v in headers.items():
                header_list.append((k.lower().encode("latin1"), v.encode("latin1")))
        scope = {
            "type": "http",
            "method": method,
            "path": scope_path,
            "raw_path": scope_path.encode("ascii"),
            "headers": header_list,
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "scheme": "http",
            "http_version": "1.1",
        }
        super().__init__(scope)
        self._custom_url = URL(f"http://testserver{url_path}")

    @property
    def url(self) -> URL:
        return self._custom_url


async def _dummy_call_next(request: Request):
    return JSONResponse(status_code=200, content={"status": "ok"})


class TestGatewayAuth(unittest.TestCase):
    def test_case_1_unset_empty_whitespace_commas_rejected(self):
        """① 未设置 / 空 / 空白 / 仅逗号 → 保护路径 401"""
        empty_secrets = ["", "   ", "\t  \n", ",", ",,,", " , ,  , "]
        for sec in empty_secrets:
            with patch.object(shared, "GATEWAY_SECRET", sec):
                # 保护路径应该返回 401
                req = _make_request("/v1/models")
                resp = asyncio.run(auth.gateway_auth_middleware(req, _dummy_call_next))
                self.assertEqual(resp.status_code, 401, f"GATEWAY_SECRET={sec!r} 应拒绝保护路径")
                self.assertEqual(resp.body, b'{"error":"Unauthorized. Provide X-Gateway-Key header."}')

                # 带任意 header 也应该返回 401
                req_with_key = _make_request("/v1/models", headers={"X-Gateway-Key": "some-random-key"})
                resp_with_key = asyncio.run(auth.gateway_auth_middleware(req_with_key, _dummy_call_next))
                self.assertEqual(resp_with_key.status_code, 401)

                # 公开路径仍应放行（200）
                for public_path in ["/", "/static/app.js", "/favicon.ico", "/dashboard/login"]:
                    pub_req = _make_request(public_path)
                    pub_resp = asyncio.run(auth.gateway_auth_middleware(pub_req, _dummy_call_next))
                    self.assertEqual(pub_resp.status_code, 200, f"公开路径 {public_path} 应放行")

                # OPTIONS 请求应放行
                opt_req = _make_request("/v1/models", method="OPTIONS")
                opt_resp = asyncio.run(auth.gateway_auth_middleware(opt_req, _dummy_call_next))
                self.assertEqual(opt_resp.status_code, 200)

    def test_case_2_single_key_pass_wrong_key_401(self):
        """② 单 Key 通过、错 Key 401"""
        with patch.object(shared, "GATEWAY_SECRET", "secret-test-key-1"):
            # 正确 Key 通过
            req_ok = _make_request("/v1/models", headers={"X-Gateway-Key": "secret-test-key-1"})
            resp_ok = asyncio.run(auth.gateway_auth_middleware(req_ok, _dummy_call_next))
            self.assertEqual(resp_ok.status_code, 200)

            # 错误 Key 401
            req_wrong = _make_request("/v1/models", headers={"X-Gateway-Key": "wrong-key"})
            resp_wrong = asyncio.run(auth.gateway_auth_middleware(req_wrong, _dummy_call_next))
            self.assertEqual(resp_wrong.status_code, 401)
            self.assertEqual(resp_wrong.body, b'{"error":"Unauthorized. Provide X-Gateway-Key header."}')

            # 缺少 Header 401
            req_none = _make_request("/v1/models")
            resp_none = asyncio.run(auth.gateway_auth_middleware(req_none, _dummy_call_next))
            self.assertEqual(resp_none.status_code, 401)

    def test_case_3_two_keys_pass_third_401(self):
        """③ 'a,b' 两个 Key 都通过、第三个 401"""
        with patch.object(shared, "GATEWAY_SECRET", "key-alpha,key-beta"):
            # Key A 通过
            req_a = _make_request("/v1/chat/completions", headers={"X-Gateway-Key": "key-alpha"})
            resp_a = asyncio.run(auth.gateway_auth_middleware(req_a, _dummy_call_next))
            self.assertEqual(resp_a.status_code, 200)

            # Key B 通过
            req_b = _make_request("/v1/chat/completions", headers={"X-Gateway-Key": "key-beta"})
            resp_b = asyncio.run(auth.gateway_auth_middleware(req_b, _dummy_call_next))
            self.assertEqual(resp_b.status_code, 200)

            # Key C 401
            req_c = _make_request("/v1/chat/completions", headers={"X-Gateway-Key": "key-gamma"})
            resp_c = asyncio.run(auth.gateway_auth_middleware(req_c, _dummy_call_next))
            self.assertEqual(resp_c.status_code, 401)
            self.assertEqual(resp_c.body, b'{"error":"Unauthorized. Provide X-Gateway-Key header."}')

    def test_case_4_empty_items_ignored_empty_header_401(self):
        """④ 'a,,b' 空项被忽略、空请求头 401"""
        with patch.object(shared, "GATEWAY_SECRET", "  key-first , , , key-second ,  "):
            # Key 1 通过
            req_1 = _make_request("/v1/models", headers={"X-Gateway-Key": "key-first"})
            resp_1 = asyncio.run(auth.gateway_auth_middleware(req_1, _dummy_call_next))
            self.assertEqual(resp_1.status_code, 200)

            # Key 2 通过
            req_2 = _make_request("/v1/models", headers={"X-Gateway-Key": "key-second"})
            resp_2 = asyncio.run(auth.gateway_auth_middleware(req_2, _dummy_call_next))
            self.assertEqual(resp_2.status_code, 200)

            # 空字符串请求头 401（不能与空项匹配）
            req_empty_header = _make_request("/v1/models", headers={"X-Gateway-Key": ""})
            resp_empty_header = asyncio.run(auth.gateway_auth_middleware(req_empty_header, _dummy_call_next))
            self.assertEqual(resp_empty_header.status_code, 401)

            # 纯空格请求头 401
            req_space_header = _make_request("/v1/models", headers={"X-Gateway-Key": "   "})
            resp_space_header = asyncio.run(auth.gateway_auth_middleware(req_space_header, _dummy_call_next))
            self.assertEqual(resp_space_header.status_code, 401)

    def test_case_5_scope_path_vs_url_path_mismatch_prevents_regression(self):
        """⑤ 用一个 url.path 与 scope['path'] 故意不一致的假请求对象，断言鉴权按 scope['path'] 判断"""
        with patch.object(shared, "GATEWAY_SECRET", "test-key-5"):
            # 假请求 A: scope['path'] 是受保护路径 /v1/models，url.path 故意伪装为公开路径 /
            # 若代码用 scope['path'] 则返回 401；若被改回 request.url.path 则会误判为公开路径放行 (200)
            req_fake_protected = MismatchedRequest(scope_path="/v1/models", url_path="/")
            resp_fake_protected = asyncio.run(auth.gateway_auth_middleware(req_fake_protected, _dummy_call_next))
            self.assertEqual(
                resp_fake_protected.status_code,
                401,
                "当 scope['path'] 是保护路径而 url.path 是公开路径时，必须按 scope['path'] 判断并返回 401！",
            )

            # 假请求 B: scope['path'] 是公开路径 /，url.path 是 /v1/models
            # 若代码用 scope['path'] 则放行 (200)；若用 url.path 则会误拦截 (401)
            req_fake_public = MismatchedRequest(scope_path="/", url_path="/v1/models")
            resp_fake_public = asyncio.run(auth.gateway_auth_middleware(req_fake_public, _dummy_call_next))
            self.assertEqual(
                resp_fake_public.status_code,
                200,
                "当 scope['path'] 是公开路径而 url.path 是保护路径时，必须按 scope['path'] 判断并放行！",
            )

    def test_case_6_memories_post_requires_gateway_key(self):
        """/api/memories POST 显式记忆写入只认 X-Gateway-Key，不认 Cookie"""
        with patch.object(shared, "GATEWAY_SECRET", "key-mem-1, key-mem-2"):
            # 正确 Key 1 通过
            req_ok1 = _make_request("/api/memories", method="POST", headers={"X-Gateway-Key": "key-mem-1"})
            resp_ok1 = asyncio.run(auth.gateway_auth_middleware(req_ok1, _dummy_call_next))
            self.assertEqual(resp_ok1.status_code, 200)

            # 正确 Key 2 通过
            req_ok2 = _make_request("/api/memories", method="POST", headers={"X-Gateway-Key": "key-mem-2"})
            resp_ok2 = asyncio.run(auth.gateway_auth_middleware(req_ok2, _dummy_call_next))
            self.assertEqual(resp_ok2.status_code, 200)

            # 错 Key 401
            req_wrong = _make_request("/api/memories", method="POST", headers={"X-Gateway-Key": "wrong-key"})
            resp_wrong = asyncio.run(auth.gateway_auth_middleware(req_wrong, _dummy_call_next))
            self.assertEqual(resp_wrong.status_code, 401)
            self.assertEqual(resp_wrong.body, b'{"error":"Unauthorized. Provide X-Gateway-Key header."}')

            # 未提供 Key 401
            req_none = _make_request("/api/memories", method="POST")
            resp_none = asyncio.run(auth.gateway_auth_middleware(req_none, _dummy_call_next))
            self.assertEqual(resp_none.status_code, 401)


if __name__ == "__main__":
    unittest.main()
