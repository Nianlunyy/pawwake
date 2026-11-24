"""Unit and integration tests for pure-Python Xiaohongshu note reader."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from starlette.requests import Request

import mcp_server
import shared
import xhs_client
from routes import dashboard as dashboard_routes


ROOT = Path(__file__).resolve().parent


class _FixedRNG:
    def __init__(self, vals):
        self.vals = vals
        self.idx = 0

    def randint(self, a: int, b: int) -> int:
        val = self.vals[self.idx % len(self.vals)]
        self.idx += 1
        return a + (val % (b - a + 1))

    def randbytes(self, n: int) -> bytes:
        return b"\x11" * n


class XhsSignatureGoldenTests(unittest.TestCase):
    """Verify byte-for-byte parity with xhs_lite_worker.js golden vectors."""

    def test_signature_vectors_match_js_worker(self):
        rng = _FixedRNG(
            [
                0x12345678,
                25,
                33,
                1111,
                0x89ABCDEF,
                17,
                42,
                1088,
                0x0F1E2D3C,
                12,
                20,
                1024,
            ]
        )
        fp = {
            "x33": "0",
            "x34": "0",
            "x35": "0",
            "x36": "7",
            "x37": "0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|1|0|0|0|0|0",
            "x38": "0|0|1|0|1|0|0|0|0|0|1|0|1|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0",
            "x39": 0,
            "x42": "3.4.4",
            "x43": "0123456789abcdef0123456789abcdef",
            "x44": "1700000000000",
            "x45": "__SEC_CAV__1-1-1-1-1|__SEC_WSA__|",
            "x46": "false",
            "x48": "",
            "x49": "{list:[],type:}",
            "x50": "",
            "x51": "",
            "x52": "",
            "x82": "_0x17a2|_0x1954",
        }
        a1 = "1900123456abcdef7890abcdef1234567890abcdef500012345"
        ts_sec = 1700000000.123
        post_payload = {
            "source_note_id": "65a1b2c3000000001e02a3b4",
            "image_formats": ["jpg", "webp", "avif"],
            "extra": {"need_body_topic": "1"},
            "xsec_source": "pc_feed",
            "xsec_token": "AB123_token",
        }
        get_params = {
            "note_id": "65a1b2c3000000001e02a3b4",
            "cursor": "",
            "top_comment_id": "",
            "image_formats": "jpg,webp,avif",
            "xsec_token": "AB123_token",
        }

        expected_b1 = (
            "I38rHdgsjopgIvesdVwgIC+oIELmBZ5e3VwXLgFTIxS3bqwErFeexd0ekncAzMFYnqthIhJeDBMDKutRI3KsYorWHPtGrbV0P9WfIi/eWc6eYqtyQApPI37ekmR6QL+5Ii6sdnoeSfqYHqwl2qt5B0DoIx+PGDi/sVtkIxdsxuwr4qtiIhuaIE3e3LV0I3VTIC7e0utl2ADmsLveDSKsSPw5IEvsiVtJOqw8BuwfPpdeTFWOIx4TIiu6ZPwrPut5IvlaLbgs3qtxIxes1VwHIkumIkIyejgsY/WTge7eSqte/D7sDcpipedeYrDtIC6eDVw2IENsSqtlnlSuNjVtIvoekqt3cZ7sVo4gIESyIEle+AFDI3EPKI8BIiWIZPwAIvGj4sesYINsxVwSIC7edBve00EJIi0e1SL5qPtlIhKe3/0e1peeVPtwLUNeYqwvIvSgaAes0VtkmeoeYqt/IxltIxZSouwwgVwpssDcJbos0josiutbIkOeSUvs1MFvIiAsfI/sxBDdIkve3PwHIvln/PtCcPwmIhOsfPwQIv5sVVwIIEKsYPt9+PwKrqw7I3kLIhbMrYMwBBgsfPwKIxzLIk0ejqwonYQ2caesDoG5IEYbQPtKoqw3I3OexPtiQL5eicAs3phwIhesk9Os3utsBqtaIx7sjutXIi7sdpZyI3osdUJsYqtRICNsxuwU4eQNIEZ+Iv7sxM6ex7vsYDAsSutzIkL1IxkYrVw4IC+UIhgskFq3+A/eWutHIEp2eut1mZeedMAeWVtnePt2IiM9IhhQnuw48VtbIxZ="
        )
        expected_xs_common = (
            "2UQAPsHC+aIjqArjwjHjNsQhPsHCH0rjNsQhPaHCH0c1Pjh9HjIj2eHjwjQgynEDJ74AHjIj2ePjwjQhyoPTqBPT49pjHjIj2ecjwjHFN0W9N0ZjNsQh+aHCH0rEPeZlP0PF+/8YGf+D8nG7weDIGnQ08BpfP/HA+eL9+AWEPBbjG9zS80LIPeZlP0PF+aHVHdW9H0ijHjIj2eqjwjHjNsQhwsHCHDDAwoQH8B4AyfRI8FS98g+Dpd4daLP3JFSb/BMsn0pSPM87nrldzSzQ2bPAGdb7zgQB8nph8emSy9E0cgk+zSS1qgzianYt8Lzs/LzN4gzaa/+NqMS6qS4HLozoqfQnPbZEp98QyaRSp9P98pSl4oSzcgmca/P78nTTL08z/sVManD9q9z1J9p/8db8aob7JeQl4epsPrz6agW3Lr4ryaRApdz3agYDq7YM47HFqgzkanYMGLSbP9LA/bGIa/+nprSe+9LI4gzVPDbrJg+P4fprLFTALMm7+LSb4d+kpdzt/7b7wrQM498cqBzSpr8g/FSh+bzQygL9nSm7qSmM4epQ4flY/BQdqA+l4oYQ2BpAPp87arS34nMQyFSE8nkdqMD6pMzd8/4SL7bF8aRr+7+rG7mkqBpD8pSUzozQcA8Szb87PDSb/d+/qgzVJfl/4LExpdzQ4fRSy7bFP9+y+7+nJAzdaLp/2LSbJBL3cL8ra/+bLrTQwrQQyp4QnSm7cLS9z9iFq9pAnLSwq7Yn4M+QcA4S8rQ98/ZIzLkQy/mSPp+P+gbc4BlQyrTSPUuI8/bI8npnLoz7/bpw8pSl478Q4S+dGLbSqAmn4BTT8nRSngbFNFShJozQ2bk/J7p7494n47mAqFz0afQ6qAmxJ7+k4gzjanTO8p+p4dPl/L89anSmq98QN7+hcDzDanT98/+c4FYQ4fl1NMmFc9+c49MQyrRA8Sm7LLS9+g+npd4QaLpNqMSc4eD3Lo4Nqdb7+FDAyFlQyBQ+qSS+4FQs87+fLo4NagYC/rS3PBpxqg46JSSzPf+Y8g+rJFqMaLp8GSbc4rT6qgqAa/+O8gYc4BSz/epSyn+mqA+Iyo4QyBpAyASOqA+M4o+sqgzYagW7q9kM4bYQy/4A8omy2LDAJ7+DpLkAngbFLDSe/d+h4g4p+Bpz/DSbnjTQ404A2rF98gW74d+8zrbAL7pF2DS3/ebQ2BT8qS87+rSetMpQyB4AyF8lPUTmN9pg4gzHaLpIPfpM4ebTnfpS8rMm8p4n4BESLocUanS+wLSiybb14gqFwb8FGDShn0FjNsQhwaHCP/DF+eLI+/rMwaIj2erIH0iINsQhP/rjwjQ1J7QTGnIjKc=="
        )
        expected_xs_post = (
            "XYS_2UQhPsHCH0c1Pjh9HjIj2erjwjQhyoPTqBPT49pjHjIj2eHjwjQgynEDJ74AHjIj2ePjwjQTJdPIPAZlg94aGLTl/L4Aq0DA/9F9yB8t4emk8epx/Ac7yL+awepncDTx2bDAJLWUy0mB+FDF8B8LL9p/qDlkp94NLgSIpnpMaS+3JnH3P0mDabkaPMkDpBkCcDp64/DI+0S8N9L7+Fla2nEBnrPhLnQEzLTD4/DFaMbBPpmHaMY/aS4PwBSecSz+c9EIqMQCLDkcpnbLP9IUJDT/Jfznnfl0yLLIaSQQyAmOarEaLSz+qFzD4BTHN7mA/rM6cUReqAScpLbp2aHVHdWFH0ijHdF="
        )
        expected_xs_get = (
            "XYS_2UQhPsHCH0c1Pjh9HjIj2erjwjQhyoPTqBPT49pjHjIj2eHjwjQgynEDJ74AHjIj2ePjwjQTJdPIPAZlg94aGLTl2Szr8oi7/9F9yB8t4emk8rSxG0c7yL+awepn+URx2bSi/rWUy0Z7+FDF8o408Dkg4d4aLMLALgSIpnpMaS+3JnH3P0mDabkaPMkDpBkCcDp64/DI+0S8N9L7+Fla2nEBnrPhLnQEzLTD4/DFaMbBPpmHaMY/aS4PwBSecSz+c9EIqMQCLDkcpnbLP9l12LT/Jfznnfl0yLLIaSQQyAmOarEaLSz+G0+98gG3qDQpP0GhqF+GLrlUcD8wnaHVHdWFH0ijHdF="
        )
        expected_xs_get_empty = (
            "XYS_2UQhPsHCH0c1Pjh9HjIj2erjwjQhyoPTqBPT49pjHjIj2eHjwjQgynEDJ74AHjIj2ePjwjQTJdPIPAZlg94aGLTlqdZA4r+c/9F9yB8t4emk8r+Qn/zdyL+awepnz/4x2bDF/rLUyfE/+FDF8rQdnezh+dGM/fh7LgSIpnpMaS+3JnH3P0mDabkaPMkDpBkCcDp64/DI+0S8N9L7+Fla2nEBnrPhLnQEzLTD4/DFaMbBPpmHaMY/aS4PwBSecSz+c9EIqMQCLDkcpnbLP9lQqDT/Jfznnfl0yLLIaSQQyAmOarEaLSz+G9YxaUV9+gSSanMzyBE0ngzoy789+aHVHdWFH0ijHdF="
        )

        self.assertEqual(
            xhs_client.md5_hex("pawwake-xhs-test-中文"),
            "41d64e0bbda7117674850b90cd7b06de",
        )
        self.assertEqual(
            xhs_client.encode_custom_str("hello 小红书 123"),
            "yBpVJBuWEJsOEvxjENffHerUPI==",
        )
        b1 = xhs_client.generate_b1(fp)
        self.assertEqual(b1, expected_b1)
        self.assertEqual(xhs_client.crc32_js_int(b1), 1944505159)
        self.assertEqual(
            xhs_client.crc32_js_int(
                "ZmserbBoHQtNP+wOcza/LpngG8yJq42KWYj0DSfdikx3VT16IlUAFM97hECvuRX5"
            ),
            -1287845881,
        )
        self.assertEqual(xhs_client.sign_xs_common({"a1": a1}, fp), expected_xs_common)
        self.assertEqual(
            xhs_client.sign_xs(
                "POST",
                "/api/sns/web/v1/feed",
                a1,
                payload=post_payload,
                timestamp_sec=ts_sec,
                rng=rng,
            ),
            expected_xs_post,
        )
        self.assertEqual(
            xhs_client.sign_xs(
                "GET",
                "/api/sns/web/v2/comment/page",
                a1,
                payload=get_params,
                timestamp_sec=ts_sec,
                rng=rng,
            ),
            expected_xs_get,
        )
        self.assertEqual(
            xhs_client.sign_xs(
                "GET",
                "/api/sns/web/v2/user/me",
                a1,
                payload=None,
                timestamp_sec=ts_sec,
                rng=rng,
            ),
            expected_xs_get_empty,
        )


class XhsLinkParsingTests(unittest.TestCase):
    def test_extract_xhs_url_from_share_copy(self):
        copy_text = "48 【南京超好喝的手冲咖啡 - 小红书】 😆 abc123 😆 https://xhslink.com/o/2aBcDeFgHiJ，复制后打开【小红书】查看笔记！"
        self.assertEqual(
            xhs_client.extract_xhs_url(copy_text),
            "https://xhslink.com/o/2aBcDeFgHiJ",
        )

        cn_copy_text = "99 【测试分享 - 小红书】 http://xhslink.cn/o/9xYzAbCdEfG 复制打开"
        self.assertEqual(
            xhs_client.extract_xhs_url(cn_copy_text),
            "http://xhslink.cn/o/9xYzAbCdEfG",
        )
        self.assertTrue(xhs_client._is_short_host("http://xhslink.cn/o/9xYzAbCdEfG"))

        glued = "看看这篇 http://xhslink.com/a/AbCdEf12345复制打开小红书"
        self.assertEqual(
            xhs_client.extract_xhs_url(glued),
            "http://xhslink.com/a/AbCdEf12345",
        )

        wrapped = "（https://www.xiaohongshu.com/explore/65a1b2c3000000001e02a3b4?xsec_token=tok123）"
        self.assertEqual(
            xhs_client.extract_xhs_url(wrapped),
            "https://www.xiaohongshu.com/explore/65a1b2c3000000001e02a3b4?xsec_token=tok123",
        )

    def test_rejects_non_whitelisted_and_userinfo_urls(self):
        self.assertEqual(xhs_client.extract_xhs_url("https://evil.com/explore/1234567890abcdef"), "")
        self.assertEqual(
            xhs_client.extract_xhs_url("https://xhslink.com@evil.com/o/abc123"),
            "",
        )
        self.assertEqual(xhs_client.extract_xhs_url("http://127.0.0.1:8000/test"), "")
        self.assertEqual(xhs_client.extract_xhs_url("file:///etc/passwd"), "")

    def test_parse_note_url_path_variants(self):
        nid = "65a1b2c3000000001e02a3b4"
        for path in (
            f"https://www.xiaohongshu.com/explore/{nid}?xsec_token=tok_a&xsec_source=pc_user",
            f"https://www.xiaohongshu.com/discovery/item/{nid}?xsec_token=tok_a&xsec_source=app_share",
            f"https://www.xiaohongshu.com/user/profile/5f123456/{nid}?xsec_token=tok_a",
            f"https://www.xiaohongshu.com/notes/{nid}?xsec_token=tok_a",
        ):
            parsed_id, token, source = xhs_client.parse_note_url(path)
            self.assertEqual(parsed_id, nid)
            self.assertEqual(token, "tok_a")
            self.assertTrue(source in ("pc_user", "app_share", "pc_feed"))


class XhsFetchAndMcpTests(unittest.IsolatedAsyncioTestCase):
    async def test_shortlink_redirect_and_note_truncation(self):
        nid = "65a1b2c3000000001e02a3b4"
        long_desc = "好" * 1800
        seen_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(str(request.url))
            if request.url.host == "xhslink.com":
                return httpx.Response(
                    302,
                    headers={
                        "Location": f"https://www.xiaohongshu.com/discovery/item/{nid}?xsec_token=tok_xyz&xsec_source=app_share"
                    },
                )
            if request.url.path == "/api/sns/web/v1/feed":
                body = json.loads(request.content.decode("utf-8"))
                self.assertEqual(body["source_note_id"], nid)
                self.assertEqual(body["xsec_token"], "tok_xyz")
                self.assertEqual(body["xsec_source"], "app_share")
                self.assertTrue(request.headers.get("x-s", "").startswith("XYS_"))
                self.assertTrue(len(request.headers.get("x-s-common", "")) > 20)
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {
                            "items": [
                                {
                                    "note_card": {
                                        "title": "标题" * 80,
                                        "desc": long_desc,
                                        "type": "normal",
                                        "user": {"nickname": "咖啡脑袋"},
                                        "image_list": [
                                            {"url": "https://sns-img.xhscdn.com/secret1.jpg"},
                                            {"url": "https://sns-img.xhscdn.com/secret2.jpg"},
                                        ],
                                        "tag_list": [{"name": f"标签{i}"} for i in range(15)],
                                    }
                                }
                            ]
                        },
                    },
                )
            if request.url.path == "/api/sns/web/v2/comment/page":
                raise AssertionError("include_comments=False should not request comments")
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            res = await xhs_client.fetch_note_detail(
                "分享给你 https://xhslink.com/o/short123 快看",
                cookie="a1=test_a1_val; web_session=secret_cookie_val",
                include_comments=False,
                client=client,
            )

        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["note_id"], nid)
        self.assertLessEqual(len(res["title"]), xhs_client.MAX_TITLE_CHARS + 1)
        self.assertIn("…（正文较长，已截断）", res["desc"])
        self.assertEqual(len(res["tags"]), xhs_client.MAX_TAGS)
        text = res["formatted_text"]
        self.assertIn("阿狸", text)
        self.assertIn("不是阿狸自己的亲身经历", text)
        self.assertIn("图文笔记（共 2 张图）", text)
        self.assertNotIn("用户", text)
        self.assertNotIn("AI", text)
        self.assertNotIn("secret1.jpg", text)
        self.assertNotIn("secret_cookie_val", text)
        self.assertNotIn("comment/page", "\n".join(seen_paths))

    async def test_include_comments_caps_count_and_length(self):
        nid = "65a1b2c3000000001e02a3b4"
        comments_data = [
            {
                "id": f"c{i}",
                "content": ("评" * 150) if i == 0 else f"评论内容 {i}",
                "like_count": str(100 - i),
                "user_info": {"nickname": f"路人{i}"},
            }
            for i in range(10)
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/sns/web/v1/feed":
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {
                            "items": [
                                {
                                    "note_card": {
                                        "title": "短标题",
                                        "desc": "短正文",
                                        "type": "video",
                                        "user": {"nickname": "视频博主"},
                                    }
                                }
                            ]
                        },
                    },
                )
            if request.url.path == "/api/sns/web/v2/comment/page":
                return httpx.Response(
                    200,
                    json={"success": True, "data": {"comments": comments_data}},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            res = await xhs_client.fetch_note_detail(
                f"https://www.xiaohongshu.com/explore/{nid}?xsec_token=tok1",
                cookie="a1=test_a1_val",
                include_comments=True,
                client=client,
            )

        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["comments"]), xhs_client.MAX_COMMENTS)
        self.assertLessEqual(len(res["comments"][0]["content"]), xhs_client.MAX_COMMENT_CHARS + 1)
        self.assertIn("精选评论：", res["formatted_text"])
        self.assertIn("视频笔记", res["formatted_text"])

    async def test_ssrf_redirect_blocked(self):
        called_targets: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            called_targets.append(str(request.url))
            if request.url.host == "xhslink.com":
                return httpx.Response(
                    302,
                    headers={"Location": "http://127.0.0.1:8080/internal-secret"},
                )
            raise AssertionError(f"Should never follow redirect to {request.url}")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            res = await xhs_client.fetch_note_detail(
                "https://xhslink.com/o/evilredirect",
                cookie="a1=test_a1_val",
                client=client,
            )

        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error_code"], "invalid_url")
        self.assertEqual(called_targets, ["https://xhslink.com/o/evilredirect"])
        self.assertIn("阿狸", res["message"])
        self.assertNotIn("用户", res["message"])
        self.assertNotIn("AI", res["message"])

    async def test_error_states_and_gentle_wording(self):
        # 1. Missing cookie
        r_missing = await xhs_client.fetch_note_detail(
            "https://www.xiaohongshu.com/explore/65a1b2c3000000001e02a3b4",
            cookie="",
        )
        self.assertEqual(r_missing["status"], "error")
        self.assertEqual(r_missing["error_code"], "missing_cookie")

        # 2. Missing a1 field
        r_no_a1 = await xhs_client.fetch_note_detail(
            "https://www.xiaohongshu.com/explore/65a1b2c3000000001e02a3b4",
            cookie="web_session=abc",
        )
        self.assertEqual(r_no_a1["status"], "error")
        self.assertEqual(r_no_a1["error_code"], "invalid_cookie")

        # 3. Expired cookie (-100)
        def expired_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"success": False, "code": -100, "msg": "未登录"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(expired_handler)) as client:
            r_expired = await xhs_client.fetch_note_detail(
                "https://www.xiaohongshu.com/explore/65a1b2c3000000001e02a3b4",
                cookie="a1=expired_val",
                client=client,
            )
        self.assertEqual(r_expired["status"], "error")
        self.assertEqual(r_expired["error_code"], "cookie_expired")

        # 4. Timeout
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
            r_timeout = await xhs_client.fetch_note_detail(
                "https://www.xiaohongshu.com/explore/65a1b2c3000000001e02a3b4",
                cookie="a1=ok_val",
                client=client,
            )
        self.assertEqual(r_timeout["status"], "error")
        self.assertEqual(r_timeout["error_code"], "timeout")

        for item in (r_missing, r_no_a1, r_expired, r_timeout):
            self.assertIn("阿狸", item["message"])
            self.assertNotIn("用户", item["message"])
            self.assertNotIn("AI", item["message"])
            self.assertNotIn("expired_val", item["message"])
            self.assertNotIn("ok_val", item["message"])

    async def test_mcp_tool_works_without_db_and_validates_inputs(self):
        orig_db = shared.DATABASE_ENABLED
        orig_mem = shared.MEMORY_ENABLED
        orig_cookie = shared.XHS_COOKIE
        try:
            shared.DATABASE_ENABLED = False
            shared.MEMORY_ENABLED = False
            shared.XHS_COOKIE = "a1=mcp_cookie"

            # Blank URL returns gentle error
            blank = await mcp_server.read_xiaohongshu_note("   ")
            self.assertEqual(blank["status"], "error")
            self.assertEqual(blank["error_code"], "invalid_url")
            self.assertIn("阿狸", blank["message"])

            # Non-bool include_comments rejected cleanly
            with self.assertRaises(ValueError):
                await mcp_server.read_xiaohongshu_note(
                    "https://xhslink.com/o/abc",
                    include_comments="true",  # type: ignore[arg-type]
                )

            # Successful call passes cookie from shared.XHS_COOKIE
            with patch(
                "xhs_client.fetch_note_detail",
                new=AsyncMock(
                    return_value={
                        "status": "ok",
                        "note_id": "65a1b2c3000000001e02a3b4",
                        "title": "测试标题",
                        "author": "测试作者",
                        "formatted_text": "这是阿狸刚刚分享给你看的小红书笔记",
                    }
                ),
            ) as mock_fetch:
                ok_res = await mcp_server.read_xiaohongshu_note(
                    "https://xhslink.com/o/abc",
                    include_comments=True,
                )
            mock_fetch.assert_awaited_once_with(
                "https://xhslink.com/o/abc",
                cookie="a1=mcp_cookie",
                include_comments=True,
            )
            self.assertEqual(ok_res["status"], "ok")
            self.assertEqual(ok_res["note_id"], "65a1b2c3000000001e02a3b4")
            self.assertEqual(ok_res["content"], "这是阿狸刚刚分享给你看的小红书笔记")

            # Verify MCP tool description avoids "用户" and "AI"
            tools = await mcp_server.server.list_tools()
            xhs_tool = next((t for t in tools if t.name == "read_xiaohongshu_note"), None)
            self.assertIsNotNone(xhs_tool)
            desc = xhs_tool.description or ""
            self.assertIn("阿狸", desc)
            self.assertNotIn("用户", desc)
            self.assertNotIn("AI", desc)
        finally:
            shared.DATABASE_ENABLED = orig_db
            shared.MEMORY_ENABLED = orig_mem
            shared.XHS_COOKIE = orig_cookie


class XhsDashboardSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_xhs_cookie_settings_mask_save_and_ui_wiring(self):
        orig_cookie = shared.XHS_COOKIE
        orig_env = os.environ.get("XHS_COOKIE")
        try:
            self.assertIn("XHS_COOKIE", shared.SETTINGS_TYPES)
            self.assertIn("XHS_COOKIE", shared.SETTINGS_ALLOW_EMPTY)
            self.assertIn("XHS_COOKIE", dashboard_routes._MASKED_KEYS)
            self.assertEqual(shared.SETTINGS_TYPES["XHS_COOKIE"]("  a1=abc  "), "a1=abc")

            shared.XHS_COOKIE = "a1=1234567890; web_session=abcdef"
            with patch(
                "db.core.get_all_gateway_config",
                new=AsyncMock(return_value={"XHS_COOKIE": "a1=1234567890; web_session=abcdef"}),
            ), patch(
                "db.core.get_vector_dimensions",
                new=AsyncMock(return_value={}),
            ), patch(
                "db.core.get_embedding_vector_presence",
                new=AsyncMock(return_value=None),
            ):
                got = await dashboard_routes.get_settings()
            masked_val = got["settings"]["XHS_COOKIE"]
            self.assertIn("****", masked_val)
            self.assertNotIn("web_session=abcdef", masked_val)

            async def _save(payload: dict):
                req = Request(
                    {
                        "type": "http",
                        "method": "PUT",
                        "path": "/api/settings",
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                req._json = payload
                return await dashboard_routes.save_settings(req)

            with patch("db.core.set_gateway_config", new=AsyncMock()) as mock_set:
                # 1. Submitting masked value skips update
                res_skip = await _save({"XHS_COOKIE": masked_val})
                self.assertIn("XHS_COOKIE", res_skip["skipped"])
                mock_set.assert_not_awaited()

                # 2. Submitting new cookie updates shared and DB
                res_upd = await _save({"XHS_COOKIE": " a1=new_val; webId=xyz "})
                self.assertIn("XHS_COOKIE", res_upd["updated"])
                self.assertEqual(shared.XHS_COOKIE, "a1=new_val; webId=xyz")
                mock_set.assert_awaited_once_with("XHS_COOKIE", "a1=new_val; webId=xyz")

                # 3. Submitting empty string clears cookie
                mock_set.reset_mock()
                res_clr = await _save({"XHS_COOKIE": ""})
                self.assertIn("XHS_COOKIE", res_clr["updated"])
                self.assertEqual(shared.XHS_COOKIE, "")
                mock_set.assert_awaited_once_with("XHS_COOKIE", "")

            html_src = (ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")
            js_src = (ROOT / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
            self.assertIn('id="set-XHS_COOKIE"', html_src)
            self.assertIn('id="set-XHS_COOKIE-hint"', html_src)
            self.assertIn("'XHS_COOKIE'", js_src)
        finally:
            shared.XHS_COOKIE = orig_cookie
            if orig_env is None:
                os.environ.pop("XHS_COOKIE", None)
            else:
                os.environ["XHS_COOKIE"] = orig_env


if __name__ == "__main__":
    unittest.main()
