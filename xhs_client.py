"""Pure-Python Xiaohongshu Web API client for reading shared notes."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import struct
import time
from typing import Any
from urllib.parse import parse_qs, quote, urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

STANDARD_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
CUSTOM_B64 = "ZmserbBoHQtNP+wOcza/LpngG8yJq42KWYj0DSfdikx3VT16IlUAFM97hECvuRX5"
X3_B64 = "MfgqrsbcyzPQRStuvC7mn501HIJBo2DEFTKdeNOwxWXYZap89+/A4UVLhijkl63G"
HEX_KEY = (
    "71a302257793271ddd273bcee3e4b98d9d7935e1da33f5765e2ea8afb6dc77a5"
    "1a499d23b67c20660025860cbf13d4540d92497f58686c574e508f46e1956344"
    "f39139bf4faf22a3eef120b79258145b2feb5193b6478669961298e79bedca64"
    "6e1a693a926154a5a7a1bd1cf0dedb742f917a747a1e388b234f2277516db711"
    "6035439730fa61e9822a0eca7bff72d8"
)
VERSION_BYTES = [121, 104, 96, 41]
PAYLOAD_LENGTH = 144
A1_LENGTH = 52
APP_ID_LENGTH = 10
A3_PREFIX = [2, 97, 51, 16]
ENV_TABLE = [115, 248, 83, 102, 103, 201, 181, 131, 99, 94, 4, 68, 250, 132, 21]
ENV_CHECKS_DEFAULT = [0, 1, 18, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0]
HASH_IV = [1831565813, 461845907, 2246822507, 3266489909]
X3_PREFIX = "mns0301_"
XYS_PREFIX = "XYS_"
B1_SECRET_KEY = "xhswebmplfbt"

SIGNATURE_DATA_TEMPLATE = {
    "x0": "4.2.6",
    "x1": "xhs-pc-web",
    "x2": "Windows",
    "x3": "",
    "x4": "",
}
SIGNATURE_XSCOMMON_TEMPLATE = {
    "s0": 5,
    "s1": "",
    "x0": "1",
    "x1": "4.2.6",
    "x2": "Windows",
    "x3": "xhs-pc-web",
    "x4": "4.86.0",
    "x5": "",
    "x6": "",
    "x7": "",
    "x8": "",
    "x9": -596800761,
    "x10": 0,
    "x11": "normal",
}

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
)
IMG_FORMATS = ["jpg", "webp", "avif"]
EDITH = "https://edith.xiaohongshu.com"
WWW = "https://www.xiaohongshu.com"

CRC_POLY = 0xEDB88320
_CRC_TABLE: list[int] | None = None

GPU_VENDORS = [
    "Google Inc. (Intel)|ANGLE (Intel, Intel(R) UHD Graphics 630 (0x00003E9B) Direct3D11 vs_5_0 ps_5_0, D3D11)",
    "Google Inc. (NVIDIA)|ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 (0x0000250F) Direct3D11 vs_5_0 ps_5_0, D3D11)",
    "Google Inc. (AMD)|ANGLE (AMD, AMD Radeon RX 6600 (0x000073FF) Direct3D11 vs_5_0 ps_5_0, D3D11)",
]
SCREEN_RES = ["1366;768", "1920;1080", "2560;1440"]
HEX_CHARS = "abcdef0123456789"

DEFAULT_TIMEOUT_SECONDS = 8.0
MAX_REDIRECTS = 5
MAX_TITLE_CHARS = 120
MAX_DESC_CHARS = 1500
MAX_COMMENTS = 5
MAX_COMMENT_CHARS = 100
MAX_TAGS = 10

_ALLOWED_HOSTS = ("xhslink.com", "xhslink.cn", "xiaohongshu.com")
_SHORT_HOSTS = ("xhslink.com", "xhslink.cn")
_URL_RE = re.compile(
    r"https?://[^\s<>'\"`，。；：！？、\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+",
    re.IGNORECASE,
)
_TRAILING_PUNCTUATION = ".,;:!?，。；：！？、"
_TRAILING_BRACKETS = ")]}）】》」』"
_BRACKET_OPENERS = {
    "）": "（",
    ")": "(",
    "]": "[",
    "}": "{",
    "】": "【",
    "》": "《",
    "」": "「",
    "』": "『",
}
_NOTE_PATH_PATTERNS = (
    re.compile(r"^/explore/([0-9A-Za-z_-]{16,32})(?:/|$)"),
    re.compile(r"^/discovery/item/([0-9A-Za-z_-]{16,32})(?:/|$)"),
    re.compile(r"^/user/profile/[^/]+/([0-9A-Za-z_-]{16,32})(?:/|$)"),
    re.compile(r"^/notes/([0-9A-Za-z_-]{16,32})(?:/|$)"),
)


class _DefaultRNG:
    """Default cryptographic RNG matching JS RNG interface."""

    def randint(self, a: int, b: int) -> int:
        return a + secrets.randbelow(b - a + 1)

    def randbytes(self, n: int) -> bytes:
        return os.urandom(n)


_DEFAULT_RNG = _DefaultRNG()


def _u32(v: int) -> int:
    return v & 0xFFFFFFFF


def _rotl(v: int, n: int) -> int:
    return ((v << n) | (v >> (32 - n))) & 0xFFFFFFFF


def _int_to_le_bytes(val: int, length: int = 4) -> list[int]:
    arr = []
    v = int(val)
    for _ in range(length):
        arr.append(v & 0xFF)
        v //= 256
    return arr


def _hex_to_bytes(hex_str: str) -> list[int]:
    return [int(hex_str[i : i + 2], 16) for i in range(0, len(hex_str), 2)]


def md5_hex(data: str | bytes | bytearray | list[int]) -> str:
    """Compute lowercase MD5 hex digest."""
    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        raw = bytes(b & 0xFF for b in data)
    return hashlib.md5(raw).hexdigest()


def _bytes_to_std_b64(data: bytes | bytearray | list[int]) -> str:
    n = len(data)
    out: list[str] = []
    for i in range(0, n, 3):
        b0 = data[i]
        b1 = data[i + 1] if i + 1 < n else 0
        b2 = data[i + 2] if i + 2 < n else 0
        out.append(STANDARD_B64[b0 >> 2])
        out.append(STANDARD_B64[((b0 & 3) << 4) | (b1 >> 4)])
        out.append(STANDARD_B64[((b1 & 15) << 2) | (b2 >> 6)] if i + 1 < n else "=")
        out.append(STANDARD_B64[b2 & 63] if i + 2 < n else "=")
    return "".join(out)


def _translate_alphabet(encoded: str, to_alphabet: str) -> str:
    table = {ch: to_alphabet[idx] for idx, ch in enumerate(STANDARD_B64)}
    return "".join(table.get(ch, ch) for ch in encoded)


def encode_custom(data: bytes | bytearray | list[int]) -> str:
    return _translate_alphabet(_bytes_to_std_b64(data), CUSTOM_B64)


def encode_x3(data: bytes | bytearray | list[int]) -> str:
    return _translate_alphabet(_bytes_to_std_b64(data), X3_B64)


def encode_custom_str(text: str) -> str:
    return encode_custom(text.encode("utf-8"))


def _crc_table() -> list[int]:
    global _CRC_TABLE
    if _CRC_TABLE is not None:
        return _CRC_TABLE
    tbl: list[int] = []
    for d in range(256):
        r = d
        for _ in range(8):
            r = ((r >> 1) ^ CRC_POLY) if (r & 1) else (r >> 1)
        tbl.append(r & 0xFFFFFFFF)
    _CRC_TABLE = tbl
    return tbl


def crc32_js_int(text: str) -> int:
    """Exact port of xhs_lite_worker.js crc32JsInt (note extra CRC_POLY XOR)."""
    tbl = _crc_table()
    c = 0xFFFFFFFF
    for ch in text:
        b = ord(ch) & 0xFF
        c = (tbl[(c ^ b) & 0xFF] ^ (c >> 8)) & 0xFFFFFFFF
    v = ((0xFFFFFFFF ^ c) ^ CRC_POLY) & 0xFFFFFFFF
    return v - 0x100000000 if (v & 0x80000000) else v


def rc4(key_bytes: bytes, data_bytes: bytes) -> bytes:
    """Standard 256-byte RC4 stream cipher."""
    s_box = list(range(256))
    j = 0
    key_len = len(key_bytes)
    for i in range(256):
        j = (j + s_box[i] + key_bytes[i % key_len]) & 0xFF
        s_box[i], s_box[j] = s_box[j], s_box[i]
    out = bytearray(len(data_bytes))
    a = 0
    b = 0
    for k, byte_val in enumerate(data_bytes):
        a = (a + 1) & 0xFF
        b = (b + s_box[a]) & 0xFF
        s_box[a], s_box[b] = s_box[b], s_box[a]
        out[k] = byte_val ^ s_box[(s_box[a] + s_box[b]) & 0xFF]
    return bytes(out)


def custom_hash_v2(input_bytes: bytes | bytearray | list[int]) -> list[int]:
    """Exact port of xhs_lite_worker.js customHashV2."""
    buf = bytes(b & 0xFF for b in input_bytes)
    s0, s1, s2, s3 = HASH_IV
    length = len(buf)
    s0 = _u32(s0 ^ length)
    s1 = _u32(s1 ^ _u32(length << 8))
    s2 = _u32(s2 ^ _u32(length << 16))
    s3 = _u32(s3 ^ _u32(length << 24))
    for i in range(length // 8):
        v0, v1 = struct.unpack_from("<II", buf, i * 8)
        s0 = _rotl(_u32(_u32(s0 + v0) ^ s2), 7)
        s1 = _rotl(_u32(_u32(v0 ^ s1) + s3), 11)
        s2 = _rotl(_u32(_u32(s2 + v1) ^ s0), 13)
        s3 = _rotl(_u32(_u32(s3 ^ v1) + s1), 17)
    t0 = _u32(s0 ^ length)
    t1 = _u32(s1 ^ t0)
    t2 = _u32(s2 + t1)
    t3 = _u32(s3 ^ t2)
    r0 = _rotl(t0, 9)
    r1 = _rotl(t1, 13)
    r2 = _rotl(t2, 17)
    r3 = _rotl(t3, 19)
    s0 = _u32(r0 + r2)
    s1 = _u32(r1 ^ r3)
    s2 = _u32(r2 + s0)
    s3 = _u32(r3 ^ s1)
    result: list[int] = []
    for s_val in (s0, s1, s2, s3):
        result.extend(_int_to_le_bytes(s_val, 4))
    return result


def py_quote(value: str, safe_extra: str = "") -> str:
    """Percent-encode UTF-8 bytes matching JS pyQuote."""
    always = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~"
    return quote(value, safe=always + safe_extra)


def json_compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def build_content_string(method: str, uri: str, payload: dict | None) -> str:
    payload = payload or {}
    if method.upper() == "POST":
        return uri + json_compact(payload)
    if not payload:
        return uri
    parts: list[str] = []
    for k, v in payload.items():
        if isinstance(v, list):
            s = ",".join(str(item) for item in v)
        elif v is not None:
            s = str(v)
        else:
            s = ""
        parts.append(f"{k}={py_quote(s, ',')}")
    return f"{uri}?{'&'.join(parts)}"


def extract_api_path(s: str) -> str:
    brace = s.find("{")
    q = s.find("?")
    if brace != -1 and q != -1:
        return s[: min(brace, q)]
    if brace != -1:
        return s[:brace]
    if q != -1:
        return s[:q]
    return s


def extract_uri(uri: str) -> str:
    uri = uri.strip()
    if uri.startswith("http"):
        return urlparse(uri).path
    q = uri.find("?")
    return uri if q == -1 else uri[:q]


def build_payload_array(
    d_value: str,
    a1_value: str,
    app_id: str,
    string_param: str,
    timestamp_sec: float,
    *,
    rng: Any = None,
) -> list[int]:
    rng = rng or _DEFAULT_RNG
    seed = rng.randint(0, 0xFFFFFFFF)
    seed_byte = seed & 0xFF
    payload: list[int] = list(VERSION_BYTES)
    payload.extend(_int_to_le_bytes(seed, 4))
    ts_ms = math.floor(timestamp_sec * 1000)
    ts_bytes = _int_to_le_bytes(ts_ms, 8)
    payload.extend(ts_bytes)
    time_offset = rng.randint(10, 50)
    payload.extend(_int_to_le_bytes(math.floor((timestamp_sec - time_offset) * 1000), 8))
    payload.extend(_int_to_le_bytes(rng.randint(15, 50), 4))
    payload.extend(_int_to_le_bytes(rng.randint(1000, 1200), 4))
    payload.extend(_int_to_le_bytes(len(string_param.encode("utf-8")), 4))
    md5_bytes = _hex_to_bytes(d_value)
    for i in range(8):
        payload.append(md5_bytes[i] ^ seed_byte)
    a1_full = a1_value.encode("utf-8")[:A1_LENGTH]
    a1_bytes = a1_full + b"\x00" * (A1_LENGTH - len(a1_full))
    payload.append(A1_LENGTH)
    payload.extend(a1_bytes)
    app_full = app_id.encode("utf-8")[:APP_ID_LENGTH]
    app_bytes = app_full + b"\x00" * (APP_ID_LENGTH - len(app_full))
    payload.append(APP_ID_LENGTH)
    payload.extend(app_bytes)
    part11 = [1, seed_byte ^ ENV_TABLE[0]]
    for i in range(1, 15):
        part11.append(ENV_TABLE[i] ^ ENV_CHECKS_DEFAULT[i])
    payload.extend(part11)
    md5_path_bytes = _hex_to_bytes(md5_hex(extract_api_path(string_param)))
    hashed = custom_hash_v2(ts_bytes + md5_path_bytes)
    payload.extend(A3_PREFIX)
    payload.extend(b ^ seed_byte for b in hashed)
    return payload


def xor_transform(src: list[int]) -> bytes:
    key = _hex_to_bytes(HEX_KEY)
    key_len = len(key)
    return bytes(
        (src[i] ^ key[i]) & 0xFF if i < key_len else src[i] & 0xFF
        for i in range(len(src))
    )


def sign_xs(
    method: str,
    uri: str,
    a1_value: str,
    *,
    app_id: str = "xhs-pc-web",
    payload: dict | None = None,
    timestamp_sec: float | None = None,
    rng: Any = None,
) -> str:
    uri = extract_uri(uri)
    if timestamp_sec is None:
        timestamp_sec = time.time()
    content_string = build_content_string(method, uri, payload)
    d_value = md5_hex(content_string)
    payload_arr = build_payload_array(
        d_value,
        a1_value,
        app_id,
        content_string,
        timestamp_sec,
        rng=rng,
    )
    xor_result = xor_transform(payload_arr)
    x3_sig = encode_x3(xor_result[:PAYLOAD_LENGTH])
    data = {**SIGNATURE_DATA_TEMPLATE, "x3": X3_PREFIX + x3_sig}
    return XYS_PREFIX + encode_custom_str(json_compact(data))


def generate_b1(fp: dict[str, Any]) -> str:
    """Exact port of xhs_lite_worker.js generateB1 (preserving Latin-1->UTF-8 encoding)."""
    keys = [
        "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x42", "x43",
        "x44", "x45", "x46", "x48", "x49", "x50", "x51", "x52", "x82",
    ]
    b1_fp = {k: fp.get(k) for k in keys}
    cipher = rc4(B1_SECRET_KEY.encode("utf-8"), json_compact(b1_fp).encode("utf-8"))
    cipher_str = "".join(chr(b) for b in cipher)
    encoded_url = py_quote(cipher_str, "!*'()~_-")
    out_bytes: list[int] = []
    for seg in encoded_url.split("%")[1:]:
        out_bytes.append(int(seg[:2], 16))
        for ch in seg[2:]:
            out_bytes.append(ord(ch))
    return encode_custom(out_bytes)


def generate_fingerprint(
    cookies: dict[str, str],
    *,
    rng: Any = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    rng = rng or _DEFAULT_RNG

    def pick(arr: list[Any]) -> Any:
        return arr[rng.randint(0, len(arr) - 1)]

    def rand_md5() -> str:
        return md5_hex(rng.randbytes(32))

    cookie_string = "; ".join(f"{k}={v}" for k, v in cookies.items())
    w_str, h_str = pick(SCREEN_RES).split(";")
    w, h = int(w_str), int(h_str)
    incognito = "true" if rng.randint(0, 99) < 95 else "false"
    vendor, renderer = pick(GPU_VENDORS).split("|", 1)
    ts_now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    return {
        "x1": UA,
        "x2": "false",
        "x3": "zh-CN",
        "x4": pick([16, 24, 30, 32]),
        "x5": pick([2, 4, 8, 16]),
        "x6": "24",
        "x7": f"{vendor},{renderer}",
        "x8": pick([4, 6, 8, 12, 16]),
        "x9": f"{w};{h}",
        "x10": f"{w};{h}",
        "x11": "-480",
        "x12": "Asia/Shanghai",
        "x13": incognito,
        "x14": incognito,
        "x15": incognito,
        "x16": "false",
        "x17": "false",
        "x18": "un",
        "x19": "Win32",
        "x20": "",
        "x21": "PDF Viewer,Chrome PDF Viewer",
        "x22": rand_md5(),
        "x23": "false",
        "x24": "false",
        "x25": "false",
        "x26": "false",
        "x27": "false",
        "x28": "0,false,false",
        "x29": "4,7,8",
        "x30": "swf object not loaded",
        "x33": "0",
        "x34": "0",
        "x35": "0",
        "x36": str(rng.randint(1, 20)),
        "x37": "0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|1|0|0|0|0|0",
        "x38": "0|0|1|0|1|0|0|0|0|0|1|0|1|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0",
        "x39": 0,
        "x40": "0",
        "x41": "0",
        "x42": "3.4.4",
        "x43": rand_md5(),
        "x44": str(ts_now_ms),
        "x45": "__SEC_CAV__1-1-1-1-1|__SEC_WSA__|",
        "x46": "false",
        "x47": "1|0|0|0|0|0",
        "x48": "",
        "x49": "{list:[],type:}",
        "x50": "",
        "x51": "",
        "x52": "",
        "x82": "_0x17a2|_0x1954",
        "x53": rand_md5(),
        "x57": cookie_string,
    }


def sign_xs_common(
    cookie_dict: dict[str, str],
    fingerprint: dict[str, Any] | None = None,
    *,
    rng: Any = None,
) -> str:
    fp = fingerprint if fingerprint is not None else generate_fingerprint(cookie_dict, rng=rng)
    b1 = generate_b1(fp)
    payload = {
        **SIGNATURE_XSCOMMON_TEMPLATE,
        "x5": cookie_dict.get("a1", ""),
        "x8": b1,
        "x9": crc32_js_int(b1),
    }
    return encode_custom_str(json_compact(payload))


def b3_trace_id(*, rng: Any = None) -> str:
    rng = rng or _DEFAULT_RNG
    return "".join(HEX_CHARS[rng.randint(0, 15)] for _ in range(16))


def xray_trace_id(ts_ms: int | None = None, *, rng: Any = None) -> str:
    rng = rng or _DEFAULT_RNG
    if not ts_ms:
        ts_ms = int(time.time() * 1000)
    part1 = f"{((ts_ms << 23) | rng.randint(0, 8388607)):016x}"
    part2 = "".join(HEX_CHARS[rng.randint(0, 15)] for _ in range(16))
    return part1 + part2


def sign_headers(
    method: str,
    uri: str,
    cookie_dict: dict[str, str],
    *,
    params: dict | None = None,
    payload: dict | None = None,
    timestamp_sec: float | None = None,
    fingerprint: dict[str, Any] | None = None,
    rng: Any = None,
) -> dict[str, str]:
    if timestamp_sec is None:
        timestamp_sec = time.time()
    m = method.upper()
    request_data = params if m == "GET" else payload
    ts_ms = math.floor(timestamp_sec * 1000)
    return {
        "x-s": sign_xs(
            m,
            uri,
            cookie_dict.get("a1", ""),
            payload=request_data,
            timestamp_sec=timestamp_sec,
            rng=rng,
        ),
        "x-s-common": sign_xs_common(cookie_dict, fingerprint=fingerprint, rng=rng),
        "x-t": str(ts_ms),
        "x-b3-traceid": b3_trace_id(rng=rng),
        "x-xray-traceid": xray_trace_id(ts_ms, rng=rng),
    }


def parse_cookies(cookie_str: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not cookie_str:
        return out
    for part in cookie_str.split(";"):
        idx = part.find("=")
        if idx == -1:
            continue
        key = part[:idx].strip()
        if key:
            out[key] = part[idx + 1 :].strip()
    return out


def base_headers(cookie_str: str) -> dict[str, str]:
    return {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "content-type": "application/json;charset=UTF-8",
        "origin": WWW,
        "referer": WWW + "/",
        "user-agent": UA,
        "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "x-mns": "unload",
        "cookie": cookie_str,
    }


def build_signed_query(params: dict | None) -> str:
    if not params:
        return ""
    parts: list[str] = []
    for k, v in params.items():
        if isinstance(v, list):
            s = ",".join(str(item) for item in v)
        elif v is not None:
            s = str(v)
        else:
            s = ""
        parts.append(f"{k}={py_quote(s, ',')}")
    return "&".join(parts)


async def _signed_get(
    client: httpx.AsyncClient,
    base: str,
    uri: str,
    params: dict | None,
    cookie_str: str,
    ck: dict[str, str],
    *,
    extra_headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    query = build_signed_query(params)
    sig = sign_headers("GET", uri, ck, params=params or {})
    headers = {**base_headers(cookie_str), **sig, **(extra_headers or {})}
    headers.pop("content-type", None)
    url = base + uri + (("?" + query) if query else "")
    resp = await client.get(url, headers=headers, timeout=timeout)
    text = resp.text
    try:
        data = json.loads(text)
        if not resp.is_success and isinstance(data, dict):
            data["http_status"] = resp.status_code
        return data if isinstance(data, dict) else {"success": False, "msg": text[:240]}
    except Exception:
        return {
            "success": False,
            "http_status": resp.status_code,
            "msg": f"HTTP {resp.status_code}: {text[:240]}",
        }


async def _signed_post(
    client: httpx.AsyncClient,
    base: str,
    uri: str,
    payload: dict,
    cookie_str: str,
    ck: dict[str, str],
    *,
    extra_headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    sig = sign_headers("POST", uri, ck, payload=payload)
    headers = {**base_headers(cookie_str), **sig, **(extra_headers or {})}
    resp = await client.post(
        base + uri,
        headers=headers,
        content=json_compact(payload).encode("utf-8"),
        timeout=timeout,
    )
    text = resp.text
    try:
        data = json.loads(text)
        if not resp.is_success and isinstance(data, dict):
            data["http_status"] = resp.status_code
        return data if isinstance(data, dict) else {"success": False, "msg": text[:240]}
    except Exception:
        return {
            "success": False,
            "http_status": resp.status_code,
            "msg": f"HTTP {resp.status_code}: {text[:240]}",
        }


def _clean_url(raw_url: str) -> str:
    url = (raw_url or "").strip().rstrip(_TRAILING_PUNCTUATION)
    while url and url[-1] in _TRAILING_BRACKETS:
        opener = _BRACKET_OPENERS.get(url[-1])
        if opener and url.count(opener) >= url.count(url[-1]):
            break
        url = url[:-1]
    return url


def _is_allowed_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return False
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in _ALLOWED_HOSTS)


def _is_allowed_xhs_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.username or parsed.password:
        return False
    return _is_allowed_host(parsed.hostname or "")


def extract_xhs_url(raw_text: str) -> str:
    """Extract the first whitelisted Xiaohongshu URL from raw text or share copy."""
    for match in _URL_RE.finditer(raw_text or ""):
        candidate = _clean_url(match.group(0))
        if candidate and _is_allowed_xhs_url(candidate):
            return candidate
    return ""


def parse_note_url(url: str) -> tuple[str, str, str]:
    """Parse (note_id, xsec_token, xsec_source) from a Xiaohongshu note URL."""
    if not _is_allowed_xhs_url(url):
        return "", "", "pc_feed"
    parsed = urlparse(url)
    path = parsed.path or ""
    query = parse_qs(parsed.query)
    note_id = ""
    for pattern in _NOTE_PATH_PATTERNS:
        m = pattern.match(path)
        if m:
            note_id = m.group(1)
            break
    if not note_id:
        q_note_id = (query.get("noteId") or query.get("note_id") or [""])[0].strip()
        if re.fullmatch(r"[0-9A-Za-z_-]{16,32}", q_note_id):
            note_id = q_note_id
    xsec_token = (query.get("xsec_token") or query.get("xsecToken") or [""])[0].strip()
    xsec_source = (
        query.get("xsec_source") or query.get("xsecSource") or ["pc_feed"]
    )[0].strip() or "pc_feed"
    return note_id, xsec_token, xsec_source


def _is_short_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").strip().lower().rstrip(".")
    return any(host == short or host.endswith(f".{short}") for short in _SHORT_HOSTS)


async def resolve_xhs_link(
    raw_input: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, str]:
    """Extract and safely resolve a Xiaohongshu share text or URL."""
    url = extract_xhs_url(raw_input)
    if not url:
        return {"ok": "0", "error_code": "invalid_url"}

    note_id, xsec_token, xsec_source = parse_note_url(url)
    if note_id and not _is_short_host(url):
        return {
            "ok": "1",
            "url": url,
            "note_id": note_id,
            "xsec_token": xsec_token,
            "xsec_source": xsec_source,
        }

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(2.0, timeout)),
            follow_redirects=False,
        )

    current_url = url
    deadline = time.monotonic() + timeout
    browser_headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        for _ in range(MAX_REDIRECTS):
            if not _is_allowed_xhs_url(current_url):
                return {"ok": "0", "error_code": "invalid_url"}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"ok": "0", "error_code": "timeout"}

            resp = await client.get(
                current_url,
                headers=browser_headers,
                follow_redirects=False,
                timeout=remaining,
            )
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location") or resp.headers.get("Location") or ""
                if not location:
                    break
                next_url = urljoin(current_url, location.strip())
                if not _is_allowed_xhs_url(next_url):
                    return {"ok": "0", "error_code": "invalid_url"}
                current_url = next_url
                note_id, xsec_token, xsec_source = parse_note_url(current_url)
                if note_id:
                    return {
                        "ok": "1",
                        "url": current_url,
                        "note_id": note_id,
                        "xsec_token": xsec_token,
                        "xsec_source": xsec_source,
                    }
                continue

            final_url = str(resp.url) if getattr(resp, "url", None) else current_url
            if _is_allowed_xhs_url(final_url):
                note_id, xsec_token, xsec_source = parse_note_url(final_url)
                if note_id:
                    return {
                        "ok": "1",
                        "url": final_url,
                        "note_id": note_id,
                        "xsec_token": xsec_token,
                        "xsec_source": xsec_source,
                    }
            break
    except httpx.TimeoutException:
        return {"ok": "0", "error_code": "timeout"}
    except Exception:
        return {"ok": "0", "error_code": "fetch_failed"}
    finally:
        if owns_client and client is not None:
            await client.aclose()

    note_id, xsec_token, xsec_source = parse_note_url(current_url)
    if note_id:
        return {
            "ok": "1",
            "url": current_url,
            "note_id": note_id,
            "xsec_token": xsec_token,
            "xsec_source": xsec_source,
        }
    return {"ok": "0", "error_code": "invalid_url"}


def _truncate_text(text: str, limit: int, suffix: str = "…") -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + suffix


def _error_message(error_code: str) -> str:
    if error_code in ("missing_cookie", "invalid_cookie"):
        return (
            "还没配置有效的小红书 Cookie，暂时读不了这条链接，"
            "可以先和阿狸说一声，让阿狸去后台设置里填一下小红书 Cookie，或者换种方式发给你看。"
        )
    if error_code == "cookie_expired":
        return (
            "小红书 Cookie 好像过期了，这条链接暂时没能读出来，"
            "可以直接和阿狸说一声，让阿狸更新一下 Cookie 或者换种方式发给你看。"
        )
    if error_code == "invalid_url":
        return (
            "这条内容里没识别出有效的小红书笔记链接，"
            "可以请阿狸确认一下链接，或者换种方式发给你看。"
        )
    return (
        "这条小红书链接没能成功读出来（可能是 Cookie 过期了，或者这篇笔记暂时无法访问），"
        "可以直接和阿狸说一声，让阿狸换种方式发给你看。"
    )


def _extract_tags(note_card: dict[str, Any]) -> list[str]:
    raw_tags = note_card.get("tag_list") or note_card.get("tags") or []
    if not isinstance(raw_tags, list):
        return []
    tags: list[str] = []
    for item in raw_tags:
        name = ""
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("tag_name") or "").strip()
        elif isinstance(item, str):
            name = item.strip()
        if name and name not in tags:
            tags.append(name)
            if len(tags) >= MAX_TAGS:
                break
    return tags


def _extract_top_comments(comment_resp: dict[str, Any]) -> list[dict[str, str]]:
    data = comment_resp.get("data")
    if not isinstance(data, dict):
        return []
    raw_comments = data.get("comments") or data.get("list") or []
    if not isinstance(raw_comments, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw_comments:
        if not isinstance(item, dict):
            continue
        content = re.sub(r"\s+", " ", str(item.get("content") or item.get("text") or "")).strip()
        if not content:
            continue
        user_info = item.get("user_info") or item.get("user") or {}
        if not isinstance(user_info, dict):
            user_info = {}
        author = str(user_info.get("nickname") or item.get("nickname") or "匿名").strip() or "匿名"
        likes = str(item.get("like_count") or item.get("likes") or "0").strip() or "0"
        out.append(
            {
                "author": _truncate_text(author, 40),
                "content": _truncate_text(content, MAX_COMMENT_CHARS),
                "likes": likes,
            }
        )
        if len(out) >= MAX_COMMENTS:
            break
    return out


def _format_note_text(
    *,
    title: str,
    author: str,
    note_type_label: str,
    tags: list[str],
    desc: str,
    comments: list[dict[str, str]],
    comment_note: str = "",
) -> str:
    lines = [
        "这是阿狸刚刚分享给你看的小红书笔记（外部帖子内容，不是阿狸自己的亲身经历，别把帖子里的故事记成阿狸本人的事哦）：",
        "",
        f"标题：{title}",
        f"作者：{author}",
        f"类型：{note_type_label}",
    ]
    if tags:
        lines.append(f"标签：{' '.join(f'#{t}' for t in tags)}")
    lines.append("正文：")
    lines.append(desc if desc else "（这篇笔记没有文字正文）")
    if comments:
        lines.append("")
        lines.append("精选评论：")
        for comment in comments:
            lines.append(f"- {comment['author']}（{comment['likes']}赞）：{comment['content']}")
    elif comment_note:
        lines.append("")
        lines.append(f"精选评论：{comment_note}")
    return "\n".join(lines)


async def fetch_note_detail(
    url_or_text: str,
    *,
    cookie: str,
    include_comments: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Resolve a shared Xiaohongshu link and fetch a compact text summary of the note."""
    cookie_str = (cookie or "").strip()
    if not cookie_str:
        return {
            "status": "error",
            "error_code": "missing_cookie",
            "message": _error_message("missing_cookie"),
        }
    ck = parse_cookies(cookie_str)
    if not ck.get("a1"):
        return {
            "status": "error",
            "error_code": "invalid_cookie",
            "message": _error_message("invalid_cookie"),
        }

    deadline = time.monotonic() + max(0.5, float(timeout))
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(2.0, timeout)),
            follow_redirects=False,
        )

    try:
        resolved = await resolve_xhs_link(
            url_or_text,
            client=client,
            timeout=max(0.5, deadline - time.monotonic()),
        )
        if resolved.get("ok") != "1":
            code = resolved.get("error_code", "invalid_url")
            return {
                "status": "error",
                "error_code": code,
                "message": _error_message(code),
            }

        note_id = resolved["note_id"]
        xsec_token = resolved.get("xsec_token", "")
        xsec_source = resolved.get("xsec_source", "pc_feed") or "pc_feed"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "status": "error",
                "error_code": "timeout",
                "message": _error_message("timeout"),
            }

        feed_payload = {
            "source_note_id": note_id,
            "image_formats": IMG_FORMATS,
            "extra": {"need_body_topic": "1"},
            "xsec_source": xsec_source,
            "xsec_token": xsec_token,
        }
        r = await _signed_post(
            client,
            EDITH,
            "/api/sns/web/v1/feed",
            feed_payload,
            cookie_str,
            ck,
            extra_headers={"xy-direction": "13"},
            timeout=remaining,
        )

        code = r.get("code")
        if str(code) == "-100":
            return {
                "status": "error",
                "error_code": "cookie_expired",
                "message": _error_message("cookie_expired"),
            }

        items = (r.get("data") or {}).get("items") if isinstance(r.get("data"), dict) else None
        first_item = items[0] if isinstance(items, list) and items else {}
        nc = first_item.get("note_card") if isinstance(first_item, dict) else None
        if not r.get("success") or not isinstance(nc, dict) or not nc:
            return {
                "status": "error",
                "error_code": "fetch_failed",
                "message": _error_message("fetch_failed"),
            }

        raw_title = str(nc.get("title") or nc.get("display_title") or "").strip() or "无标题"
        title = _truncate_text(raw_title, MAX_TITLE_CHARS)
        user_info = nc.get("user") if isinstance(nc.get("user"), dict) else {}
        author = str(user_info.get("nickname") or user_info.get("nick_name") or "").strip() or "未知作者"
        raw_desc = str(nc.get("desc") or nc.get("content") or "").strip()
        desc = _truncate_text(raw_desc, MAX_DESC_CHARS, "…（正文较长，已截断）")
        tags = _extract_tags(nc)

        note_type = str(nc.get("type") or "").strip().lower()
        image_list = nc.get("image_list") if isinstance(nc.get("image_list"), list) else []
        if note_type == "video":
            note_type_label = "视频笔记"
        elif image_list:
            note_type_label = f"图文笔记（共 {len(image_list)} 张图）"
        else:
            note_type_label = "图文笔记"

        comments: list[dict[str, str]] = []
        comment_status = "disabled"
        comment_note = ""
        if include_comments:
            comment_remaining = deadline - time.monotonic()
            if comment_remaining <= 0.5:
                comment_status = "skipped_timeout"
                comment_note = "请求耗时较长，评论区已超时跳过"
                logger.warning("xhs comment fetch skipped due to tight remaining budget note_id=%s", note_id)
            else:
                try:
                    cr = await _signed_get(
                        client,
                        EDITH,
                        "/api/sns/web/v2/comment/page",
                        {
                            "note_id": note_id,
                            "cursor": "",
                            "top_comment_id": "",
                            "image_formats": "jpg,webp,avif",
                            "xsec_token": xsec_token,
                        },
                        cookie_str,
                        ck,
                        extra_headers={"referer": f"{WWW}/explore/{note_id}"},
                        timeout=comment_remaining,
                    )
                    if not isinstance(cr, dict) or not cr.get("success"):
                        err_parts = []
                        if isinstance(cr, dict) and cr.get("http_status"):
                            err_parts.append(f"HTTP {cr['http_status']}")
                        if isinstance(cr, dict) and cr.get("code") is not None:
                            err_parts.append(f"code={cr['code']}")
                        if isinstance(cr, dict) and cr.get("msg"):
                            err_parts.append(str(cr["msg"])[:80])
                        diag = "，".join(err_parts) if err_parts else "接口未返回成功"
                        comment_status = "error"
                        comment_note = f"评论区暂时没能拉取到（{diag}）"
                        logger.warning("xhs comment fetch rejected note_id=%s diag=%s", note_id, diag)
                    else:
                        comments = _extract_top_comments(cr)
                        if comments:
                            comment_status = "ok"
                        else:
                            comment_status = "empty"
                            comment_note = "这篇笔记暂时还没有评论"
                except httpx.TimeoutException:
                    comment_status = "skipped_timeout"
                    comment_note = "评论区请求超时，已仅返回正文"
                    logger.warning("xhs comment fetch timed out note_id=%s", note_id)
                except Exception as exc:
                    comment_status = "error"
                    comment_note = "评论区拉取异常，已仅返回正文"
                    logger.warning("xhs comment fetch failed note_id=%s err=%s", note_id, exc)

        formatted_text = _format_note_text(
            title=title,
            author=author,
            note_type_label=note_type_label,
            tags=tags,
            desc=desc,
            comments=comments,
            comment_note=comment_note,
        )
        return {
            "status": "ok",
            "note_id": note_id,
            "title": title,
            "author": author,
            "note_type": note_type_label,
            "tags": tags,
            "desc": desc,
            "comments": comments,
            "comment_status": comment_status,
            "formatted_text": formatted_text,
        }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "error_code": "timeout",
            "message": _error_message("timeout"),
        }
    except Exception:
        return {
            "status": "error",
            "error_code": "fetch_failed",
            "message": _error_message("fetch_failed"),
        }
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def check_xhs_login(
    cookie: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Check whether a Xiaohongshu Web Cookie is currently logged in."""
    cookie_str = (cookie or "").strip()
    ck = parse_cookies(cookie_str)
    if not cookie_str or not ck.get("a1"):
        return {"logged_in": False, "error_code": "invalid_cookie"}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(2.0, timeout)),
            follow_redirects=False,
        )
    try:
        r = await _signed_get(
            client,
            EDITH,
            "/api/sns/web/v2/user/me",
            None,
            cookie_str,
            ck,
            timeout=timeout,
        )
        d = r.get("data") if isinstance(r.get("data"), dict) else {}
        logged_in = bool(
            r.get("success")
            and (d.get("user_id") or d.get("userId") or d.get("guest") is False)
        )
        return {
            "logged_in": logged_in,
            "nickname": str(d.get("nickname") or ""),
            "user_id": str(d.get("user_id") or d.get("userId") or ""),
            "red_id": str(d.get("red_id") or ""),
        }
    except httpx.TimeoutException:
        return {"logged_in": False, "error_code": "timeout"}
    except Exception:
        return {"logged_in": False, "error_code": "fetch_failed"}
    finally:
        if owns_client and client is not None:
            await client.aclose()
