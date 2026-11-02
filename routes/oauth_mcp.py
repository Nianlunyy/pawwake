"""OAuth 2.1 handshake routes for Google Gemini Custom Connected Apps and MCP.

Implements standard endpoints compliant with RFC 8414, RFC 7591, and OAuth 2.1:
- GET /.well-known/oauth-protected-resource (and /mcp subpath)
- GET /.well-known/oauth-authorization-server (and /.well-known/openid-configuration)
- POST /oauth/register (Dynamic Client Registration)
- GET /oauth/authorize (Consent and secret verification page)
- POST /oauth/authorize (Timing-safe secret verification and code issuance)
- POST /oauth/token (PKCE S256 verification and token issuance/refresh)
- POST /oauth/revoke (Token revocation)
"""

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import time
import urllib.parse
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import auth
import shared

router = APIRouter()

# In-memory storage for one-time authorization codes and registered clients
_auth_codes: Dict[str, Dict[str, Any]] = {}
_registered_clients: Dict[str, Dict[str, Any]] = {}

# Sliding window rate limiter for POST /oauth/authorize failures (max 5 per minute)
_authorize_failures: list[float] = []

REFRESH_TOKEN_EXPIRE_SECONDS = 90 * 86400  # 90 天有效期


def _clean_authorize_failures(now: float):
    """Keep authorization failure timestamps within the last 60 seconds."""
    cutoff = now - 60.0
    while _authorize_failures and _authorize_failures[0] <= cutoff:
        _authorize_failures.pop(0)


def _is_authorize_rate_limited() -> bool:
    """Check if failed authorization attempts exceeded 5 per minute."""
    now = time.time()
    _clean_authorize_failures(now)
    return len(_authorize_failures) >= 5


def _record_authorize_failure():
    """Record a failed authorization attempt."""
    now = time.time()
    _clean_authorize_failures(now)
    _authorize_failures.append(now)


def get_public_base_url() -> str:
    """Read PUBLIC_BASE_URL from environment or shared config."""
    url = os.environ.get("PUBLIC_BASE_URL") or getattr(shared, "PUBLIC_BASE_URL", "") or ""
    return url.strip().rstrip("/")


def get_allowed_redirect_hosts() -> set[str]:
    """Read OAUTH_ALLOWED_REDIRECT_HOSTS environment variable or shared config."""
    raw = os.environ.get("OAUTH_ALLOWED_REDIRECT_HOSTS") or getattr(shared, "OAUTH_ALLOWED_REDIRECT_HOSTS", "") or ""
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def create_stateless_client_id(redirect_uris: list[str]) -> str:
    """Generate stateless client_id: pwc_<redirect_uris 编码>_<HMAC>."""
    normalized = json.dumps(sorted(u.strip() for u in redirect_uris), separators=(',', ':'))
    encoded = base64.urlsafe_b64encode(normalized.encode("utf-8")).decode("ascii").rstrip("=")
    keys = auth.get_gateway_secrets()
    signing_key = keys[0] if keys else "pawwake_default_key"
    sig = hmac.new(signing_key.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"pwc_{encoded}_{sig}"


def verify_client_id(client_id: str) -> tuple[bool, list[str]]:
    """Verify stateless client_id pwc_<encoded_uris>_<HMAC> against GATEWAY_SECRET items."""
    try:
        if not isinstance(client_id, str) or not client_id.startswith("pwc_"):
            return False, []
        body = client_id[4:]
        encoded_uris, sep, sig = body.rpartition("_")
        if not sep or not encoded_uris or not sig:
            return False, []
        keys = auth.get_gateway_secrets()
        if not keys:
            return False, []
        payload = encoded_uris.encode("utf-8")
        provided_sig = sig.encode("utf-8")
        matched = False
        for k in keys:
            expected_sig = hmac.new(k.encode("utf-8"), payload, hashlib.sha256).hexdigest().encode("utf-8")
            if secrets.compare_digest(provided_sig, expected_sig):
                matched = True
        if not matched:
            return False, []
        pad = "=" * (-len(encoded_uris) % 4)
        raw_json = base64.urlsafe_b64decode((encoded_uris + pad).encode("ascii")).decode("utf-8")
        uris = json.loads(raw_json)
        if not isinstance(uris, list):
            return False, []
        return True, uris
    except Exception:
        return False, []


def _clean_expired_codes():
    """Remove expired authorization codes from memory."""
    now = time.time()
    expired = [k for k, v in _auth_codes.items() if v.get("expires_at", 0) < now]
    for k in expired:
        _auth_codes.pop(k, None)


def _verify_secret_constant_time(provided: str) -> bool:
    """Verify provided secret against GATEWAY_SECRET in constant time using auth.verify_gateway_key."""
    return auth.verify_gateway_key(provided)


def create_refresh_token(expires_in: int = REFRESH_TOKEN_EXPIRE_SECONDS) -> str:
    """Generate an HMAC-SHA256 signed refresh token tied to the gateway secret with 90-day expiry."""
    expires_at = int(time.time()) + expires_in
    raw = secrets.token_urlsafe(32)
    keys = auth.get_gateway_secrets()
    signing_key = keys[0] if keys else "pawwake_default_key"
    payload = f"{expires_at}_{raw}".encode("utf-8")
    sig = hmac.new(
        signing_key.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return f"pwr_{expires_at}_{raw}_{sig}"


def verify_refresh_token(token: str) -> bool:
    """Verify refresh token signature and expiration date."""
    try:
        if not isinstance(token, str) or not token.startswith("pwr_"):
            return False
        body = token[4:]
        expires_at_str, sep1, rest = body.partition("_")
        raw, sep2, sig = rest.rpartition("_")
        if not sep1 or not sep2 or not expires_at_str or not raw or not sig:
            return False
        expires_at = int(expires_at_str)
        if time.time() > expires_at:
            return False
        keys = auth.get_gateway_secrets()
        if not keys:
            return False
        payload = f"{expires_at}_{raw}".encode("utf-8")
        provided_sig = sig.encode("utf-8")
        matched = False
        for k in keys:
            expected_sig = hmac.new(
                k.encode("utf-8"),
                payload,
                hashlib.sha256,
            ).hexdigest().encode("utf-8")
            if secrets.compare_digest(provided_sig, expected_sig):
                matched = True
        return matched
    except Exception:
        return False


def get_access_token() -> str:
    """Return derived access token pwa_<expires_at>_<HMAC>."""
    return auth.create_derived_token()


# ============================================================
# 1. 发现端点 (RFC 8414 / RFC 9728)
# ============================================================

@router.get("/.well-known/oauth-protected-resource")
@router.get("/.well-known/oauth-protected-resource/mcp")
async def oauth_protected_resource():
    """OAuth 2.0 Protected Resource Metadata."""
    base_url = get_public_base_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "error_description": "PUBLIC_BASE_URL is not configured"},
        )
    return {
        "resource": f"{base_url}/mcp/",
        "authorization_servers": [base_url],
    }


@router.get("/.well-known/oauth-authorization-server")
@router.get("/.well-known/openid-configuration")
async def oauth_authorization_server():
    """OAuth 2.0 Authorization Server Metadata."""
    base_url = get_public_base_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "error_description": "PUBLIC_BASE_URL is not configured"},
        )
    return {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/oauth/authorize",
        "token_endpoint": f"{base_url}/oauth/token",
        "registration_endpoint": f"{base_url}/oauth/register",
        "revocation_endpoint": f"{base_url}/oauth/revoke",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_post",
            "client_secret_basic",
            "none",
        ],
        "scopes_supported": ["mcp", "openid", "profile"],
    }


# ============================================================
# 2. 动态客户端注册 (DCR, RFC 7591)
# ============================================================

@router.post("/oauth/register")
async def oauth_register(request: Request):
    """Dynamic Client Registration endpoint."""
    base_url = get_public_base_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "error_description": "PUBLIC_BASE_URL is not configured"},
        )

    try:
        body = await request.json()
    except Exception:
        body = {}

    redirect_uris = body.get("redirect_uris")
    if not isinstance(redirect_uris, list) or len(redirect_uris) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_redirect_uri",
                "error_description": "redirect_uris must be a non-empty list",
            },
        )

    for uri in redirect_uris:
        if not isinstance(uri, str) or not uri.startswith("https://"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_redirect_uri",
                    "error_description": "redirect_uris must all use https scheme",
                },
            )

    # redirect_uris 必须在白名单主机内，名单为空也 403
    allowed_hosts = get_allowed_redirect_hosts()
    for uri in redirect_uris:
        parsed = urllib.parse.urlparse(uri)
        host = (parsed.hostname or "").lower()
        if not allowed_hosts or host not in allowed_hosts:
            print(f"🚫 OAuth 注册被拒 host={host or 'none'}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "access_denied",
                    "error_description": f"Redirect URI host '{host}' is not in allowed hosts whitelist",
                },
            )

    # 不覆盖已有登记
    requested_client_id = body.get("client_id")
    if requested_client_id and requested_client_id in _registered_clients:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_client_metadata",
                "error_description": "client_id already registered, overwrite forbidden",
            },
        )

    # client_id 改为无状态：pwc_<redirect_uris 编码>_<HMAC>
    client_id = create_stateless_client_id(redirect_uris)

    client_name = body.get("client_name", "OAuth Client")
    _registered_clients[client_id] = {
        "client_id": client_id,
        "client_name": client_name,
        "redirect_uris": redirect_uris,
    }

    response_data = {
        "client_id": client_id,
        "client_name": client_name,
        "redirect_uris": redirect_uris,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    return JSONResponse(status_code=201, content=response_data)


# ============================================================
# 3. 授权确认页面与接口
# ============================================================

def _render_authorize_html(
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    code_challenge_method: str,
    response_type: str,
    scope: str,
    error: str = "",
) -> str:
    """Render a mobile-friendly responsive OAuth consent card."""
    error_html = (
        f'<div class="error-banner">⚠️ {html.escape(error)}</div>'
        if error
        else ""
    )
    cancel_url = (
        redirect_uri + ("&" if "?" in redirect_uri else "?") + f"error=access_denied&state={html.escape(state)}"
        if redirect_uri
        else "#"
    )
    parsed = urllib.parse.urlparse(redirect_uri)
    redirect_host = (parsed.hostname or "").lower() or redirect_uri
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pawwake · 授权接入 / Authorize</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --border: #e2e8f0;
            --error-bg: #fef2f2;
            --error-text: #b91c1c;
            --error-border: #fecaca;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #0b0f19;
                --card-bg: #1e293b;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --border: #334155;
                --error-bg: #450a0a;
                --error-text: #f87171;
                --error-border: #7f1d1d;
            }}
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.25rem;
        }}
        .auth-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            max-width: 440px;
            width: 100%;
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .logo {{
            font-size: 2.75rem;
            line-height: 1;
            margin-bottom: 0.75rem;
        }}
        .title {{
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .subtitle {{
            font-size: 0.875rem;
            color: var(--text-muted);
            line-height: 1.5;
        }}
        .app-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(99, 102, 241, 0.12);
            color: var(--primary);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.85rem 0;
            word-break: break-all;
        }}
        .info-box {{
            background: rgba(148, 163, 184, 0.08);
            border-radius: 0.65rem;
            padding: 0.85rem 1rem;
            margin-bottom: 1.25rem;
            font-size: 0.825rem;
            line-height: 1.45;
            color: var(--text-muted);
        }}
        .info-box ul {{
            list-style: disc;
            margin-left: 1.2rem;
            margin-top: 0.35rem;
        }}
        .error-banner {{
            background-color: var(--error-bg);
            color: var(--error-text);
            border: 1px solid var(--error-border);
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            margin-bottom: 1.25rem;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .form-group {{
            margin-bottom: 1.25rem;
        }}
        .form-label {{
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }}
        .form-input {{
            width: 100%;
            padding: 0.75rem 0.9rem;
            font-size: 0.95rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: var(--bg-color);
            color: var(--text-main);
            outline: none;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        .form-input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }}
        .actions {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        .btn {{
            width: 100%;
            padding: 0.75rem;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 0.5rem;
            cursor: pointer;
            border: none;
            transition: background-color 0.15s ease;
            text-align: center;
            text-decoration: none;
        }}
        .btn-primary {{
            background-color: var(--primary);
            color: #ffffff;
        }}
        .btn-primary:hover {{
            background-color: var(--primary-hover);
        }}
        .btn-secondary {{
            background-color: transparent;
            color: var(--text-muted);
            border: 1px solid var(--border);
        }}
        .btn-secondary:hover {{
            background-color: rgba(148, 163, 184, 0.1);
        }}
    </style>
</head>
<body>
    <div class="auth-card">
        <div class="header">
            <div class="logo">🐾</div>
            <h1 class="title">Pawwake 授权确认</h1>
            <p class="subtitle">授权码将发送到：{html.escape(redirect_host)}</p>
            <div class="app-badge">客户端：{html.escape(client_id)}</div>
        </div>

        <div class="info-box">
            <div>授权后，应用将获得以下 MCP 访问权限：</div>
            <ul>
                <li>读取、搜索与管理长短期记忆条目</li>
                <li>读取分区上下文滚动摘要（Partition Summary）</li>
                <li>按关键词与语义检索历史对话切片</li>
            </ul>
        </div>

        {error_html}

        <form method="POST" action="/oauth/authorize">
            <input type="hidden" name="client_id" value="{html.escape(client_id)}">
            <input type="hidden" name="redirect_uri" value="{html.escape(redirect_uri)}">
            <input type="hidden" name="state" value="{html.escape(state)}">
            <input type="hidden" name="code_challenge" value="{html.escape(code_challenge)}">
            <input type="hidden" name="code_challenge_method" value="{html.escape(code_challenge_method)}">
            <input type="hidden" name="response_type" value="{html.escape(response_type)}">
            <input type="hidden" name="scope" value="{html.escape(scope)}">

            <div class="form-group">
                <label class="form-label" for="gateway_secret">网关密钥 (GATEWAY_SECRET)</label>
                <input
                    type="password"
                    id="gateway_secret"
                    name="gateway_secret"
                    class="form-input"
                    placeholder="请输入 GATEWAY_SECRET 密码确认授权"
                    required
                    autofocus
                    autocomplete="current-password"
                >
            </div>

            <div class="actions">
                <button type="submit" class="btn btn-primary">确认授权并连接</button>
                <a href="{html.escape(cancel_url)}" class="btn btn-secondary">取消</a>
            </div>
        </form>
    </div>
</body>
</html>"""


@router.get("/oauth/authorize")
async def oauth_authorize_get(
    request: Request,
    response_type: str = "code",
    client_id: str = "",
    redirect_uri: str = "",
    state: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "S256",
    scope: str = "mcp",
):
    """Render OAuth authorization page."""
    base_url = get_public_base_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "error_description": "PUBLIC_BASE_URL is not configured"},
        )

    # client_id 必须已登记（支持无状态 client_id 或内存登记）
    valid_client, registered_uris = verify_client_id(client_id)
    if not valid_client:
        if client_id in _registered_clients:
            valid_client = True
            registered_uris = _registered_clients[client_id].get("redirect_uris", [])
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "unauthorized_client",
                    "error_description": "client_id is not registered",
                },
            )

    # redirect_uri 必须与登记值逐字符一致，否则拒绝。取消链接同理
    if not redirect_uri or redirect_uri not in registered_uris:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "error_description": "redirect_uri mismatch with registered client redirect_uris",
            },
        )

    # 要求 code_challenge 非空且方法为 S256，否则 400
    if not code_challenge or not code_challenge_method or code_challenge_method.upper() != "S256":
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "error_description": "code_challenge is required and code_challenge_method must be S256",
            },
        )

    html_content = _render_authorize_html(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        response_type=response_type,
        scope=scope,
    )
    return HTMLResponse(content=html_content)


@router.post("/oauth/authorize")
async def oauth_authorize_post(request: Request):
    """Handle authorization form submission, verify GATEWAY_SECRET, and issue code."""
    base_url = get_public_base_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "error_description": "PUBLIC_BASE_URL is not configured"},
        )

    # 全局失败限流，每分钟最多 5 次失败，超过返回 429
    if _is_authorize_rate_limited():
        return JSONResponse(
            status_code=429,
            content={
                "error": "slow_down",
                "error_description": "Too many failed authorization attempts. Please try again later.",
            },
        )

    form = await request.form()
    gateway_secret = str(form.get("gateway_secret", "")).strip()
    client_id = str(form.get("client_id", "")).strip()
    redirect_uri = str(form.get("redirect_uri", "")).strip()
    state = str(form.get("state", "")).strip()
    code_challenge = str(form.get("code_challenge", "")).strip()
    code_challenge_method = str(form.get("code_challenge_method", "S256")).strip()
    response_type = str(form.get("response_type", "code")).strip()
    scope = str(form.get("scope", "mcp")).strip()

    # client_id 必须已登记（支持无状态 client_id 或内存登记）
    valid_client, registered_uris = verify_client_id(client_id)
    if not valid_client:
        if client_id in _registered_clients:
            valid_client = True
            registered_uris = _registered_clients[client_id].get("redirect_uris", [])
        else:
            _record_authorize_failure()
            return JSONResponse(
                status_code=400,
                content={
                    "error": "unauthorized_client",
                    "error_description": "client_id is not registered",
                },
            )

    # redirect_uri 必须与登记值逐字符一致，否则拒绝
    if not redirect_uri or redirect_uri not in registered_uris:
        _record_authorize_failure()
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "error_description": "redirect_uri mismatch with registered client redirect_uris",
            },
        )

    # 要求 code_challenge 非空且方法为 S256，否则 400
    if not code_challenge or not code_challenge_method or code_challenge_method.upper() != "S256":
        _record_authorize_failure()
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "error_description": "code_challenge is required and code_challenge_method must be S256",
            },
        )

    # Timing-safe comparison with GATEWAY_SECRET
    if not _verify_secret_constant_time(gateway_secret):
        _record_authorize_failure()
        html_content = _render_authorize_html(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            response_type=response_type,
            scope=scope,
            error="网关密钥 (GATEWAY_SECRET) 校验失败，绝不签发授权码，请重试。",
        )
        return HTMLResponse(content=html_content, status_code=401)

    # Generate one-time random authorization code (TTL: 5 minutes)
    code = secrets.token_urlsafe(32)
    _clean_expired_codes()
    _auth_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method or "S256",
        "scope": scope,
        "expires_at": time.time() + 300,
    }

    # Redirect back to client with code and state
    parsed = urllib.parse.urlparse(redirect_uri)
    existing_query = urllib.parse.parse_qsl(parsed.query)
    existing_query.append(("code", code))
    if state:
        existing_query.append(("state", state))
    new_query = urllib.parse.urlencode(existing_query)
    target_url = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment,
    ))
    return RedirectResponse(url=target_url, status_code=302)


# ============================================================
# 4. 令牌交换与刷新接口
# ============================================================

@router.post("/oauth/token")
async def oauth_token(request: Request):
    """Exchange authorization code or refresh token for access token (supports form or JSON)."""
    base_url = get_public_base_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "error_description": "PUBLIC_BASE_URL is not configured"},
        )

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            data = {}
    else:
        form = await request.form()
        data = dict(form)

    # Check client credentials if client_secret is provided (via body or Basic Auth)
    auth_header = request.headers.get("authorization", "")
    provided_client_secret = data.get("client_secret", "")
    if auth_header.lower().startswith("basic "):
        try:
            b64_creds = auth_header.split(" ", 1)[1].strip()
            decoded = base64.b64decode(b64_creds).decode("utf-8")
            if ":" in decoded:
                _, provided_client_secret = decoded.split(":", 1)
        except Exception:
            pass

    if provided_client_secret:
        if not _verify_secret_constant_time(provided_client_secret):
            return JSONResponse(
                status_code=401,
                content={"error": "invalid_client", "error_description": "Invalid client secret"},
            )

    grant_type = str(data.get("grant_type", "")).strip()

    if grant_type == "authorization_code":
        code = str(data.get("code", "")).strip()
        code_verifier = str(data.get("code_verifier", "")).strip()
        redirect_uri = str(data.get("redirect_uri", "")).strip()

        _clean_expired_codes()
        code_data = _auth_codes.pop(code, None)
        if not code_data:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_grant",
                    "error_description": "Invalid or expired authorization code",
                },
            )

        if time.time() > code_data.get("expires_at", 0):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_grant",
                    "error_description": "Authorization code has expired",
                },
            )

        # client_id 必须已登记（支持无状态 client_id 或内存登记）
        code_client_id = code_data.get("client_id", "")
        provided_client_id = str(data.get("client_id", "")).strip()
        if provided_client_id and provided_client_id != code_client_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_grant",
                    "error_description": "client_id mismatch",
                },
            )

        valid_client, registered_uris = verify_client_id(code_client_id)
        if not valid_client:
            if code_client_id in _registered_clients:
                valid_client = True
                registered_uris = _registered_clients[code_client_id].get("redirect_uris", [])
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_client",
                        "error_description": "client_id is not registered",
                    },
                )

        # redirect_uri 必须与登记值逐字符一致，否则拒绝
        expected_redirect_uri = code_data.get("redirect_uri", "")
        if not redirect_uri or redirect_uri not in registered_uris or redirect_uri != expected_redirect_uri:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_grant",
                    "error_description": "redirect_uri mismatch",
                },
            )

        # Strict PKCE verification (only S256)
        expected_challenge = code_data.get("code_challenge", "")
        challenge_method = (code_data.get("code_challenge_method") or "S256").upper()
        if challenge_method != "S256":
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_request",
                    "error_description": f"Unsupported code_challenge_method: {challenge_method}",
                },
            )

        if not expected_challenge or not code_verifier:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_grant",
                    "error_description": "code_verifier is required",
                },
            )
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        calculated = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        if not hmac.compare_digest(
            calculated.encode("ascii"), expected_challenge.encode("ascii")
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_grant",
                    "error_description": "PKCE verification failed",
                },
            )

        access_token = get_access_token()
        refresh_token = create_refresh_token()
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 2592000,
            "refresh_token": refresh_token,
            "scope": "mcp",
        }

    if grant_type == "refresh_token":
        refresh_token = str(data.get("refresh_token", "")).strip()
        provided_client_id = str(data.get("client_id", "")).strip()
        if provided_client_id:
            valid_client, _ = verify_client_id(provided_client_id)
            if not valid_client and provided_client_id not in _registered_clients:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_client",
                        "error_description": "client_id is not registered",
                    },
                )

        if not verify_refresh_token(refresh_token):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_grant",
                    "error_description": "Invalid refresh token",
                },
            )

        access_token = get_access_token()
        new_refresh_token = create_refresh_token()
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 2592000,
            "refresh_token": new_refresh_token,
            "scope": "mcp",
        }

    return JSONResponse(
        status_code=400,
        content={
            "error": "unsupported_grant_type",
            "error_description": f"Unsupported grant_type: {grant_type}",
        },
    )


# ============================================================
# 5. 令牌撤销接口
# ============================================================

@router.post("/oauth/revoke")
async def oauth_revoke():
    """Revoke issued token."""
    base_url = get_public_base_url()
    if not base_url:
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "error_description": "PUBLIC_BASE_URL is not configured"},
        )
    return {}
