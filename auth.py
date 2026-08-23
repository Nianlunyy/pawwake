"""Gateway header and Dashboard cookie authentication."""

import base64
import hashlib
import hmac
import secrets
import time

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

import shared

# ============================================================
# 网关鉴权中间件
# ============================================================

# 不需要鉴权的路径（根路径精确匹配，其余按前缀匹配）
PUBLIC_PATHS = ("/", "/static/", "/health", "/favicon.ico")
DASHBOARD_PATH_PREFIXES = ("/api/", "/import/", "/export/")
DASHBOARD_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def dashboard_auth_ready() -> bool:
    return bool(shared.DASHBOARD_PASSWORD and len(shared.SESSION_SECRET) >= 32)


def is_dashboard_path(path: str) -> bool:
    return path == "/dashboard" or path.startswith("/dashboard/") or path.startswith(DASHBOARD_PATH_PREFIXES)


def make_dashboard_session() -> str:
    expires_at = int(time.time()) + shared.DASHBOARD_SESSION_SECONDS
    payload = str(expires_at).encode("ascii")
    signature = hmac.new(shared.SESSION_SECRET.encode("utf-8"), payload, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{expires_at}.{encoded_signature}"


def valid_dashboard_session(token: str) -> bool:
    if not dashboard_auth_ready():
        return False
    try:
        expires_text, provided_signature = token.split(".", 1)
        expires_at = int(expires_text)
    except (TypeError, ValueError):
        return False
    if expires_at < int(time.time()):
        return False
    payload = expires_text.encode("ascii")
    expected = hmac.new(shared.SESSION_SECRET.encode("utf-8"), payload, hashlib.sha256).digest()
    expected_signature = base64.urlsafe_b64encode(expected).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(provided_signature, expected_signature)


def request_has_same_origin(request: Request) -> bool:
    origin = request.headers.get("origin", "").rstrip("/")
    forwarded_scheme = request.headers.get("x-forwarded-proto", "").strip().lower()
    expected_scheme = forwarded_scheme if forwarded_scheme in {"http", "https"} else request.url.scheme
    expected = f"{expected_scheme}://{request.url.netloc}".rstrip("/")
    return bool(origin) and secrets.compare_digest(origin, expected)


def require_database_enabled():
    if not shared.DATABASE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Database is disabled. Set DATABASE_ENABLED=true and restart Pawwake.",
        )

async def gateway_auth_middleware(request: Request, call_next):
    """API 使用请求头鉴权，Dashboard 使用独立的签名 Cookie。"""
    path = request.url.path

    # 公开路径不需要鉴权（根路径精确匹配）
    if path == "/":
        return await call_next(request)
    for prefix in PUBLIC_PATHS[1:]:
        if path.startswith(prefix):
            return await call_next(request)

    if path == "/dashboard/login":
        return await call_next(request)

    # OPTIONS 预检请求放行（CORS 需要）
    if request.method == "OPTIONS":
        return await call_next(request)

    # 程序调用只接受请求头，不让主密钥进入 URL、浏览器历史或前端脚本。
    provided_key = request.headers.get("X-Gateway-Key", "")
    if shared.GATEWAY_SECRET and secrets.compare_digest(provided_key, shared.GATEWAY_SECRET):
        return await call_next(request)

    # 显式记忆写入只接受网关密钥，Dashboard Cookie 不授予这个外部写入口。
    if path == "/api/memories" and request.method == "POST":
        if not shared.GATEWAY_SECRET:
            return JSONResponse(
                status_code=503,
                content={"error": "GATEWAY_SECRET is required for this endpoint."},
            )
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized. Provide X-Gateway-Key header."},
        )

    if is_dashboard_path(path) and (shared.GATEWAY_SECRET or shared.DASHBOARD_PASSWORD or shared.SESSION_SECRET):
        if not dashboard_auth_ready():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Dashboard login is not configured. Set DASHBOARD_PASSWORD and a stable SESSION_SECRET of at least 32 characters."
                },
            )
        token = request.cookies.get(shared.DASHBOARD_SESSION_COOKIE, "")
        if valid_dashboard_session(token):
            if request.method in DASHBOARD_WRITE_METHODS and not request_has_same_origin(request):
                return JSONResponse(status_code=403, content={"error": "Invalid request origin."})
            return await call_next(request)
        if path == "/dashboard" and request.method == "GET":
            return RedirectResponse(url="/dashboard/login", status_code=303)
        return JSONResponse(
            status_code=401,
            content={"error": "Dashboard login required."},
        )

    # 未设置密钥时保留旧部署行为，但明确提示公网部署风险。
    if not shared.GATEWAY_SECRET:
        if not hasattr(gateway_auth_middleware, "_warned"):
            print("⚠️  GATEWAY_SECRET 未设置！API 端点不受保护！")
            print("⚠️  请在环境变量中设置 GATEWAY_SECRET 以启用 API 鉴权")
            gateway_auth_middleware._warned = True
        return await call_next(request)

    return JSONResponse(
        status_code=401,
        content={"error": "Unauthorized. Provide X-Gateway-Key header."},
    )
