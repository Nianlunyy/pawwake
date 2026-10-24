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

_warned_no_secret = False


def _warn_missing_secret_once():
    """未设置密钥时启动或首次请求打印一次不含值的警告日志。"""
    global _warned_no_secret
    if not _warned_no_secret:
        _warned_no_secret = True
        print("⚠️  GATEWAY_SECRET 未设置！网关保护端点将拒绝所有未授权请求！")
        print("⚠️  请在环境变量中设置 GATEWAY_SECRET 以启用 API 鉴权")


def get_gateway_secrets() -> list[str]:
    """支持逗号分隔的多个 Key：按逗号拆分、strip、丢弃空项。"""
    raw = getattr(shared, "GATEWAY_SECRET", "") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def verify_gateway_key(provided_key: str) -> bool:
    """X-Gateway-Key 与每一项都用 secrets.compare_digest 比较，且不因命中而提前跳出（遍历全部再汇总）。"""
    keys = get_gateway_secrets()
    if not keys or not provided_key:
        return False
    matched = False
    for expected in keys:
        if secrets.compare_digest(provided_key, expected):
            matched = True
    return matched


if not get_gateway_secrets():
    _warn_missing_secret_once()


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
    path = request.scope["path"]

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
    if verify_gateway_key(provided_key):
        return await call_next(request)

    # 显式记忆写入只接受网关密钥，Dashboard Cookie 不授予这个外部写入口。
    if path == "/api/memories" and request.method == "POST":
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized. Provide X-Gateway-Key header."},
        )

    if is_dashboard_path(path) and (get_gateway_secrets() or shared.DASHBOARD_PASSWORD or shared.SESSION_SECRET):
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

    # 未设置密钥时网关保护路径一律拒绝（返回 401），并确保警告已打印。
    if not get_gateway_secrets():
        _warn_missing_secret_once()

    return JSONResponse(
        status_code=401,
        content={"error": "Unauthorized. Provide X-Gateway-Key header."},
    )
