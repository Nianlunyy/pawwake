"""Secret wish blind-box cards CRUD, draw, clear, and status routes."""

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse

import shared
from db import secret_cards as db_secret_cards

router = APIRouter()


@router.get("/api/secret-cards")
async def api_list_secret_cards(
    dimension: str | None = Query(default=None),
    enabled_only: bool = Query(default=False),
):
    try:
        cards = await db_secret_cards.list_secret_cards(
            dimension=dimension,
            enabled_only=enabled_only,
        )
        pool_counts = await db_secret_cards.get_pool_counts()
        return {
            "status": "ok",
            "cards": cards,
            "pool_counts": pool_counts,
        }
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return shared._api_failure("读取暗牌心愿池失败", cards=[])


@router.post("/api/secret-cards")
async def api_create_secret_card(body: dict = Body(default={})):
    try:
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": "请求体必须为 JSON 对象"})
        dimension = body.get("dimension")
        enabled = bool(body.get("enabled", True))
        if any(k in body for k in ("text", "lines", "batch")):
            raw_text = body.get("text") if body.get("text") is not None else (
                body.get("lines") if body.get("lines") is not None else body.get("batch")
            )
            result = await db_secret_cards.batch_import_secret_cards(
                dimension,
                raw_text,
                enabled=enabled,
            )
            return {"status": "ok", **result}

        card = await db_secret_cards.create_secret_card(
            dimension,
            body.get("content", ""),
            enabled=enabled,
        )
        pool_counts = await db_secret_cards.get_pool_counts()
        return {"status": "ok", "card": card, "pool_counts": pool_counts}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return shared._api_failure("新增暗牌词条失败")


@router.post("/api/secret-cards/batch-import")
async def api_batch_import_secret_cards(body: dict = Body(default={})):
    try:
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": "请求体必须为 JSON 对象"})
        dimension = body.get("dimension")
        raw_text = body.get("text") if body.get("text") is not None else (
            body.get("lines") if body.get("lines") is not None else ""
        )
        enabled = bool(body.get("enabled", True))
        result = await db_secret_cards.batch_import_secret_cards(
            dimension,
            raw_text,
            enabled=enabled,
        )
        return {"status": "ok", **result}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return shared._api_failure("批量导入暗牌失败")


@router.patch("/api/secret-cards/{card_id}")
@router.put("/api/secret-cards/{card_id}")
async def api_update_secret_card(card_id: int, body: dict = Body(default={})):
    try:
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": "请求体必须为 JSON 对象"})
        content = body.get("content") if "content" in body else None
        dimension = body.get("dimension") if "dimension" in body else None
        enabled = bool(body["enabled"]) if "enabled" in body and body["enabled"] is not None else None
        updated = await db_secret_cards.update_secret_card(
            card_id,
            content=content,
            dimension=dimension,
            enabled=enabled,
        )
        if not updated:
            return JSONResponse(status_code=404, content={"error": "暗牌词条不存在"})
        pool_counts = await db_secret_cards.get_pool_counts()
        return {"status": "ok", "card": updated, "pool_counts": pool_counts}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return shared._api_failure("更新暗牌词条失败")


@router.delete("/api/secret-cards/{card_id}")
async def api_delete_secret_card(card_id: int):
    try:
        deleted = await db_secret_cards.delete_secret_card(card_id)
        if not deleted:
            return JSONResponse(status_code=404, content={"error": "暗牌词条不存在"})
        pool_counts = await db_secret_cards.get_pool_counts()
        return {"status": "ok", "deleted": True, "id": card_id, "pool_counts": pool_counts}
    except Exception:
        return shared._api_failure("删除暗牌词条失败")


@router.post("/api/secret-cards/batch-delete")
@router.delete("/api/secret-cards")
async def api_batch_delete_secret_cards(
    body: dict = Body(default={}),
    dimension: str | None = Query(default=None),
    disabled_only: bool = Query(default=False),
):
    try:
        payload = body if isinstance(body, dict) else {}
        ids = payload.get("ids")
        dim = payload.get("dimension") if payload.get("dimension") is not None else dimension
        dis_only = bool(payload.get("disabled_only", disabled_only))
        deleted_count = await db_secret_cards.batch_delete_secret_cards(
            ids=ids,
            dimension=dim,
            disabled_only=dis_only,
        )
        pool_counts = await db_secret_cards.get_pool_counts()
        return {
            "status": "ok",
            "deleted_count": deleted_count,
            "pool_counts": pool_counts,
        }
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return shared._api_failure("批量清理暗牌失败")


@router.get("/api/secret-cards/status")
@router.get("/api/secret-deck/status")
async def api_secret_deck_status():
    try:
        status_view = await db_secret_cards.get_public_deck_status()
        return {"status": "ok", **status_view}
    except Exception:
        return shared._api_failure("获取暗牌盲盒状态失败")


@router.post("/api/secret-cards/draw")
@router.post("/api/secret-deck/draw")
async def api_secret_deck_draw(body: dict = Body(default={})):
    """前端阿狸主动点单发牌：最高优先级，覆盖并刷新 2 小时有效期，且严禁返回具体词条 content。"""
    try:
        payload = body if isinstance(body, dict) else {}
        ratios = payload.get("ratios")
        if ratios is None and any(k in payload for k in db_secret_cards.VALID_DIMENSIONS):
            ratios = {k: payload.get(k, 0) for k in db_secret_cards.VALID_DIMENSIONS}
        raw_count = payload.get("count", 2)
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            count = int(raw_count or 2)
        else:
            count = raw_count

        drawn_result = await db_secret_cards.draw_secret_cards(
            ratios=ratios,
            count=count,
            source="user",
            force_redraw=True,
        )
        pool_counts = await db_secret_cards.get_pool_counts()
        sanitized = db_secret_cards.sanitize_hand_for_frontend(
            drawn_result if drawn_result.get("active") else None,
            current_round=drawn_result.get("current_round", 0),
            pool_counts=pool_counts,
        )
        return {"status": "ok", **sanitized}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return shared._api_failure("暗牌发牌失败")


@router.post("/api/secret-cards/clear")
@router.post("/api/secret-deck/clear")
@router.delete("/api/secret-cards/hand")
async def api_secret_deck_clear():
    try:
        sanitized = await db_secret_cards.clear_active_hand()
        return {"status": "ok", **sanitized}
    except Exception:
        return shared._api_failure("清空暗牌手牌失败")


@router.post("/api/secret-cards/ai-draw")
@router.post("/api/secret-deck/ai-draw")
async def api_secret_deck_ai_draw(body: dict = Body(default={})):
    """AI 工具抽牌入口：受 2 小时防打架守卫保护，若有效期内直接原样返回生效手牌。"""
    try:
        payload = body if isinstance(body, dict) else {}
        ratios = payload.get("ratios")
        if ratios is None and any(k in payload for k in db_secret_cards.VALID_DIMENSIONS):
            ratios = {k: payload.get(k, 0) for k in db_secret_cards.VALID_DIMENSIONS}
        raw_count = payload.get("count", 2)
        count = int(raw_count) if raw_count is not None else 2
        result = await db_secret_cards.draw_secret_cards(
            ratios=ratios,
            count=count,
            source="ai",
            force_redraw=False,
        )
        return {"status": "ok", **result}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return shared._api_failure("AI 抽暗牌失败")


@router.post("/api/secret-cards/sync")
async def api_secret_deck_sync(body: dict = Body(default={})):
    try:
        payload = body if isinstance(body, dict) else {}
        sanitized = await db_secret_cards.sync_deck_from_client(payload)
        return {"status": "ok", **sanitized}
    except Exception:
        return shared._api_failure("同步暗牌状态失败")
