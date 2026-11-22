"""Secret wish blind-box cards CRUD, round-based cooldown draw, and deck state."""

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from db import core as db_core

VALID_DIMENSIONS = ("position", "play", "location", "prop")
DIMENSION_LABELS = {
    "position": "体位",
    "play": "玩法",
    "location": "地点",
    "prop": "道具",
}
ACTIVE_HAND_TTL_SECONDS = 12 * 3600
COOLDOWN_ROUNDS = 8


def _utc_now(now_utc: Optional[datetime] = None) -> datetime:
    dt = now_utc or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return _utc_now(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return _utc_now(parsed)
    except (TypeError, ValueError):
        return None


def normalize_dimension(dimension: str) -> str:
    dim = str(dimension or "").strip().lower()
    if dim not in VALID_DIMENSIONS:
        raise ValueError(f"无效的维度: {dimension!r}，必须是 {VALID_DIMENSIONS} 之一")
    return dim


def normalize_ratios(ratios: Optional[dict]) -> Optional[dict[str, int]]:
    if not isinstance(ratios, dict):
        return None
    normalized: dict[str, int] = {}
    total = 0
    for dim in VALID_DIMENSIONS:
        raw_val = ratios.get(dim)
        if raw_val is None or raw_val == "":
            val = 0
        else:
            if isinstance(raw_val, bool):
                raise ValueError(f"维度 {dim} 的张数不能为布尔值")
            try:
                val = int(raw_val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"维度 {dim} 的张数必须是整数") from exc
            if val < 0:
                val = 0
        normalized[dim] = min(20, val)
        total += normalized[dim]
    return normalized if total > 0 else None


def parse_batch_lines(raw_text: str | list[str]) -> list[str]:
    """按换行符拆分并去除空白与批内重复项（保持首次出现顺序）。"""
    if isinstance(raw_text, list):
        raw_lines = [str(item) for item in raw_text]
    else:
        raw_lines = str(raw_text or "").splitlines()
    seen: set[str] = set()
    items: list[str] = []
    for raw in raw_lines:
        cleaned = raw.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(cleaned)
    return items


def select_cards_for_round(
    candidates: list[dict],
    count: int,
    current_round: int,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """按局数冷却优先筛选暗牌；池量不足时按 last_drawn_round 升序兜底补齐。"""
    if count <= 0 or not candidates:
        return []
    enabled_cards = [dict(card) for card in candidates if card.get("enabled", True)]
    if not enabled_cards:
        return []

    rand = rng if rng is not None else random
    preferred: list[dict] = []
    fallback: list[dict] = []
    for card in enabled_cards:
        last_round = int(card.get("last_drawn_round") or 0)
        if last_round == 0 or (current_round - last_round) > COOLDOWN_ROUNDS:
            preferred.append(card)
        else:
            fallback.append(card)

    preferred_shuffled = list(preferred)
    rand.shuffle(preferred_shuffled)
    if len(preferred_shuffled) >= count:
        return preferred_shuffled[:count]

    need = count - len(preferred_shuffled)
    fallback_ordered = list(fallback)
    rand.shuffle(fallback_ordered)
    fallback_ordered.sort(key=lambda c: int(c.get("last_drawn_round") or 0))
    return preferred_shuffled + fallback_ordered[:need]


def _coerce_hand(raw_hand: Any) -> Optional[dict]:
    if not raw_hand:
        return None
    if isinstance(raw_hand, str):
        try:
            raw_hand = json.loads(raw_hand)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(raw_hand, dict):
        return None
    return dict(raw_hand)


def is_hand_active(active_hand: Any, now_utc: Optional[datetime] = None) -> bool:
    hand = _coerce_hand(active_hand)
    if not hand:
        return False
    cards = hand.get("cards")
    if not isinstance(cards, list) or len(cards) == 0:
        return False
    now = _utc_now(now_utc)
    expires_dt = _parse_iso_dt(hand.get("expires_at"))
    if expires_dt is None:
        exp_ms = hand.get("expires_at_ms")
        if isinstance(exp_ms, (int, float)) and exp_ms > 0:
            expires_dt = datetime.fromtimestamp(exp_ms / 1000.0, tz=timezone.utc)
    if expires_dt is None:
        return False
    return expires_dt > now


def _compute_hand_counts(hand: Optional[dict]) -> dict[str, int]:
    counts = {dim: 0 for dim in VALID_DIMENSIONS}
    if not hand:
        return counts
    cards = hand.get("cards")
    if isinstance(cards, list):
        for card in cards:
            dim = str((card or {}).get("dimension") or "").strip().lower()
            if dim in counts:
                counts[dim] += 1
        return counts
    raw_counts = hand.get("counts")
    if isinstance(raw_counts, dict):
        for dim in VALID_DIMENSIONS:
            counts[dim] = max(0, int(raw_counts.get(dim) or 0))
    return counts


def sanitize_hand_for_frontend(
    active_hand: Any,
    current_round: int = 0,
    pool_counts: Optional[dict] = None,
    now_utc: Optional[datetime] = None,
) -> dict:
    """防剧透脱敏视图：严禁暴露 cards 列表或任何词条 content。"""
    hand = _coerce_hand(active_hand)
    active = is_hand_active(hand, now_utc=now_utc)
    if active and hand is not None:
        counts = _compute_hand_counts(hand)
        total_count = sum(counts.values())
        expires_dt = _parse_iso_dt(hand.get("expires_at"))
        expires_at_ms = hand.get("expires_at_ms")
        if expires_at_ms is None and expires_dt is not None:
            expires_at_ms = int(expires_dt.timestamp() * 1000)
        expires_at = expires_dt.isoformat() if expires_dt is not None else hand.get("expires_at")
        source = hand.get("source")
        drawn_at = hand.get("drawn_at")
        hand_round = int(hand.get("round") or current_round or 0)
    else:
        counts = {dim: 0 for dim in VALID_DIMENSIONS}
        total_count = 0
        expires_at = None
        expires_at_ms = None
        source = None
        drawn_at = None
        hand_round = int(current_round or 0)

    result = {
        "active": bool(active),
        "total_count": total_count,
        "counts": counts,
        "current_round": int(current_round or 0),
        "hand_round": hand_round,
        "drawn_at": drawn_at,
        "expires_at": expires_at,
        "expires_at_ms": expires_at_ms,
        "source": source,
    }
    if pool_counts is not None:
        result["pool_counts"] = pool_counts
    return result


def format_active_hand_hint(active_hand: Any, now_utc: Optional[datetime] = None) -> str:
    """将生效手牌拼接成极简自然的灵感提示；未生效或过期返回空串。"""
    hand = _coerce_hand(active_hand)
    if not is_hand_active(hand, now_utc=now_utc):
        return ""
    cards = hand.get("cards") or []
    parts: list[str] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        dim = str(card.get("dimension") or "").strip().lower()
        content = str(card.get("content") or "").strip()
        if not content:
            continue
        label = DIMENSION_LABELS.get(dim, dim or "灵感")
        parts.append(f"【{label}】{content}")
    if not parts:
        return ""
    joined = "、".join(parts)
    return (
        f"<secret_wish_hint>暗牌心愿灵感：{joined}。"
        "请以细腻的感官厮磨、呼吸留白与张弛有度的节奏变化为主线，"
        "待情动水到渠成时，将牌面化作心照不宣的动作细节自然融于兴致之中，"
        "亦可随当下心境自在取舍。</secret_wish_hint>"
    )


def _row_to_card(row) -> dict:
    created = row["created_at"]
    return {
        "id": int(row["id"]),
        "dimension": row["dimension"],
        "content": row["content"],
        "enabled": bool(row["enabled"]),
        "last_drawn_round": int(row["last_drawn_round"] or 0),
        "created_at": created.isoformat() if isinstance(created, datetime) else (str(created) if created else None),
    }


async def list_secret_cards(
    dimension: Optional[str] = None,
    enabled_only: bool = False,
) -> list[dict]:
    pool = await db_core.get_pool()
    clauses: list[str] = []
    params: list[Any] = []
    if dimension is not None and str(dimension).strip():
        dim = normalize_dimension(dimension)
        params.append(dim)
        clauses.append(f"dimension = ${len(params)}")
    if enabled_only:
        clauses.append("enabled = TRUE")
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT id, dimension, content, enabled, last_drawn_round, created_at "
        f"FROM secret_cards {where_sql} "
        "ORDER BY id DESC"
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [_row_to_card(r) for r in rows]


async def get_pool_counts() -> dict:
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT dimension,
                   COUNT(*)::INT AS total_count,
                   COUNT(*) FILTER (WHERE enabled = TRUE)::INT AS enabled_count
            FROM secret_cards
            GROUP BY dimension
            """
        )
    total = {dim: 0 for dim in VALID_DIMENSIONS}
    enabled = {dim: 0 for dim in VALID_DIMENSIONS}
    for row in rows:
        dim = row["dimension"]
        if dim in total:
            total[dim] = int(row["total_count"] or 0)
            enabled[dim] = int(row["enabled_count"] or 0)
    total["all"] = sum(total[d] for d in VALID_DIMENSIONS)
    enabled["all"] = sum(enabled[d] for d in VALID_DIMENSIONS)
    return {"total": total, "enabled": enabled}


async def create_secret_card(
    dimension: str,
    content: str,
    enabled: bool = True,
) -> dict:
    dim = normalize_dimension(dimension)
    clean_content = str(content or "").strip()
    if not clean_content:
        raise ValueError("词条文本 content 不能为空")
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            """
            SELECT id, dimension, content, enabled, last_drawn_round, created_at
            FROM secret_cards
            WHERE dimension = $1 AND content = $2
            ORDER BY id ASC
            LIMIT 1
            """,
            dim,
            clean_content,
        )
        if existing:
            return _row_to_card(existing)
        row = await conn.fetchrow(
            """
            INSERT INTO secret_cards (dimension, content, enabled, last_drawn_round)
            VALUES ($1, $2, $3, 0)
            RETURNING id, dimension, content, enabled, last_drawn_round, created_at
            """,
            dim,
            clean_content,
            bool(enabled),
        )
    return _row_to_card(row)


async def batch_import_secret_cards(
    dimension: str,
    raw_text: str | list[str],
    enabled: bool = True,
) -> dict:
    dim = normalize_dimension(dimension)
    lines = parse_batch_lines(raw_text)
    if not lines:
        counts = await get_pool_counts()
        return {"added_count": 0, "skipped_count": 0, "cards": [], "pool_counts": counts}

    pool = await db_core.get_pool()
    added_cards: list[dict] = []
    skipped_count = 0
    async with pool.acquire() as conn:
        existing_rows = await conn.fetch(
            "SELECT content FROM secret_cards WHERE dimension = $1",
            dim,
        )
        existing_set = {str(r["content"]).strip() for r in existing_rows}
        to_insert: list[str] = []
        for item in lines:
            if item in existing_set:
                skipped_count += 1
                continue
            existing_set.add(item)
            to_insert.append(item)

        if to_insert:
            async with conn.transaction():
                for item in to_insert:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO secret_cards (dimension, content, enabled, last_drawn_round)
                        VALUES ($1, $2, $3, 0)
                        RETURNING id, dimension, content, enabled, last_drawn_round, created_at
                        """,
                        dim,
                        item,
                        bool(enabled),
                    )
                    added_cards.append(_row_to_card(row))

    counts = await get_pool_counts()
    return {
        "added_count": len(added_cards),
        "skipped_count": skipped_count,
        "cards": added_cards,
        "pool_counts": counts,
    }


async def update_secret_card(
    card_id: int,
    *,
    content: Optional[str] = None,
    dimension: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> Optional[dict]:
    sets: list[str] = []
    params: list[Any] = []
    if content is not None:
        clean_content = str(content).strip()
        if not clean_content:
            raise ValueError("词条文本 content 不能为空")
        params.append(clean_content)
        sets.append(f"content = ${len(params)}")
    if dimension is not None:
        dim = normalize_dimension(dimension)
        params.append(dim)
        sets.append(f"dimension = ${len(params)}")
    if enabled is not None:
        params.append(bool(enabled))
        sets.append(f"enabled = ${len(params)}")
    if not sets:
        raise ValueError("至少提供一个需要修改的字段")

    params.append(int(card_id))
    sql = (
        f"UPDATE secret_cards SET {', '.join(sets)} "
        f"WHERE id = ${len(params)} "
        "RETURNING id, dimension, content, enabled, last_drawn_round, created_at"
    )
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
    return _row_to_card(row) if row else None


async def delete_secret_card(card_id: int) -> bool:
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        res = await conn.execute("DELETE FROM secret_cards WHERE id = $1", int(card_id))
    return str(res).endswith("1") or "DELETE 1" in str(res)


async def batch_delete_secret_cards(
    *,
    ids: Optional[list[int]] = None,
    dimension: Optional[str] = None,
    disabled_only: bool = False,
) -> int:
    pool = await db_core.get_pool()
    clauses: list[str] = []
    params: list[Any] = []
    if ids is not None:
        clean_ids = [int(i) for i in ids if not isinstance(i, bool)]
        if not clean_ids:
            return 0
        params.append(clean_ids)
        clauses.append(f"id = ANY(${len(params)}::int[])")
    if dimension is not None and str(dimension).strip():
        dim = normalize_dimension(dimension)
        params.append(dim)
        clauses.append(f"dimension = ${len(params)}")
    if disabled_only:
        clauses.append("enabled = FALSE")

    if not clauses:
        raise ValueError("批量删除至少需要指定 ids、dimension 或 disabled_only")

    sql = f"DELETE FROM secret_cards WHERE {' AND '.join(clauses)} RETURNING id"
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return len(rows)


async def _ensure_deck_state_row(conn) -> None:
    await conn.execute(
        """
        INSERT INTO secret_deck_state (id, current_round, active_hand)
        VALUES (1, 0, NULL)
        ON CONFLICT (id) DO NOTHING
        """
    )


async def get_deck_state(now_utc: Optional[datetime] = None) -> dict:
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await _ensure_deck_state_row(conn)
        row = await conn.fetchrow(
            "SELECT current_round, active_hand, updated_at FROM secret_deck_state WHERE id = 1"
        )
    current_round = int(row["current_round"] or 0) if row else 0
    hand = _coerce_hand(row["active_hand"] if row else None)
    active = is_hand_active(hand, now_utc=now_utc)
    return {
        "current_round": current_round,
        "active": active,
        "active_hand": hand if active else None,
    }


async def get_public_deck_status(now_utc: Optional[datetime] = None) -> dict:
    state = await get_deck_state(now_utc=now_utc)
    pool_counts = await get_pool_counts()
    return sanitize_hand_for_frontend(
        state["active_hand"],
        current_round=state["current_round"],
        pool_counts=pool_counts,
        now_utc=now_utc,
    )


async def get_active_hand_hint(now_utc: Optional[datetime] = None) -> str:
    state = await get_deck_state(now_utc=now_utc)
    if not state["active"]:
        return ""
    return format_active_hand_hint(state["active_hand"], now_utc=now_utc)


async def draw_secret_cards(
    *,
    ratios: Optional[dict] = None,
    count: int = 2,
    source: str = "user",
    force_redraw: bool = False,
    now_utc: Optional[datetime] = None,
    rng: Optional[random.Random] = None,
) -> dict:
    """抽牌核心入口。

    - force_redraw=False（AI 工具调用）：若 12 小时内已有生效手牌，直接原样返回，不重抽不递增局数。
    - force_redraw=True（前端主动触发）或当前无有效手牌：current_round + 1，按 8 局冷却与兜底算法发牌并锁定 12 小时。
    """
    now = _utc_now(now_utc)
    norm_ratios = normalize_ratios(ratios)
    pool = await db_core.get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            await _ensure_deck_state_row(conn)
            state_row = await conn.fetchrow(
                "SELECT current_round, active_hand FROM secret_deck_state WHERE id = 1 FOR UPDATE"
            )
            current_round = int(state_row["current_round"] or 0) if state_row else 0
            existing_hand = _coerce_hand(state_row["active_hand"] if state_row else None)

            if not force_redraw and is_hand_active(existing_hand, now_utc=now):
                return {
                    **existing_hand,
                    "active": True,
                    "reused": True,
                    "current_round": current_round,
                    "blind_box": True,
                }

            card_rows = await conn.fetch(
                """
                SELECT id, dimension, content, enabled, last_drawn_round, created_at
                FROM secret_cards
                WHERE enabled = TRUE
                ORDER BY id ASC
                """
            )
            enabled_cards = [_row_to_card(r) for r in card_rows]

            new_round = current_round + 1
            drawn: list[dict] = []
            if norm_ratios is not None:
                for dim in VALID_DIMENSIONS:
                    dim_count = norm_ratios.get(dim, 0)
                    if dim_count > 0:
                        dim_cards = [c for c in enabled_cards if c["dimension"] == dim]
                        drawn.extend(
                            select_cards_for_round(dim_cards, dim_count, new_round, rng=rng)
                        )
            else:
                target_count = max(1, min(20, int(count or 2)))
                drawn = select_cards_for_round(enabled_cards, target_count, new_round, rng=rng)

            if drawn:
                drawn_ids = [int(c["id"]) for c in drawn]
                await conn.execute(
                    "UPDATE secret_cards SET last_drawn_round = $1 WHERE id = ANY($2::int[])",
                    new_round,
                    drawn_ids,
                )
                counts = {dim: 0 for dim in VALID_DIMENSIONS}
                formatted_cards = []
                for card in drawn:
                    dim = card["dimension"]
                    counts[dim] = counts.get(dim, 0) + 1
                    formatted_cards.append({
                        "id": int(card["id"]),
                        "dimension": dim,
                        "dimension_label": DIMENSION_LABELS.get(dim, dim),
                        "content": card["content"],
                    })
                expires_dt = now + timedelta(seconds=ACTIVE_HAND_TTL_SECONDS)
                new_hand = {
                    "cards": formatted_cards,
                    "counts": counts,
                    "total_count": len(formatted_cards),
                    "round": new_round,
                    "source": source,
                    "drawn_at": now.isoformat(),
                    "expires_at": expires_dt.isoformat(),
                    "expires_at_ms": int(expires_dt.timestamp() * 1000),
                }
            else:
                new_hand = None

            await conn.execute(
                """
                UPDATE secret_deck_state
                SET current_round = $1,
                    active_hand = $2::jsonb,
                    updated_at = NOW()
                WHERE id = 1
                """,
                new_round,
                json.dumps(new_hand, ensure_ascii=False) if new_hand is not None else None,
            )

    if new_hand is None:
        return {
            "active": False,
            "reused": False,
            "current_round": new_round,
            "cards": [],
            "counts": {dim: 0 for dim in VALID_DIMENSIONS},
            "total_count": 0,
            "expires_at": None,
            "expires_at_ms": None,
            "source": source,
            "blind_box": True,
        }

    return {
        **new_hand,
        "active": True,
        "reused": False,
        "current_round": new_round,
        "blind_box": True,
    }


async def clear_active_hand() -> dict:
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await _ensure_deck_state_row(conn)
        row = await conn.fetchrow(
            """
            UPDATE secret_deck_state
            SET active_hand = NULL, updated_at = NOW()
            WHERE id = 1
            RETURNING current_round
            """
        )
    current_round = int(row["current_round"] or 0) if row else 0
    pool_counts = await get_pool_counts()
    return sanitize_hand_for_frontend(
        None,
        current_round=current_round,
        pool_counts=pool_counts,
    )


async def sync_deck_from_client(payload: dict) -> dict:
    """供 Cuplivo Worker 同步当前 active_hand 与 current_round 到 Pawwake。"""
    if not isinstance(payload, dict):
        return await get_public_deck_status()
    pool = await db_core.get_pool()
    async with pool.acquire() as conn:
        await _ensure_deck_state_row(conn)
        row = await conn.fetchrow(
            "SELECT current_round, active_hand FROM secret_deck_state WHERE id = 1"
        )
        cur_round = int(row["current_round"] or 0) if row else 0
        next_round = cur_round
        if payload.get("current_round") is not None:
            try:
                next_round = max(0, int(payload["current_round"]))
            except (TypeError, ValueError):
                next_round = cur_round

        has_hand_key = "active_hand" in payload
        hand_val = _coerce_hand(payload.get("active_hand")) if has_hand_key else _coerce_hand(row["active_hand"] if row else None)
        await conn.execute(
            """
            UPDATE secret_deck_state
            SET current_round = $1,
                active_hand = $2::jsonb,
                updated_at = NOW()
            WHERE id = 1
            """,
            next_round,
            json.dumps(hand_val, ensure_ascii=False) if hand_val is not None else None,
        )
    return await get_public_deck_status()
