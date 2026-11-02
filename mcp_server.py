"""Official MCP transport for Pawwake memory tools."""

from datetime import date, datetime

from mcp.server.mcpserver import MCPServer

import shared
from db import core as db_core
from db import conversations as db_conversations
from db import memories as db_memories
from db import search as db_search


server = MCPServer(
    "Pawwake",
    title="Pawwake Memory",
    description="Search and manage Pawwake memories.",
    version="4.2.0",
)


def _require_memory_enabled():
    if not shared.DATABASE_ENABLED:
        raise ValueError("数据库未启用")
    if not shared.MEMORY_ENABLED:
        raise ValueError("记忆系统未启用")


def _memory_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("memory_id 必须是正整数")
    return value


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("limit 必须是 1 到 100 的整数")
    return value


def _serialize_memory(memory) -> dict:
    result = dict(memory)
    for key, value in result.items():
        if isinstance(value, (date, datetime)):
            result[key] = value.isoformat()
    return result


@server.tool(
    description=(
        "添加一条 Pawwake 记忆。记忆默认进入碎片层且不会自动设为常驻；"
        "如需常驻，请再调用 pin_memory。"
    )
)
async def add_memory(
    content: str,
    importance: int = 5,
    title: str | None = None,
    layer: int = 1,
    event_date: str | None = None,
    external_id: str | None = None,
) -> dict:
    _require_memory_enabled()
    content = content.strip()
    if not content:
        raise ValueError("content 不能为空")
    if (
        isinstance(importance, bool)
        or not isinstance(importance, int)
        or not 1 <= importance <= 10
    ):
        raise ValueError("importance 必须是 1 到 10 的整数")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer not in (1, 2, 3):
        raise ValueError("layer 必须是 1、2 或 3")
    title = (title or "").strip() or None
    external_id = (external_id or "").strip() or None
    parsed_event_date = None
    if event_date:
        try:
            parsed_event_date = date.fromisoformat(event_date)
        except ValueError as exc:
            raise ValueError("event_date 必须是有效的 YYYY-MM-DD 日期") from exc

    # 双端防重保护：未传 external_id 时先查最近是否已有内容相同或关键词高度相似的激活态记忆
    if not external_id:
        dup = await db_memories.check_duplicate_memory(content)
        if dup.get("is_duplicate") and dup.get("matched_id"):
            matched_id = dup["matched_id"]
            existing = await db_memories.get_memory_by_id(matched_id)
            if existing:
                return {
                    "status": "exists",
                    "memory": _serialize_memory(existing),
                    "duplicate_reason": dup.get("reason"),
                }

    memory_id = await db_memories.save_memory(
        content=content,
        importance=importance,
        source_session="mcp",
        title=title,
        layer=layer,
        event_date=parsed_event_date,
        external_id=external_id,
    )
    if memory_id is None:
        existing = (
            await db_memories.get_memory_by_external_id(external_id)
            if external_id
            else None
        )
        if existing:
            return {"status": "exists", "memory": _serialize_memory(existing)}
        raise RuntimeError("添加记忆失败")

    return {
        "status": "created",
        "memory": {
            "id": memory_id,
            "content": content,
            "importance": importance,
            "title": title,
            "layer": layer,
            "event_date": parsed_event_date.isoformat() if parsed_event_date else None,
            "external_id": external_id,
            "is_active": True,
            "is_pinned": False,
        },
    }


@server.tool(description="搜索活跃 Pawwake 记忆，返回本次实际使用的搜索模式。")
async def search_memories(query: str, limit: int = 10) -> dict:
    _require_memory_enabled()
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")
    search_output = await db_memories.search_memories(
        query, _limit(limit), include_mode=True
    )
    results = search_output["results"]
    mode = search_output["mode"]
    return {
        "status": "ok",
        "mode": mode,
        "memories": [_serialize_memory(memory) for memory in results],
    }


@server.tool(
    description=(
        "列出最近记忆。默认只列活跃记忆；include_archived=true 时也列出已归档记忆，"
        "便于找到可恢复的 memory_id。"
    )
)
async def list_recent_memories(
    limit: int = 20,
    include_archived: bool = False,
) -> dict:
    _require_memory_enabled()
    memories = await db_memories.get_recent_memories(
        _limit(limit),
        include_archived=include_archived,
    )
    return {
        "status": "ok",
        "memories": [_serialize_memory(memory) for memory in memories],
    }


@server.tool(description="更新一条 Pawwake 记忆的正文、重要度、标题或层级。")
async def update_memory(
    memory_id: int,
    content: str | None = None,
    importance: int | None = None,
    title: str | None = None,
    layer: int | None = None,
) -> dict:
    _require_memory_enabled()
    memory_id = _memory_id(memory_id)
    if content is not None:
        content = content.strip()
        if not content:
            raise ValueError("content 不能为空")
    if importance is not None and (
        isinstance(importance, bool)
        or not isinstance(importance, int)
        or not 1 <= importance <= 10
    ):
        raise ValueError("importance 必须是 1 到 10 的整数")
    if layer is not None and (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or layer not in (1, 2, 3)
    ):
        raise ValueError("layer 必须是 1、2 或 3")
    if title is not None:
        title = title.strip()
    if all(value is None for value in (content, importance, title, layer)):
        raise ValueError("至少提供一个要更新的字段")

    updated = await db_memories.update_memory_with_layer(
        memory_id,
        content=content,
        importance=importance,
        title=title,
        layer=layer,
    )
    if not updated:
        raise ValueError("记忆不存在")
    return {"status": "updated", "id": memory_id}


@server.tool(
    description=(
        "删除一条 Pawwake 记忆。此工具只做可恢复的软删除：记忆会被移入归档并立即停止召回；"
        "不会永久删除，可用 restore_memory 恢复。"
    )
)
async def delete_memory(memory_id: int) -> dict:
    _require_memory_enabled()
    memory_id = _memory_id(memory_id)
    deleted = await db_memories.update_memory_with_layer(memory_id, is_active=False)
    if not deleted:
        raise ValueError("记忆不存在")
    return {
        "status": "archived",
        "id": memory_id,
        "message": "已删除并移入归档；不会再被召回，可用 restore_memory 恢复。",
    }


@server.tool(
    description=(
        "恢复一条已归档记忆。若它保留常驻标记，恢复时会重新检查 4000 字常驻预算。"
    )
)
async def restore_memory(memory_id: int) -> dict:
    _require_memory_enabled()
    memory_id = _memory_id(memory_id)
    result = await db_memories.restore_archived_memory(memory_id)
    if result["status"] == "not_found":
        raise ValueError("记忆不存在")
    if result["status"] == "superseded":
        raise ValueError("该记忆属于版本链，请在 Dashboard 撤销取代")
    return {"status": "restored", "id": memory_id}


@server.tool(
    description=(
        "将一条记忆设为常驻。活跃常驻正文总计最多 4000 字；常驻记忆会占用每轮记忆注入名额。"
    )
)
async def pin_memory(memory_id: int) -> dict:
    _require_memory_enabled()
    memory_id = _memory_id(memory_id)
    pinned = await db_memories.update_memory_with_layer(memory_id, is_pinned=True)
    if not pinned:
        raise ValueError("记忆不存在或已归档")
    return {"status": "pinned", "id": memory_id}


@server.tool(description="取消一条记忆的常驻标记。记忆本身仍保持活跃或归档状态。")
async def unpin_memory(memory_id: int) -> dict:
    _require_memory_enabled()
    memory_id = _memory_id(memory_id)
    unpinned = await db_memories.update_memory_with_layer(memory_id, is_pinned=False)
    if not unpinned:
        raise ValueError("记忆不存在")
    return {"status": "unpinned", "id": memory_id}


@server.tool(
    description=(
        "读取指定会话（或不传时默认读取最近活跃主会话）的当前 A/B 分区滚动记忆摘要（Partition Summary）及核心上下文状态。"
    )
)
async def get_memory_summary(session_id: str | None = None) -> dict:
    if not shared.DATABASE_ENABLED:
        raise ValueError("数据库未启用")
    session_id = (session_id or "").strip() or None
    if not session_id:
        session_id = shared.get_active_session_id()
    if not session_id:
        threads = await db_conversations.list_all_session_cache_states()
        if threads:
            session_id = threads[0]["session_id"]
        else:
            recent_convs, _ = await db_conversations.get_conversations_paginated(
                page=1, per_page=1
            )
            if recent_convs:
                session_id = recent_convs[0]["session_id"]
            else:
                session_id = "default"

    state = await db_conversations.get_session_cache_state(session_id)
    summary_parts = state.get("summary_parts", [])
    summary = "\n\n".join(summary_parts)
    updated_at = state.get("updated_at")
    updated_at_str = (
        updated_at.isoformat()
        if isinstance(updated_at, (date, datetime))
        else (str(updated_at) if updated_at else None)
    )

    return {
        "status": "ok",
        "session_id": session_id,
        "summary": summary,
        "summary_parts": summary_parts,
        "summary_count": len(summary_parts),
        "summary_length": len(summary),
        "a_start_round": state.get("a_start_round", 0),
        "updated_at": updated_at_str,
        "partition_enabled": shared.CACHE_PARTITION_ENABLED,
        "partition_x": shared.CACHE_PARTITION_X,
    }


@server.tool(
    description=(
        "根据关键词或语义在历史对话记录中检索相关的原始对话片段，或者列出最近对话切片。"
    )
)
async def recall_conversation_snippets(query: str, limit: int = 5) -> dict:
    if not shared.DATABASE_ENABLED:
        raise ValueError("数据库未启用")
    query = (query or "").strip()
    if not query:
        raise ValueError("query 不能为空")
    limit = _limit(limit) if 1 <= limit <= 100 else 5

    snippets = []
    if shared.CONVERSATION_RECALL_ENABLED:
        try:
            results, _ = await db_search.search_chat_fragments(
                query=query,
                max_sessions=limit,
                max_matches_per_session=1,
                context=1,
                mode="hybrid",
            )
            for item in results:
                sid = item.get("session_id", "")
                for frag in item.get("fragments", []):
                    snippets.append({
                        "session_id": sid,
                        "snippet": frag,
                    })
        except Exception:
            pass

    if not snippets:
        pool = await db_core.get_pool()
        keywords = db_search.extract_search_keywords(query)
        async with pool.acquire() as conn:
            if keywords:
                where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
                sql = f"""
                    SELECT id, session_id, role, content, created_at
                    FROM conversations
                    WHERE deleted_at IS NULL AND ({" OR ".join(where_parts)})
                    ORDER BY created_at DESC
                    LIMIT ${len(keywords) + 1}
                """
                rows = await conn.fetch(sql, *keywords, limit)
            else:
                sql = """
                    SELECT id, session_id, role, content, created_at
                    FROM conversations
                    WHERE deleted_at IS NULL AND content ILIKE '%' || $1 || '%'
                    ORDER BY created_at DESC
                    LIMIT $2
                """
                rows = await conn.fetch(sql, query, limit)

            for row in rows:
                t_str = row["created_at"].isoformat() if row.get("created_at") else ""
                role = row.get("role", "")
                content_snippet = (row.get("content") or "")[:500]
                snippets.append({
                    "session_id": row["session_id"],
                    "message_id": row["id"],
                    "role": role,
                    "created_at": t_str,
                    "content": content_snippet,
                    "snippet": f"[{t_str}] {role}: {content_snippet}",
                })

    return {
        "status": "ok",
        "query": query,
        "count": len(snippets),
        "snippets": snippets,
    }


http_app = server.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
    host="0.0.0.0",
)
