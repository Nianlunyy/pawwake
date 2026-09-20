"""Partition-cache message assembly and summary rotation."""

import json
import os
from datetime import datetime, timedelta, timezone

import httpx

import shared
from db import core as db_core
from db import conversations as db_conversations
from db import memories as db_memories

# ============================================================
# 分区缓存（Partition Cache）
# ============================================================

def _is_anthropic_model(model: str) -> bool:
    """判断是否为 Anthropic Claude 系列模型（只有 Claude 支持 cache_control）"""
    model_lower = model.lower()
    return "claude" in model_lower or "anthropic" in model_lower


def _strip_cache_control(messages: list):
    """
    剥掉消息中的 cache_control 字段，非 Claude 模型用不了。
    如果 content 数组只剩纯文本 block，降级回字符串格式。
    """
    stripped = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                del block["cache_control"]
                stripped += 1
        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
            msg["content"] = content[0]["text"]
    if stripped > 0:
        print(f"🔧 兼容性处理: 剥离了 {stripped} 个 cache_control 字段（非 Claude 模型）")


def _assemble_current_user_message(parts: list, raw_content) -> dict:
    """
    组装当前轮 user 消息：注入文本（时间/记忆，parts）+ 客户端原始 content。
    content 为多模态数组时保留图片等非文本块，只把文本块并进注入文本，
    否则 image_url 块会在拼接时被丢弃，模型永远看不到图。
    """
    if isinstance(raw_content, list):
        media_blocks = [
            b for b in raw_content
            if not (isinstance(b, dict) and b.get("type") == "text")
        ]
        text_joined = " ".join(
            b.get("text", "") for b in raw_content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if media_blocks:
            merged = "\n\n".join(parts + ([text_joined] if text_joined else []))
            return {"role": "user", "content": media_blocks + [{"type": "text", "text": merged}]}
        raw_content = text_joined
    parts.append(raw_content)
    return {"role": "user", "content": "\n\n".join(parts)}


def _message_text(message: dict) -> str:
    """Extract text from an OpenAI-compatible message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _extract_client_system_text(messages: list) -> str:
    """提取客户端自带的 system 消息文本（兼容字符串与多模态数组格式）。

    部分客户端（如 rikkahub 及其二改版）把工具列表和使用指引写在 system prompt 里，
    分区模式重建 messages 时若直接丢弃，会导致模型"不知道有什么工具"。
    """
    parts = []
    for message in messages:
        if message.get("role") != "system":
            continue
        text = _message_text(message).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _is_title_generation_request(messages: list) -> bool:
    """Detect client-side title generation prompts that must not enter chat history."""
    user_texts = [
        _message_text(message).strip()
        for message in messages
        if message.get("role") == "user"
    ]
    user_texts = [text for text in user_texts if text]
    if len(user_texts) != 1:
        return False

    text = user_texts[0].lower()
    strong_signatures = (
        "summarize the conversation between user and assistant into a short title",
        "summarize the conversation into a short title",
        "generate a concise title for the conversation",
        "generate a short title for the conversation",
    )
    if any(signature in text for signature in strong_signatures):
        return True

    # Some clients localize or slightly rewrite the boilerplate. Requiring three
    # independent markers avoids treating an ordinary title request as metadata.
    marker_groups = (
        ("<content>", "</content>"),
        ("reply directly with the title", "only output the title", "只输出标题", "直接输出标题"),
        ("title should not exceed", "title must not exceed", "标题不超过", "标题不得超过"),
        ("conversation between user and assistant", "dialogue between user and assistant", "用户和助手的对话", "用户与助手的对话"),
        ("short title", "concise title", "简短标题", "简洁标题"),
    )
    matched_groups = sum(
        1 for markers in marker_groups if any(marker in text for marker in markers)
    )
    return matched_groups >= 3


# 分区缓存模式下拼接到 system prompt 尾部的记忆使用说明。
# 非缓存模式的对应说明在 build_system_prompt_with_memories 里（记忆和说明都在 system）；
# 分区缓存模式记忆走 user 消息注入（<retrieved_memories> 块），这里只补静态说明，
# 内容固定所以不破坏 system 缓存。
MEMORY_USAGE_GUIDE = """

# 记忆应用
用户消息中的 <retrieved_memories> 块是网关自动检索的过往记忆，使用时：
- 像朋友般自然运用，不刻意展示；仅在相关话题出现时引用，避免主动提及
- 对重要信息（如健康、日期、约定）保持一致性
- 新信息与记忆冲突时，以新信息为准
- 模糊记忆可表达不确定性："记得你似乎说过..."
- 自然引用："记得你说过..."，避免机械式表达如"根据检索到的信息..."
"""


def build_time_injection() -> str:
    """构建时间注入文本（东八区）"""
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc + timedelta(hours=shared.TIMEZONE_HOURS)
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekday_names[now_local.weekday()]
    time_str = now_local.strftime("%Y年%m月%d日 %H:%M")
    return (
        f"<gateway_context>当前时间：{time_str} {weekday}。"
        f"此块由网关自动注入，不是用户发送的内容，无需回应或提及；"
        f"回答涉及日期、年份、时间时以此为准。</gateway_context>"
    )


async def generate_summary(messages: list, session_id: str = "") -> str:
    """调用轻量模型压缩A区消息为摘要"""
    if not messages:
        return ""
    if not shared.CACHE_SUMMARY_MODEL:
        print("📝 摘要模型未配置，跳过摘要生成（纯轮转模式：A区直接滑出上下文）")
        return ""

    conversation_text = ""
    for msg in messages:
        role_label = "用户" if msg['role'] == 'user' else "AI"
        content = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])
        conversation_text += f"{role_label}: {content}\n\n"

    prompt = f"""请将以下对话压缩成摘要。这份摘要会作为AI的记忆注入后续对话，请以AI的第一人称视角叙述（"我"指AI，用户用对话中的称呼）。
优先保留：情感节点、关系里程碑、双方的约定和决定、正在进行的话题。
保留双方的关键原话，用引号标注是谁说的。
去掉日常寒暄和重复内容。控制在300字以内。

---
{conversation_text}
---

摘要："""

    summary = await _call_summary_model(prompt, "摘要生成")
    if summary:
        print(f"📝 摘要生成完成: {len(summary)}字 (压缩{len(messages)}条消息)")
    return summary


async def _call_summary_model(prompt: str, label: str = "摘要生成") -> str:
    """用摘要模型跑一段 prompt，返回去空白后的正文；失败返回空串并自己打日志。"""
    try:
        # 摘要请求发往主API_BASE_URL，直接用主API_KEY（MEMORY_API_KEY可能是其他提供商的key）
        async with httpx.AsyncClient(timeout=60) as client:
            response = await shared.post_chat_completion(
                client,
                shared.API_BASE_URL,
                shared.API_KEY,
                {
                    "model": shared.CACHE_SUMMARY_MODEL,
                    # 推理模型的思考也消耗max_tokens，给足空间避免content为空
                    "max_tokens": shared.CACHE_SUMMARY_MAX_TOKENS,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    # 推理模型偶发返回content为None（思考吃光token或只返回reasoning_content）
                    # 空列表和缺字段都得接住：这条路走下去就是异常日志，它自己不能再抛
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    summary = content.strip()
                    if summary:
                        return summary

                    # 空content分不清是额度被思考吃光还是模型没给答案，把上游的判据一起打出来
                    finish_reason = choice.get("finish_reason")
                    usage = data.get("usage") or {}
                    completion_tokens = usage.get("completion_tokens")
                    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                    usage_part = (
                        f"，completion_tokens={completion_tokens}/{shared.CACHE_SUMMARY_MAX_TOKENS}"
                        if completion_tokens is not None else "，usage 未提供"
                    )
                    # 推理 0 是上游给的答案，不是没给。用 or 判断会把它退化成拿字符数瞎猜
                    if reasoning_tokens is not None:
                        usage_part += f"（其中推理 {reasoning_tokens}）"
                    else:
                        reasoning_text = message.get("reasoning_content") or message.get("reasoning") or ""
                        if reasoning_text:
                            usage_part += f"（上游未给推理token，reasoning正文 {len(reasoning_text)} 字符）"
                    print(
                        f"⚠️ {label}失败: 模型返回空content, model={shared.CACHE_SUMMARY_MODEL}, "
                        f"finish_reason={finish_reason}{usage_part}"
                    )
                    return ""

        print(f"⚠️ {label}失败: HTTP {response.status_code}, model={shared.CACHE_SUMMARY_MODEL}: {response.text[:500]}")
        return ""
    except Exception as e:
        print(f"⚠️ {label}异常: {e}")
        return ""


# 折叠时保留最近几段旧摘要原文（本轮新生成的段另外全保护，不占这个名额）
SUMMARY_KEEP_RECENT = 2


async def fold_summary_parts(summary_parts: list, protected_tail: int = 0):
    """
    摘要区有界折叠：总字符超过 CACHE_SUMMARY_BUDGET_CHARS 时，把最旧的几段合成一段基础摘要，
    保留最近 SUMMARY_KEEP_RECENT 段旧摘要原文，外加尾部 protected_tail 段本轮新生成的摘要。

    返回新的段列表；无需折叠时原样返回；折叠调用失败返回 None（调用方整份丢弃本轮候选状态）。
    """
    budget = shared.CACHE_SUMMARY_BUDGET_CHARS
    if budget <= 0 or not summary_parts:
        return summary_parts
    total = sum(len(p) for p in summary_parts)
    if total <= budget:
        return summary_parts
    if not shared.CACHE_SUMMARY_MODEL:
        print(f"📝 摘要区{total}字超预算{budget}，但摘要模型未配置，跳过折叠")
        return summary_parts

    keep = SUMMARY_KEEP_RECENT + max(0, protected_tail)
    if len(summary_parts) <= keep:
        print(f"📝 摘要区{total}字超预算{budget}，但只有{len(summary_parts)}段无可折叠（受保护{keep}段不丢）")
        return summary_parts

    old_parts = summary_parts[:-keep]
    recent_parts = summary_parts[-keep:]
    old_total = sum(len(p) for p in old_parts)
    target_chars = max(300, budget // 3)  # 基础摘要目标字数，模型返回后按它校验
    old_text = "\n\n".join(f"【第{i + 1}段】\n{p}" for i, p in enumerate(old_parts))
    prompt = f"""下面是同一段对话历史的多段摘要，按时间从早到晚排列。请把它们合并压缩成一段更精简的基础摘要。
仍以AI的第一人称视角叙述（"我"指AI，用户用摘要中的称呼）。
优先保留：情感节点、关系里程碑、双方的约定和决定、仍在进行的话题；重复出现的内容只留一次。
保留双方的关键原话，用引号标注是谁说的。控制在{target_chars}字以内。

---
{old_text}
---

合并后的摘要："""

    base = await _call_summary_model(prompt, "摘要折叠")
    if not base:
        return None
    # 不能全信模型守字数：基础摘要必须比被折叠的旧段短，且不超过目标；否则预算形同虚设，按失败走整轮回滚
    if len(base) > target_chars or len(base) >= old_total:
        print(
            f"⚠️ 摘要折叠失败: 模型返回{len(base)}字，超过目标{target_chars}字或不短于原{old_total}字，"
            f"本次折叠结果不保存"
        )
        return None
    folded = [base] + recent_parts
    print(
        f"📝 摘要折叠完成: {len(old_parts)}段/{old_total}字 → 1段/{len(base)}字，"
        f"摘要区 {len(summary_parts)}段/{total}字 → {len(folded)}段/{sum(len(p) for p in folded)}字"
    )
    return folded


def group_by_rounds(history: list) -> list:
    """
    按逻辑轮分组：每个user消息开始一轮，到下一个user前结束。
    一轮可能包含: [user, assistant] 或 [user, assistant(tool_calls), tool, assistant] 等。
    """
    rounds = []
    current_round = []
    for msg in history:
        if msg['role'] == 'user' and current_round:
            rounds.append(current_round)
            current_round = []
        current_round.append(msg)
    if current_round:
        rounds.append(current_round)
    return rounds


def _clean_orphan_tool_messages(history: list, *, log: bool = False) -> list:
    """移除前面没有 assistant(tool_calls) 或 tool 的孤立 tool 消息。"""
    cleaned = []
    orphan_count = 0
    for msg in history:
        if msg.get("role") == "tool":
            prev = cleaned[-1] if cleaned else None
            if prev and (
                prev.get("role") == "tool"
                or (prev.get("role") == "assistant" and prev.get("tool_calls"))
            ):
                cleaned.append(msg)
            else:
                orphan_count += 1
        else:
            cleaned.append(msg)
    if log and orphan_count:
        print(f"⚠️ 清理了 {orphan_count} 条孤立tool消息")
    return cleaned


def _build_memory_extraction_messages(
    context_messages: list,
    assistant_msg: str,
    interval: int,
) -> tuple[list, int]:
    """按逻辑轮截取近期上下文，并附上本轮最终 assistant 回复。"""
    non_system = [
        msg for msg in (context_messages or [])
        if msg.get("role") != "system"
    ]
    recent_rounds = group_by_rounds(non_system)[-max(1, interval):]
    messages = [
        {"role": msg.get("role"), "content": msg.get("content", "")}
        for round_messages in recent_rounds
        for msg in round_messages
        if msg.get("role") in {"user", "assistant"}
    ]
    messages.append({"role": "assistant", "content": assistant_msg})
    return messages, len(recent_rounds)


def _should_rotate(b_rounds_count: int, X: int, a_msgs: list) -> bool:
    """
    判断是否应该触发A区→摘要的轮转。

    rounds模式（默认）：B区轮数 >= X 时触发
    time模式：A区最晚一条消息距今 >= 时间窗口 时触发。
        看的是即将被压缩的整批：只要A区末条仍在窗口内，说明这批里还有新对话，不轮转；
        短时间内大量消息不会被频繁摘要，长时间沉默后恢复聊天时旧A区也能及时退出。
    """
    if b_rounds_count == 0:
        return False

    if shared.CACHE_PARTITION_TRIGGER == "time":
        a_last_time = None
        for msg in reversed(a_msgs):
            t = msg.get('created_at')
            if t:
                a_last_time = t
                break

        if a_last_time:
            now = datetime.now(timezone.utc)
            if a_last_time.tzinfo is None:
                a_last_time = a_last_time.replace(tzinfo=timezone.utc)
            age_minutes = (now - a_last_time).total_seconds() / 60
            return age_minutes >= shared.CACHE_PARTITION_WINDOW

        return b_rounds_count >= X

    return b_rounds_count >= X

# 时间窗口模式下单次请求最大轮转次数（防止一口气压完所有历史）
CACHE_MAX_ROTATIONS = int(os.getenv("CACHE_MAX_ROTATIONS", "2"))


def _apply_breakpoint(msg: dict) -> bool:
    """
    给消息打上 cache_control breakpoint。
    支持 content 为 str 或 list（多模态block数组）两种格式。
    返回 True 表示成功打上，False 表示无法打（比如content为空）。
    """
    content = msg.get('content')

    # content 是纯字符串
    if isinstance(content, str) and content.strip():
        msg['content'] = [{"type": "text", "text": content, "cache_control": shared.make_cache_control()}]
        return True

    # content 是 block 数组（多模态消息）
    if isinstance(content, list):
        # 从后往前找最后一个 text block
        for i in range(len(content) - 1, -1, -1):
            block = content[i]
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                block["cache_control"] = shared.make_cache_control()
                return True

    return False


async def build_partitioned_messages(
    session_id: str,
    all_messages: list,
    base_prompt: str,
    user_message: str,
    conversation_recall_text: str = "",
    memory_text_builder=None,
) -> list:
    """
    分区缓存模式：构建带breakpoint的messages数组。

    结构：
    system: [{人设, BP1}]                        ← 永远命中
    messages:
      [摘要blocks（每段一个block）, 最后BP]       ← 尾部追加，前面命中
      [摘要assistant]
      [A区消息... 最后一条BP2]                    ← 正常轮次不变
      [B区消息... 最后一条BP3]                    ← lookback命中
      [当前user: 时间+记忆+消息]                  ← 不缓存
    """
    X = shared.CACHE_PARTITION_X

    non_system = [m for m in all_messages if m.get('role') != 'system']

    current_user_msg = None
    history = non_system[:]
    if history and history[-1].get('role') == 'user':
        current_user_msg = history.pop()

    # 防止DB里的重复tool消息导致消息乱序
    history = _clean_orphan_tool_messages(history, log=True)

    # 按逻辑轮分组（解决tool消息导致的轮计数错乱）
    rounds = group_by_rounds(history)
    total_rounds = len(rounds)

    state = await db_conversations.get_session_cache_state(session_id)
    summary_parts = state['summary_parts']
    a_start_round = state['a_start_round']

    if total_rounds < X:
        return await _build_basic_cached(
            history,
            base_prompt,
            user_message,
            current_user_msg,
            summary_parts,
            conversation_recall_text,
            memory_text_builder,
        )

    # 计算A/B区（按逻辑轮切片）
    a_end_round = a_start_round + X
    a_round_groups = rounds[a_start_round : a_end_round]
    b_round_groups = rounds[a_end_round :]
    a_msgs = [msg for rnd in a_round_groups for msg in rnd]
    b_msgs = [msg for rnd in b_round_groups for msg in rnd]
    b_rounds_count = len(b_round_groups)

    # 折叠失败要能整份退回：进循环前把摘要和滑窗留底
    orig_summary_parts = list(summary_parts)
    orig_a_start_round = a_start_round

    rotation_count = 0
    max_rotations = CACHE_MAX_ROTATIONS if shared.CACHE_PARTITION_TRIGGER == "time" else 999
    while _should_rotate(b_rounds_count, X, a_msgs) and rotation_count < max_rotations:
        rotation_count += 1
        trigger_info = f"B区{b_rounds_count}轮 >= X={X}" if shared.CACHE_PARTITION_TRIGGER != "time" else f"A区末条消息超出{shared.CACHE_PARTITION_WINDOW}分钟窗口"
        print(f"🔄 轮转#{rotation_count}: session={session_id}, {trigger_info}")

        new_summary = await generate_summary(a_msgs, session_id)
        if new_summary:
            summary_parts.append(new_summary)
        elif shared.CACHE_SUMMARY_MODEL:
            # 配置了摘要模型但生成失败（网络/空content等）：中止本次轮转不推进滑窗，
            # A区消息保留在上下文里，下次请求重试。只有纯轮转模式（模型留空）才无摘要直接滑出。
            rotation_count -= 1
            print(f"⚠️ 摘要生成失败，本次轮转中止，下次请求重试（A区消息未丢失）")
            break

        a_start_round += X
        a_end_round = a_start_round + X
        a_round_groups = rounds[a_start_round : a_end_round]
        b_round_groups = rounds[a_end_round :]
        a_msgs = [msg for rnd in a_round_groups for msg in rnd]
        b_msgs = [msg for rnd in b_round_groups for msg in rnd]
        b_rounds_count = len(b_round_groups)

    if rotation_count > 0:
        # 同一次请求里先轮转再折叠，两次模型调用都在事务外；全部成功才一次落库
        folded = await fold_summary_parts(summary_parts, protected_tail=rotation_count)
        if folded is None:
            print(f"⚠️ 摘要折叠失败，本次{rotation_count}次轮转整份丢弃，状态不推进，下次请求重试（A区消息未丢失）")
            summary_parts = orig_summary_parts
            a_start_round = orig_a_start_round
            a_end_round = a_start_round + X
            a_round_groups = rounds[a_start_round : a_end_round]
            b_round_groups = rounds[a_end_round :]
            a_msgs = [msg for rnd in a_round_groups for msg in rnd]
            b_msgs = [msg for rnd in b_round_groups for msg in rnd]
            b_rounds_count = len(b_round_groups)
            rotation_count = 0
        else:
            summary_parts = folded
            await db_conversations.save_session_cache_state(session_id, summary_parts, a_start_round)
            summary_total = sum(len(p) for p in summary_parts)
            print(f"🔄 轮转完成(共{rotation_count}次): 摘要{len(summary_parts)}段/{summary_total}字, A区{len(a_msgs)}条, B区{len(b_msgs)}条")

    # 拼装messages
    result = []
    if base_prompt:
        result.append({
            "role": "system",
            "content": [{"type": "text", "text": base_prompt, "cache_control": shared.make_cache_control()}]
        })

    # 摘要区（多block，尾部追加模式）
    if summary_parts:
        blocks = [{"type": "text", "text": "[以下是之前对话的摘要，帮助你回忆上下文]"}]
        for i, part in enumerate(summary_parts):
            item = {"type": "text", "text": part}
            if i == len(summary_parts) - 1:
                item["cache_control"] = shared.make_cache_control()
            blocks.append(item)
        result.append({"role": "user", "content": blocks})
        result.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

    # A区：剥离tool消息和tool_calls，只保留有文本的user/assistant（节省上下文）
    cleaned_a = []
    for msg in a_msgs:
        if msg.get('role') == 'tool':
            continue
        m = {k: v for k, v in msg.items() if k not in ('created_at', 'tool_calls')}
        if m.get('role') == 'assistant' and not (m.get('content') or '').strip():
            continue
        cleaned_a.append(m)

    # A区：从末尾往前找第一条非tool消息打BP
    for j in range(len(cleaned_a) - 1, -1, -1):
        if cleaned_a[j].get('role') != 'tool' and _apply_breakpoint(cleaned_a[j]):
            break

    for m in cleaned_a:
        result.append(m)

    # B区：先构建去掉created_at的副本，再从末尾往前打BP
    b_cleaned = [{k: v for k, v in msg.items() if k not in ('created_at',)} for msg in b_msgs]

    for j in range(len(b_cleaned) - 1, -1, -1):
        if b_cleaned[j].get('role') != 'tool' and _apply_breakpoint(b_cleaned[j]):
            break

    for m in b_cleaned:
        result.append(m)

    if current_user_msg:
        parts = [build_time_injection()]

        if (
            shared.MEMORY_ENABLED
            and user_message
            and memory_text_builder
        ):
            mem_text = await memory_text_builder(user_message)
            if mem_text:
                parts.append(mem_text)

        if conversation_recall_text:
            parts.append(conversation_recall_text)

        result.append(_assemble_current_user_message(parts, current_user_msg['content']))

    bp_count = 1 + (1 if summary_parts else 0) + (1 if cleaned_a else 0) + (1 if b_msgs else 0)
    summary_total = sum(len(p) for p in summary_parts)
    tool_stripped = len(a_msgs) - len(cleaned_a)
    a_info = f"A区{len(cleaned_a)}条({len(a_round_groups)}轮)" + (f"[剥离{tool_stripped}条tool]" if tool_stripped else "")
    print(f"🔒 分区缓存: BP×{bp_count} | 摘要{'有' if summary_parts else '无'}({len(summary_parts)}段/{summary_total}字) | {a_info} | B区{len(b_msgs)}条({b_rounds_count}轮) | 总{len(result)}条messages")
    return result


async def _build_basic_cached(
    history: list,
    base_prompt: str,
    user_message: str,
    current_user_msg: dict,
    summary_parts: list = None,
    conversation_recall_text: str = "",
    memory_text_builder=None,
) -> list:
    """基础版prompt caching（历史不够分区时的降级模式）"""
    summary_parts = summary_parts or []
    result = []
    if base_prompt:
        result.append({
            "role": "system",
            "content": [{"type": "text", "text": base_prompt, "cache_control": shared.make_cache_control()}]
        })

    # 新建/继承的对话线在历史不足 X 轮时也必须读到继承摘要。
    # 否则 dashboard 和 DB 都显示摘要存在，但首轮请求不会注入。
    if summary_parts:
        blocks = [{"type": "text", "text": "[以下是之前对话的摘要，帮助你回忆上下文]"}]
        for i, part in enumerate(summary_parts):
            item = {"type": "text", "text": part}
            if i == len(summary_parts) - 1:
                item["cache_control"] = shared.make_cache_control()
            blocks.append(item)
        result.append({"role": "user", "content": blocks})
        result.append({"role": "assistant", "content": "好的，我已了解之前的对话内容。"})

    h_cleaned = [{k: v for k, v in msg.items() if k not in ('created_at',)} for msg in history]

    # 从末尾往前找第一条非tool消息打BP
    for j in range(len(h_cleaned) - 1, -1, -1):
        if h_cleaned[j].get('role') != 'tool' and _apply_breakpoint(h_cleaned[j]):
            break

    for m in h_cleaned:
        result.append(m)

    if current_user_msg:
        parts = [build_time_injection()]

        if (
            shared.MEMORY_ENABLED
            and user_message
            and memory_text_builder
        ):
            mem_text = await memory_text_builder(user_message)
            if mem_text:
                parts.append(mem_text)

        if conversation_recall_text:
            parts.append(conversation_recall_text)

        result.append(_assemble_current_user_message(parts, current_user_msg['content']))

    summary_total = sum(len(p) for p in summary_parts)
    bp_count = 1 + (1 if summary_parts else 0) + (1 if history else 0)
    print(f"🔒 基础缓存(降级): BP×{bp_count} | 摘要{'有' if summary_parts else '无'}({len(summary_parts)}段/{summary_total}字) | 历史{len(history)}条 | 总{len(result)}条messages")
    return result
