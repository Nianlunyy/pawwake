"""Gemini/Vertex multi-part thought parsing and leaked tool-reasoning draft guard.

Protects both normal chat (routes/chat.py) and autonomous background heartbeat
(heartbeat_service.py) against two related failure modes during multi-turn tool
calling:
1. Non-streaming / multi-part responses carrying explicit thought markers
   (`extra_content.google.thought`, `part.thought`, `type=thinking/thought`,
   or `<thought>`/`<think>` tags) that were previously only filtered in SSE.
2. Upstream Gemini format collapse where the model skips the hidden `thought`
   channel and drops structured `tool_calls`, returning a pure-English internal
   ReAct reasoning monologue (often declaring an unexecuted tool call such as
   "Let me call browse_xiaohongshu...") directly as ordinary text.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_THOUGHT_TAG_RE = re.compile(r"<(thought|think)>([\s\S]*?)</\1>", re.IGNORECASE)
_UNCLOSED_LEADING_THOUGHT_RE = re.compile(r"^\s*<(thought|think)>([\s\S]+)$", re.IGNORECASE)

# Fenced code blocks (including ```cuplivo-ui, ```html-preview, ```python, etc.)
_FENCED_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n[\s\S]*?(?:```|$)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EVENT_CAPSULE_BLOCK_RE = re.compile(r"<event_capsule\b[^>]*>[\s\S]*?</event_capsule>", re.IGNORECASE)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Signal A: Explicit unexecuted intent to invoke a tool (first-person / ReAct planning)
_GENERIC_TOOL_INTENT_RE = re.compile(
    r"\b(?:let\s+me|let's|i'll|i\s+will|i\s+should|i\s+need\s+to|i'm\s+going\s+to|i\s+am\s+going\s+to|going\s+to|now\s+let\s+me|next,?\s+let\s+me)\s+"
    r"(?:also\s+|first\s+|just\s+|try\s+to\s+|go\s+ahead\s+and\s+)?"
    r"(?:call|invoke|use|run|trigger|execute)\s+"
    r"[`'\"*]?([a-z][a-z0-9]*(?:_[a-z0-9]+)+)[`'\"*]?",
    re.IGNORECASE,
)

_ACTION_NEAR_TOOL_RE = re.compile(
    r"(?:"
    r"\b(?:let\s+me|let's|i'll|i\s+will|i\s+should|i\s+need\s+to|i\s+must|i'm\s+going\s+to|i\s+am\s+going\s+to|now\s+i\s+(?:can|should|will|'ll))\s+"
    r"(?:also\s+|first\s+|just\s+|try\s+(?:to\s+)?|go\s+ahead\s+and\s+)?"
    r"(?:call|invoke|use|run|trigger|execute|try)\b|"
    r"\bnow\s+(?:calling|invoking|using|running)\b|"
    r"(?:^|[.!?\n]\s*)(?:calling|invoking)\b"
    r")",
    re.IGNORECASE,
)

# Signal B: Internal chain-of-thought / ReAct self-talk markers
_OPENER_SELF_TALK_RE = re.compile(
    r"^\s*(?:"
    r"aha[!.,\s]|"
    r"wait[!.,\s]|"
    r"hmm+[.,!\s]|"
    r"oh[!.,]+\s+(?:so|wait|right|that's|that\s+is|i\s+see)|"
    r"okay,?\s+(?:so|let's|let\s+me|now|wait|great)|"
    r"alright,?\s+(?:so|let's|let\s+me|now)|"
    r"hold\s+on[!.,\s]|"
    r"let's\s+see[!.,\s]|"
    r"let\s+me\s+(?:think|check|see|look|call|search|verify|review)"
    r")",
    re.IGNORECASE,
)

_META_NOTE_RE = re.compile(
    r"(?:"
    r"\(\s*note\s*[:\-]|"
    r"\bnote\s*:\s*i\s+(?:just\s+)?(?:used|have|only|should|need|can|still)\b|"
    r"\bi\s+just\s+used\s+\d+\s+turn|"
    r"\b\d+\s+turn(?:s)?\s+(?:for|used|left|remaining)\b|"
    r"\b(?:tool\s+budget|turn\s+budget|search\s+budget|step\s+\d+\s+of\s+\d+)\b|"
    r"\b(?:tool_call|tool\s+call|function_call|function\s+call)\b"
    r")",
    re.IGNORECASE,
)

_META_PLANNING_RE = re.compile(
    r"(?:"
    r"\b(?:let\s+me|let's)\s+(?:call|use|check|search|look\s+up|see\s+what|see\s+if|fetch|query|verify|think)\b|"
    r"\bi\s+(?:should|need\s+to|will|'ll)\s+(?:call|use|invoke|search|check|also\s+check|stay\s+silent|output|respond\s+in|reply\s+in|write\s+an?)\b|"
    r"\b(?:the\s+user(?:'s)?|her\s+last\s+message|my\s+persona|system\s+prompt|next\s+step\s+is)\b|"
    r"\b(?:output|return|emit|send)\s+[`'\"]?(?:\[SILENT\]|<event_capsule|<next_wake|<topic)\b"
    r")",
    re.IGNORECASE,
)

_THIRD_PERSON_USER_RE = re.compile(
    r"\b(?:the\s+user\s+(?:is|was|has|wants|asked|mentioned|said)|"
    r"she\s+(?:mentioned|asked|said|was\s+talking|is\s+talking)|"
    r"he\s+(?:mentioned|asked|said|was\s+talking|is\s+talking)|"
    r"what\s+everyone\s+is\s+talking\s+about)\b",
    re.IGNORECASE,
)


def is_thinking_part(obj: Any) -> bool:
    """Return True if a delta / message / content-part dict is marked as thought."""
    if not isinstance(obj, dict):
        return False
    if obj.get("thought") is True:
        return True
    part_type = str(obj.get("type") or "").strip().lower()
    if part_type in {"thought", "thinking", "reasoning"}:
        return True
    if "extra_content" in obj:
        ec = obj.get("extra_content")
        if isinstance(ec, dict):
            google_ec = ec.get("google")
            if isinstance(google_ec, dict):
                if "thought" in google_ec:
                    return bool(google_ec["thought"])
                if "thought_signature" in google_ec:
                    return False
            if "thought" in ec:
                return bool(ec["thought"])
            if ec:
                return True
    return False


def extract_nonstream_message_content(msg_obj: dict | None) -> tuple[str, str]:
    """Extract (visible_text, reasoning_text) from an OpenAI-compatible message dict.

    Handles:
    - `reasoning_content` / `reasoning`
    - `extra_content.google.thought == True` on `message`
    - Multi-part `content` lists (filtering out parts where `is_thinking_part(part)`)
    - `<thought>...</thought>` / `<think>...</think>` blocks (and unclosed leading tag)
    """
    if not isinstance(msg_obj, dict):
        return "", ""

    reasoning_parts: list[str] = []
    visible_parts: list[str] = []

    for r_key in ("reasoning_content", "reasoning"):
        r_val = msg_obj.get(r_key)
        if isinstance(r_val, str) and r_val.strip():
            reasoning_parts.append(r_val.strip())

    raw_content = msg_obj.get("content")
    if isinstance(raw_content, list):
        for part in raw_content:
            if isinstance(part, str):
                visible_parts.append(part)
            elif isinstance(part, dict):
                text_val = part.get("text")
                if not isinstance(text_val, str):
                    text_val = part.get("thinking")
                if not isinstance(text_val, str):
                    text_val = part.get("content")
                if not isinstance(text_val, str) or not text_val:
                    continue
                if is_thinking_part(part) and not _THOUGHT_TAG_RE.search(text_val):
                    reasoning_parts.append(text_val.strip())
                else:
                    visible_parts.append(text_val)
    elif isinstance(raw_content, str) and raw_content:
        if is_thinking_part(msg_obj) and not _THOUGHT_TAG_RE.search(raw_content):
            reasoning_parts.append(raw_content.strip())
        else:
            visible_parts.append(raw_content)

    visible_text = "".join(visible_parts)

    # Extract and strip explicit <thought>/<think> XML blocks
    if visible_text and ("<" in visible_text):
        def _collect_thought(match: re.Match) -> str:
            inner = (match.group(2) or "").strip()
            if inner:
                reasoning_parts.append(inner)
            return ""

        visible_text = _THOUGHT_TAG_RE.sub(_collect_thought, visible_text)
        unclosed = _UNCLOSED_LEADING_THOUGHT_RE.match(visible_text)
        if unclosed:
            inner = (unclosed.group(2) or "").strip()
            if inner:
                reasoning_parts.append(inner)
            visible_text = ""

    visible_text = visible_text.strip()
    combined_reasoning = "\n\n".join(p for p in reasoning_parts if p).strip()
    return visible_text, combined_reasoning


def extract_tool_names(tools: Iterable[dict] | None) -> set[str]:
    """Extract declared function names from OpenAI-format `tools` list."""
    names: set[str] = set()
    if not tools:
        return names
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name") or "").strip()
            if name:
                names.add(name)
        else:
            name = str(item.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def strip_code_and_structured_blocks(text: str) -> str:
    """Remove fenced code blocks, inline code, and URLs to isolate natural prose."""
    cleaned = _FENCED_CODE_BLOCK_RE.sub(" ", text or "")
    cleaned = _INLINE_CODE_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    return cleaned.strip()


def _prose_language_stats(text: str) -> tuple[int, int, float]:
    """Return (cjk_count, latin_word_count, cjk_char_ratio) on prose outside code blocks."""
    prose = strip_code_and_structured_blocks(text)
    if not prose:
        return 0, 0, 0.0
    cjk_count = len(_CJK_RE.findall(prose))
    latin_chars = len(_LATIN_CHAR_RE.findall(prose))
    latin_words = len(_LATIN_WORD_RE.findall(prose))
    total_chars = cjk_count + latin_chars
    cjk_ratio = (cjk_count / total_chars) if total_chars > 0 else 0.0
    return cjk_count, latin_words, cjk_ratio


def text_is_primarily_english(text: str) -> bool:
    """Return True if the prose is primarily English (used to detect English user prompts)."""
    cjk_count, latin_words, cjk_ratio = _prose_language_stats(text)
    return latin_words >= 6 and cjk_ratio <= 0.15


def has_suspicious_stream_prefix(text: str) -> bool:
    """Fast check on a streaming prefix to decide whether to keep holding chunks."""
    if not text or not text.strip():
        return True
    # Any fenced code block (including cuplivo-ui / html-preview) -> flush immediately
    if "```" in text:
        return False
    # Any closed event capsule -> flush immediately
    if "</event_capsule>" in text.lower():
        return False
    cjk_count, latin_words, cjk_ratio = _prose_language_stats(text)
    # Normal Chinese or Chinese-English mixed reply -> flush immediately
    if cjk_count >= 2 and cjk_ratio > 0.20:
        return False
    return True


def has_reasoning_signal_in_prefix(text: str, declared_tools: Iterable[dict] | None = None) -> bool:
    """Return True if the prefix already shows ReAct self-talk or tool-call intent."""
    prose = strip_code_and_structured_blocks(text)
    if not prose:
        return False
    if _OPENER_SELF_TALK_RE.search(prose):
        return True
    if _META_NOTE_RE.search(prose):
        return True
    if _GENERIC_TOOL_INTENT_RE.search(prose):
        return True
    tool_names = extract_tool_names(declared_tools)
    if tool_names:
        lower_text = text.lower()
        for tname in tool_names:
            if tname.lower() in lower_text:
                return True
    return False


def is_leaked_tool_reasoning_draft(
    text: str,
    *,
    declared_tools: Iterable[dict] | None = None,
    after_tool_result: bool = False,
    user_message: str = "",
    is_heartbeat: bool = False,
) -> bool:
    """Detect pure-English internal ReAct reasoning drafts leaked as normal content.

    Designed for near-zero false positives on:
    - Chinese or Chinese-English mixed replies
    - Code blocks and `cuplivo-ui` / `html-preview` interactive cards
    - Legitimate English replies when the user wrote in English
    """
    if not text or not text.strip():
        return False

    # Never intercept deliberate structured UI cards or completed event capsules
    # when accompanied by non-empty structured output (unless pure English ReAct outside)
    if _EVENT_CAPSULE_BLOCK_RE.search(text):
        # Check if the capsule body itself contains Chinese (legitimate heartbeat capsule)
        cjk_all = len(_CJK_RE.findall(text))
        if cjk_all >= 4:
            return False

    cjk_count, latin_words, cjk_ratio = _prose_language_stats(text)

    # Must have substantial English prose and very low CJK char ratio (eliminates all
    # normal Chinese, Chinese-English mixed replies, and code-only / cuplivo-ui replies)
    if latin_words < 12 or cjk_ratio > 0.12:
        return False

    prose = strip_code_and_structured_blocks(text)
    tool_names = extract_tool_names(declared_tools)
    user_is_english = (not is_heartbeat) and text_is_primarily_english(user_message)

    # --- Signal A: Explicit unexecuted intent to call a tool ---
    has_tool_intent = False
    if _GENERIC_TOOL_INTENT_RE.search(text):
        has_tool_intent = True
    elif tool_names:
        for tname in tool_names:
            pattern = re.compile(rf"\b{re.escape(tname)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                window_start = max(0, match.start() - 80)
                window_end = min(len(text), match.end() + 40)
                window = text[window_start:window_end]
                if _ACTION_NEAR_TOOL_RE.search(window):
                    has_tool_intent = True
                    break
            if has_tool_intent:
                break

    if has_tool_intent:
        return True

    # If the user explicitly wrote in English, only Signal A (explicit unexecuted
    # tool-call intent) may trigger interception to avoid blocking valid English chat.
    if user_is_english:
        return False

    # --- Signal B: Internal ReAct / chain-of-thought monologue markers ---
    signal_score = 0
    if _OPENER_SELF_TALK_RE.search(prose):
        signal_score += 1
    if _META_NOTE_RE.search(prose):
        signal_score += 2  # "(Note: I just used 1 turn for...)" is unmistakable
    if _META_PLANNING_RE.search(prose):
        signal_score += 1
    if _THIRD_PERSON_USER_RE.search(prose):
        signal_score += 1

    # Mentioning a declared tool name inside an English post-tool monologue adds strong weight
    if tool_names and any(re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE) for t in tool_names):
        signal_score += 1

    if after_tool_result or is_heartbeat:
        return signal_score >= 2
    return signal_score >= 3
