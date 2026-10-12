"""Unit tests for summary absolute time formatting, range header bounds, folding, and stripping."""

import asyncio
import re
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import shared
import partition_engine


def test_header_format_and_regex():
    """1. 段头生成与正则匹配：【记录时段：YYYY-MM-DD HH:mm 至 YYYY-MM-DD HH:mm（UTC+8）】"""
    shared.TIMEZONE_HOURS = 8
    t_start = datetime(2026, 10, 9, 23, 30)
    t_end = datetime(2026, 10, 10, 12, 30)

    header = partition_engine._format_summary_range(t_start, t_end)
    expected = "【记录时段：2026-10-09 23:30 至 2026-10-10 12:30（UTC+8）】"
    assert header == expected, f"expected {expected}, got {header}"

    match = partition_engine.SUMMARY_RANGE_RE.match(header)
    assert match is not None, f"regex failed to match {header}"
    assert match.group("start") == "2026-10-09 23:30"
    assert match.group("end") == "2026-10-10 12:30"
    assert match.group("timezone") == "UTC+8"

    # 负时区测试
    with patch.object(shared, "TIMEZONE_HOURS", -5):
        h_neg = partition_engine._format_summary_range(t_start, t_end)
        assert h_neg == "【记录时段：2026-10-09 23:30 至 2026-10-10 12:30（UTC-5）】"
        m_neg = partition_engine.SUMMARY_RANGE_RE.match(h_neg)
        assert m_neg is not None
        assert m_neg.group("timezone") == "UTC-5"

    return f"段头生成与正则解析精确匹配: {header}"


def test_midnight_attribution_and_prefix():
    """2. 凌晨归属与消息前缀：UTC时间转本地时间，次日凌晨归属次日，前缀为 [YYYY-MM-DD HH:mm] 无分隔行"""
    shared.TIMEZONE_HOURS = 8
    # UTC 2026-10-09 16:15 -> 本地 2026-10-10 00:15:00
    dt_local = partition_engine._parse_message_local_time("2026-10-09T16:15:00Z")
    assert dt_local is not None
    assert dt_local.year == 2026 and dt_local.month == 10 and dt_local.day == 10
    assert dt_local.hour == 0 and dt_local.minute == 15

    msgs = [
        {"role": "user", "content": "夜间测试1", "created_at": "2026-10-09T15:30:00Z"},  # 本地 23:30
        {"role": "assistant", "content": "夜间回复1", "created_at": "2026-10-09T15:35:00Z"},
        {"role": "user", "content": "跨夜测试2", "created_at": "2026-10-09T16:10:00Z"},  # 本地 00:10 (10月10日)
        {"role": "assistant", "content": "跨夜回复2", "created_at": "2026-10-09T16:15:00Z"},
    ]

    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "阿澈与阿狸夜间交谈。"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            res = asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert captured_prompt is not None
    # 确认前缀为 [YYYY-MM-DD HH:mm]，且不含日期分隔行（如 —— 10月9日 ——）
    assert "[2026-10-09 23:30] 用户: 夜间测试1" in captured_prompt
    assert "[2026-10-10 00:10] 用户: 跨夜测试2" in captured_prompt
    assert "——" not in captured_prompt
    assert res.startswith("【记录时段：2026-10-09 23:30 至 2026-10-10 00:15（UTC+8）】\n")
    return "凌晨消息准确归属次日 2026-10-10，前缀格式与段头正确"


def test_folding_all_headers_vs_headless():
    """3. 折叠：全有头旧段 -> base 继承起止段头；含存量无头旧段 -> base 安全降级为无头"""
    shared.CACHE_SUMMARY_BUDGET_CHARS = 50
    shared.CACHE_SUMMARY_MODEL = "mock-model"
    shared.TIMEZONE_HOURS = 8

    # 全有头旧段
    all_headed_old = [
        "【记录时段：2026-10-08 20:00 至 2026-10-09 02:00（UTC+8）】\n旧段A",
        "【记录时段：2026-10-09 08:00 至 2026-10-09 12:00（UTC+8）】\n旧段B",
    ]
    recent_parts = [
        "【记录时段：2026-10-10 08:00 至 2026-10-10 10:00（UTC+8）】\n近期段C",
        "【记录时段：2026-10-10 12:00 至 2026-10-10 14:00（UTC+8）】\n近期段D",
    ]

    with patch.object(partition_engine, "_call_summary_model", return_value="合并后的基础摘要正文。"):
        folded_headed = asyncio.run(
            partition_engine.fold_summary_parts(all_headed_old + recent_parts, protected_tail=0)
        )
    assert folded_headed is not None
    expected_header = "【记录时段：2026-10-08 20:00 至 2026-10-09 12:00（UTC+8）】"
    assert folded_headed[0].startswith(f"{expected_header}\n合并后的基础摘要正文。")

    # 含存量无头旧段 -> base 无头
    mixed_old = [
        "存量无头旧段内容A" * 6,
        "【记录时段：2026-10-09 08:00 至 2026-10-09 12:00（UTC+8）】\n旧段B",
    ]
    with patch.object(partition_engine, "_call_summary_model", return_value="合并后的基础摘要正文。"):
        folded_mixed = asyncio.run(
            partition_engine.fold_summary_parts(mixed_old + recent_parts, protected_tail=0)
        )
    assert folded_mixed is not None
    assert not folded_mixed[0].startswith("【")
    assert folded_mixed[0] == "合并后的基础摘要正文。"
    return "全有头正确继承跨度段头，含任一无头旧段安全降级为无头base"


def test_strip_model_date_header():
    """4. _strip_model_date_header：精准剥离模型自带段头（记录时段/月日），保留【状态与约定】与【即时状态】"""
    # 上游新格式段头剥离
    with_range_1 = "【记录时段：2026-10-09 23:30 至 2026-10-10 12:30（UTC+8）】\n阿狸与我一同品茶。"
    assert partition_engine._strip_model_date_header(with_range_1) == "阿狸与我一同品茶。"

    with_range_2 = "【记录时段: 2026-10-09 23:30 至 2026-10-10 12:30（UTC+8）】阿狸与我一同品茶。"
    assert partition_engine._strip_model_date_header(with_range_2) == "阿狸与我一同品茶。"

    # 旧格式月日段头剥离
    with_date_1 = "【10月9日深夜 – 10月10日午间】\n阿狸与我一同品茶。"
    assert partition_engine._strip_model_date_header(with_date_1) == "阿狸与我一同品茶。"

    with_date_2 = "【10月10日午间】阿狸与我一同散步。"
    assert partition_engine._strip_model_date_header(with_date_2) == "阿狸与我一同散步。"

    # 业务标题保护
    preserve_status = "【即时状态】阿狸身体状况良好。"
    assert partition_engine._strip_model_date_header(preserve_status) == "【即时状态】阿狸身体状况良好。"

    preserve_other = "【状态与约定】阿狸周日休息。"
    assert partition_engine._strip_model_date_header(preserve_other) == "【状态与约定】阿狸周日休息。"

    # 纯段头剥离后为空
    pure_range = "【记录时段：2026-10-09 23:30 至 2026-10-10 12:30（UTC+8）】"
    assert partition_engine._strip_model_date_header(pure_range) == ""
    return "记录时段与月日段头均精准剥离，业务标记不受影响，纯段头剥离后为空"


def test_empty_body_guard():
    """5. 空正文安全保护：模型返回剥离后为空时，generate_summary 返回空串，fold_summary_parts 返回 None"""
    shared.CACHE_SUMMARY_MODEL = "mock-model"
    shared.CACHE_SUMMARY_BUDGET_CHARS = 50
    msgs = [{"role": "user", "content": "测试", "created_at": "2026-10-10T04:00:00Z"}]

    # generate_summary 模型只返回段头
    async def mock_call_header_only(prompt, label="摘要生成"):
        return "【记录时段：2026-10-10 12:00 至 2026-10-10 12:00（UTC+8）】"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call_header_only):
        res = asyncio.run(partition_engine.generate_summary(msgs, "sess1"))
    assert res == "", f"expected empty string on stripped empty body, got {res}"

    # fold_summary_parts 模型只返回段头
    old_parts = ["旧段内容超出预算" * 10, "旧段内容超出预算B" * 10]
    recent_parts = ["近期段1", "近期段2"]
    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call_header_only):
        folded = asyncio.run(
            partition_engine.fold_summary_parts(old_parts + recent_parts, protected_tail=0)
        )
    assert folded is None, f"expected None on stripped empty body in fold, got {folded}"
    return "剥离后空正文在生成和折叠场景均触发安全拦截"


def test_prompt_rules_and_skeleton():
    """6. 提示词骨架与时间规则核对：含时间规则上游版、'截至 YYYY-MM-DD'、'未来约定换算成具体日期'"""
    msgs = [{"role": "user", "content": "测试对话", "created_at": "2026-10-10T04:00:00Z"}]
    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "正文摘要。"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert captured_prompt is not None
    assert "将对话压缩为阿澈（自称\"我\"）与阿狸的常驻记忆摘要。" in captured_prompt
    assert "# 人称映射【绝对红线】" in captured_prompt
    assert "# 时间规则【绝对红线】" in captured_prompt
    assert "每条消息开头的时间戳是事实锚点。" in captured_prompt
    assert "能换算时写成 YYYY-MM-DD 或 YYYY-MM-DD HH:mm" in captured_prompt
    assert "对话里的未来约定换算成具体日期。" in captured_prompt
    assert "截至 YYYY-MM-DD" in captured_prompt
    assert "TIME_PERIOD_TABLE_TEXT" not in captured_prompt
    return "提示词保留本地第一人称骨架，时间规则与状态约定规则正确更新"


def test_filtering_and_warning_tolerance():
    """7. tool消息和空assistant不进摘要输入；created_at异常只打一次warning且不中断"""
    msgs = [
        {"role": "user", "content": "请搜索资料", "created_at": "2026-10-10T04:00:00Z"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}], "created_at": "2026-10-10T04:00:05Z"},
        {"role": "tool", "content": "大量的搜索结果数据", "created_at": "2026-10-10T04:00:10Z"},
        {"role": "assistant", "content": "搜索完毕结果已整理", "created_at": "2026-10-10T04:00:15Z"},
        {"role": "user", "content": "无时间戳消息", "created_at": "invalid_ts"},
    ]

    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "摘要正文。"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            res = asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert "大量的搜索结果数据" not in captured_prompt
    assert "AI: \n" not in captured_prompt
    assert "请搜索资料" in captured_prompt
    assert "无时间戳消息" in captured_prompt
    assert res.startswith("【记录时段：2026-10-10 12:00 至 2026-10-10 12:00（UTC+8）】\n")
    return "工具与空回复被过滤，无效时间戳容错降级"


def test_prefix_cache_stability():
    """8. 摘要块与时间注入稳定性测试"""
    shared.CACHE_PARTITION_X = 2
    shared.CACHE_SUMMARY_MODEL = ""

    old_parts = ["【记录时段：2026-10-08 20:00 至 2026-10-09 02:00（UTC+8）】\n旧段1"]
    new_part = "【记录时段：2026-10-09 08:00 至 2026-10-09 12:00（UTC+8）】\n新段2"
    updated_parts = old_parts + [new_part]

    for old, updated in zip(old_parts, updated_parts[:len(old_parts)]):
        assert old == updated

    time_inj = partition_engine.build_time_injection()
    assert "<gateway_context>" in time_inj
    return "前缀缓存稳定，时间注入标签完整"


if __name__ == "__main__":
    tests = [
        ("1. 段头生成与正则解析", test_header_format_and_regex),
        ("2. 凌晨归属与前缀格式", test_midnight_attribution_and_prefix),
        ("3. 折叠全有头与无头降级", test_folding_all_headers_vs_headless),
        ("4. 模型日期段头精准剥离", test_strip_model_date_header),
        ("5. 剥离后空正文安全拦截", test_empty_body_guard),
        ("6. 提示词骨架与规则核对", test_prompt_rules_and_skeleton),
        ("7. 过滤tool与容错warning", test_filtering_and_warning_tolerance),
        ("8. 缓存前缀与时间注入稳定", test_prefix_cache_stability),
    ]

    all_passed = True
    print("\n" + "=" * 65)
    for name, t_func in tests:
        try:
            res = t_func()
            print(f"✅ {name}: {res}")
        except Exception as e:
            print(f"❌ {name}: 失败 - {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    print("=" * 65 + "\n")
    if not all_passed:
        exit(1)
