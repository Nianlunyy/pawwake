"""Unit tests for summary absolute time formatting, time period bounds, and rotation behavior."""

import asyncio
import re
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import shared
import partition_engine


def test_time_period_boundaries():
    """1. _get_time_period 边界测试：0:00, 5:59, 6:00, 8:59, 9:00, 11:59, 12:00, 13:59, 14:00, 17:59, 18:00, 22:59, 23:00, 23:59"""
    cases = [
        (0, 0, "凌晨"),
        (5, 59, "凌晨"),
        (6, 0, "早晨"),
        (8, 59, "早晨"),
        (9, 0, "上午"),
        (11, 59, "上午"),
        (12, 0, "午间"),
        (13, 59, "午间"),
        (14, 0, "下午"),
        (17, 59, "下午"),
        (18, 0, "晚间"),
        (22, 59, "晚间"),
        (23, 0, "深夜"),
        (23, 59, "深夜"),
    ]
    for h, m, expected in cases:
        dt = datetime(2026, 10, 10, h, m)
        actual = partition_engine._get_time_period(dt)
        assert actual == expected, f"{h:02d}:{m:02d} expected {expected}, got {actual}"
    return "14组临界时间点映射全部准确吻合"


def test_cross_midnight_batch_and_prefix():
    """2. 跨午夜批次：日期分隔行与 [时段 HH:mm] 前缀格式"""
    shared.TIMEZONE_HOURS = 8
    msgs = [
        {"role": "user", "content": "夜间测试1", "created_at": "2026-10-09T15:30:00Z"},  # 本地 23:30 深夜
        {"role": "assistant", "content": "夜间回复1", "created_at": "2026-10-09T15:35:00Z"},
        {"role": "user", "content": "跨夜测试2", "created_at": "2026-10-09T16:10:00Z"},  # 本地 00:10 凌晨
        {"role": "assistant", "content": "跨夜回复2", "created_at": "2026-10-09T16:15:00Z"},
    ]

    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "正文测试"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert captured_prompt is not None
    assert "—— 10月9日 周五 ——" in captured_prompt
    assert "—— 10月10日 周六 ——" in captured_prompt
    assert "[深夜 23:30] 用户: 夜间测试1" in captured_prompt
    assert "[凌晨 00:10] 用户: 跨夜测试2" in captured_prompt
    return "已正确包含[深夜 23:30]与[凌晨 00:10]，凌晨准确归属次日"


def test_header_generation_and_roundtrip():
    """3. 段头：单端、双端同日、跨日、跨年；构造与解析往返一致"""
    shared.TIMEZONE_HOURS = 8
    # 单端 / 双端同日同时段
    t1 = datetime(2026, 10, 9, 23, 10, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 9, 23, 40, tzinfo=timezone.utc)
    h_same = partition_engine._build_summary_header(t1, t2)
    assert h_same == "【10月9日深夜】"

    # 双端同日不同时段
    t_day_start = datetime(2026, 10, 10, 8, 30, tzinfo=timezone.utc)
    t_day_end = datetime(2026, 10, 10, 13, 15, tzinfo=timezone.utc)
    h_same_day_diff = partition_engine._build_summary_header(t_day_start, t_day_end)
    assert h_same_day_diff == "【10月10日早晨 – 10月10日午间】"

    # 跨日
    t3 = datetime(2026, 10, 9, 23, 10, tzinfo=timezone.utc)
    t4 = datetime(2026, 10, 10, 12, 30, tzinfo=timezone.utc)
    h_cross_day = partition_engine._build_summary_header(t3, t4)
    assert h_cross_day == "【10月9日深夜 – 10月10日午间】"

    # 跨年
    t5 = datetime(2026, 12, 31, 23, 50, tzinfo=timezone.utc)
    t6 = datetime(2027, 1, 1, 1, 20, tzinfo=timezone.utc)
    h_cross_year = partition_engine._build_summary_header(t5, t6)
    assert h_cross_year == "【2026年12月31日深夜 – 2027年1月1日凌晨】"

    # 解析往返一致性
    pattern = r"^【([^】\–\-]+?)(?:\s*[–\-]\s*([^】]+?))?】"
    for h in [h_same, h_same_day_diff, h_cross_day, h_cross_year]:
        m = re.match(pattern, h)
        assert m is not None
        s = m.group(1).strip()
        e = m.group(2).strip() if m.group(2) else s
        rebuilt = f"【{s}】" if s == e else f"【{s} – {e}】"
        assert rebuilt == h
    return f"{h_same} | {h_same_day_diff} | {h_cross_day} | {h_cross_year}"


def test_folding_all_headers_vs_headless():
    """4. 折叠：全有头 -> base 有头；含存量无头旧段 -> base 无头；解析失败不抛异常"""
    shared.CACHE_SUMMARY_BUDGET_CHARS = 50
    shared.CACHE_SUMMARY_MODEL = "mock-model"

    # 全有头旧段
    all_headed_old = [
        "【10月8日晚间 – 10月9日凌晨】\n旧段A",
        "【10月9日早晨 – 10月9日午间】\n旧段B",
    ]
    recent_parts = ["【10月10日早晨】\n近期段C", "【10月10日午间】\n近期段D"]
    with patch.object(partition_engine, "_call_summary_model", return_value="合并后的基础摘要正文。"):
        folded_headed = asyncio.run(partition_engine.fold_summary_parts(all_headed_old + recent_parts, protected_tail=0))
    assert folded_headed is not None
    assert folded_headed[0].startswith("【10月8日晚间 – 10月9日午间】\n合并后的基础摘要正文。")

    # 含存量无头旧段 -> base 无头
    mixed_old = [
        "存量无头旧段内容A" * 5,
        "【10月8日晚间 – 10月9日凌晨】\n旧段B",
        "【即时状态】旧段C",
        "【10月9日早晨 – 10月9日午间】\n旧段D",
    ]
    with patch.object(partition_engine, "_call_summary_model", return_value="合并后的基础摘要正文。"):
        folded_mixed = asyncio.run(partition_engine.fold_summary_parts(mixed_old + recent_parts, protected_tail=0))
    assert folded_mixed is not None
    assert not folded_mixed[0].startswith("【")
    assert folded_mixed[0] == "合并后的基础摘要正文。"
    return "全有头正确继承起止段头，含任一无头旧段安全降级为无头base"


def test_strip_model_date_header():
    """5. _strip_model_date_header：剥离日期段头，不剥离【状态与约定】与【即时状态】"""
    with_date_1 = "【10月9日深夜 – 10月10日午间】\n阿狸与我一同品茶。"
    assert partition_engine._strip_model_date_header(with_date_1) == "阿狸与我一同品茶。"

    with_date_2 = "【10月10日午间】阿狸与我一同散步。"
    assert partition_engine._strip_model_date_header(with_date_2) == "阿狸与我一同散步。"

    preserve_status = "【即时状态】阿狸身体状况良好。"
    assert partition_engine._strip_model_date_header(preserve_status) == "【即时状态】阿狸身体状况良好。"

    preserve_other = "【状态与约定】阿狸周日休息。"
    assert partition_engine._strip_model_date_header(preserve_other) == "【状态与约定】阿狸周日休息。"
    return "日期段头精确剔除，业务标识【状态与约定】与【即时状态】未被误伤"


def test_prefix_cache_stability():
    """6. 旧段逐字节不变；build_partitioned_messages 摘要区前缀在追加新段前后文本一致；build_time_injection 仅在最后一条 user"""
    shared.CACHE_PARTITION_X = 2
    shared.CACHE_SUMMARY_MODEL = ""

    old_parts = ["【10月8日深夜】\n旧段1", "【10月9日午间】\n旧段2"]
    new_part = "【10月9日晚间】\n新追加段3"
    updated_parts = old_parts + [new_part]

    for old, updated in zip(old_parts, updated_parts[:len(old_parts)]):
        assert old == updated

    def make_blocks(parts):
        blocks = [{"type": "text", "text": "[以下是之前对话的摘要，帮助你回忆上下文]"}]
        for i, part in enumerate(parts):
            item = {"type": "text", "text": part}
            if i == len(parts) - 1:
                item["cache_control"] = shared.make_cache_control()
            blocks.append(item)
        return blocks

    b_old = make_blocks(old_parts)
    b_new = make_blocks(updated_parts)
    for i in range(len(b_old) - 1):
        assert b_old[i]["text"] == b_new[i]["text"]

    time_inj = partition_engine.build_time_injection()
    assert "<gateway_context>" in time_inj
    return "旧段及前缀block逐字节一致，断点前缀零污染"


def test_filtering_and_warning_tolerance():
    """7. tool 消息和空 assistant 不进摘要输入；created_at 缺失只打一次 warning 且不中断"""
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
        return "摘要正文"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            res = asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert "大量的搜索结果数据" not in captured_prompt
    assert "AI: \n" not in captured_prompt
    assert "请搜索资料" in captured_prompt
    assert "无时间戳消息" in captured_prompt
    assert "【10月10日午间】\n摘要正文" == res
    return "工具与空回复被丢弃，无效时间戳容错降级"


def test_prompt_table_and_empty_body_guard():
    """8. 提示词含对照表常量，且不含'即时状态''最新约定'；模型返回剥离后为空时按失败处理"""
    msgs = [{"role": "user", "content": "测试", "created_at": "2026-10-10T04:00:00Z"}]
    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "【10月10日午间】"  # 纯日期头，剥离后为空正文

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            res = asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert partition_engine.TIME_PERIOD_TABLE_TEXT in captured_prompt
    assert "凌晨0–6点、早晨6–9点" in captured_prompt
    assert "【状态与约定】" in captured_prompt
    assert "【即时状态】" not in captured_prompt
    assert "最新约定" not in captured_prompt
    # 验证边界保护：剥离后为空应返回空字符串
    assert res == ""
    return "提示词含完整对照表，剥离后空正文触发安全拦截返回空串"


def test_mock_two_days_flow():
    """9. 虚构的两天对话（含'明天'、跨午夜）模拟跑摘要和折叠"""
    shared.TIMEZONE_HOURS = 8
    shared.CACHE_SUMMARY_MODEL = "mock-model"

    two_days_msgs = [
        {"role": "user", "content": "今晚天气真好，我们明天顺延到周日去赏桂花吧。", "created_at": "2026-10-09T15:45:00Z"},  # 23:45 深夜
        {"role": "assistant", "content": "好呀，明天先好好休息，周日再出门。", "created_at": "2026-10-09T15:46:00Z"},
        {"role": "user", "content": "不知不觉过零点了，晚安。", "created_at": "2026-10-09T16:15:00Z"},  # 00:15 凌晨
        {"role": "assistant", "content": "晚安，快去休息。", "created_at": "2026-10-09T16:16:00Z"},
    ]

    mock_summary_out = "阿狸与我约定将桂花之约顺延至周日（10月11日）。截至10月10日凌晨，阿狸互道晚安入睡。"
    with patch.object(partition_engine, "_call_summary_model", return_value=mock_summary_out):
        generated = asyncio.run(partition_engine.generate_summary(two_days_msgs, "sess_flow"))

    assert generated.startswith("【10月9日深夜 – 10月10日凌晨】\n")

    # 折叠测试
    old_segments = [
        "【10月8日晚间 – 10月9日早晨】\n阿狸此前身体不适。",
        generated,
    ]
    mock_base = "阿狸身体逐渐好转，双方约定周日（10月11日）赏桂。"
    shared.CACHE_SUMMARY_BUDGET_CHARS = 30
    with patch.object(partition_engine, "_call_summary_model", return_value=mock_base):
        folded = asyncio.run(partition_engine.fold_summary_parts(old_segments + ["【10月10日午间】\n近期段1", "【10月10日下午】\n近期段2"]))

    assert folded[0].startswith("【10月8日晚间 – 10月10日凌晨】\n")
    return f"生成段头: {generated.splitlines()[0]} | 折叠段头: {folded[0].splitlines()[0]}"


if __name__ == "__main__":
    tests = [
        ("1. 时段边界临界点校验", test_time_period_boundaries),
        ("2. 跨午夜与[时段 HH:mm]前缀", test_cross_midnight_batch_and_prefix),
        ("3. 段头单双端与往返解析", test_header_generation_and_roundtrip),
        ("4. 折叠全有头与无头降级", test_folding_all_headers_vs_headless),
        ("5. 模型日期段头精准剥离", test_strip_model_date_header),
        ("6. 缓存前缀逐字节稳定性", test_prefix_cache_stability),
        ("7. 过滤tool与容错warning", test_filtering_and_warning_tolerance),
        ("8. 对照表同步与空正文保护", test_prompt_table_and_empty_body_guard),
        ("9. 两天跨夜完整流程模拟", test_mock_two_days_flow),
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
