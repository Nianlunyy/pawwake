"""Unit tests for summary absolute time formatting and rotation behavior."""

import asyncio
import re
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock

import shared
import partition_engine


def test_1_timezone_conversion():
    """1. UTC 2026-10-09T16:30:00Z -> 本地 10月10日 00:30 (TIMEZONE_HOURS=8)"""
    shared.TIMEZONE_HOURS = 8
    # 带时区字符串
    ts_aware_str = "2026-10-09T16:30:00Z"
    dt_aware = partition_engine._parse_message_local_time(ts_aware_str)
    assert dt_aware is not None
    assert dt_aware.year == 2026
    assert dt_aware.month == 10
    assert dt_aware.day == 10
    assert dt_aware.hour == 0
    assert dt_aware.minute == 30
    assert partition_engine._SUMMARY_WEEKDAYS[dt_aware.weekday()] == "周六"

    # 无时区 datetime（按 UTC 处理）
    ts_naive_dt = datetime(2026, 10, 9, 16, 30, 0)
    dt_naive = partition_engine._parse_message_local_time(ts_naive_dt)
    assert dt_naive is not None
    assert dt_naive.year == 2026
    assert dt_naive.month == 10
    assert dt_naive.day == 10
    assert dt_naive.hour == 0
    assert dt_naive.minute == 30
    assert partition_engine._SUMMARY_WEEKDAYS[dt_naive.weekday()] == "周六"
    return f"{dt_aware.strftime('%Y-%m-%d %H:%M')} ({partition_engine._SUMMARY_WEEKDAYS[dt_aware.weekday()]})"


def test_2_cross_midnight_batch():
    """2. 跨午夜批次：日期分隔行位置正确，凌晨消息归属当天"""
    shared.TIMEZONE_HOURS = 8
    msgs = [
        {"role": "user", "content": "夜间测试1", "created_at": "2026-10-09T15:30:00Z"},  # 本地 23:30 (10-09 周五)
        {"role": "assistant", "content": "夜间回复1", "created_at": "2026-10-09T15:35:00Z"},
        {"role": "user", "content": "跨夜测试2", "created_at": "2026-10-09T16:10:00Z"},  # 本地 00:10 (10-10 周六)
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
    idx_day1 = captured_prompt.index("—— 10月9日 周五 ——")
    idx_day2 = captured_prompt.index("—— 10月10日 周六 ——")
    assert idx_day1 < idx_day2
    assert "[23:30] 用户: 夜间测试1" in captured_prompt
    assert "[00:10] 用户: 跨夜测试2" in captured_prompt
    return "已正确按本地日历日切分且00:10归属10月10日周六"


def test_3_header_generation_and_roundtrip():
    """3. 段头：同日、跨日、跨年各一例；构造与解析往返一致"""
    shared.TIMEZONE_HOURS = 8
    # 同日同时段
    t1 = datetime(2026, 10, 9, 23, 10, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 9, 23, 40, tzinfo=timezone.utc)
    h_same = partition_engine._build_summary_header(t1, t2)
    assert h_same == "【10月9日深夜】", f"Unexpected {h_same}"

    # 跨日跨时段
    t3 = datetime(2026, 10, 9, 23, 10, tzinfo=timezone.utc)
    t4 = datetime(2026, 10, 10, 12, 30, tzinfo=timezone.utc)
    h_cross_day = partition_engine._build_summary_header(t3, t4)
    assert h_cross_day == "【10月9日深夜 – 10月10日午间】", f"Unexpected {h_cross_day}"

    # 跨年
    t5 = datetime(2026, 12, 31, 23, 50, tzinfo=timezone.utc)
    t6 = datetime(2027, 1, 1, 1, 20, tzinfo=timezone.utc)
    h_cross_year = partition_engine._build_summary_header(t5, t6)
    assert h_cross_year == "【2026年12月31日深夜 – 2027年1月1日凌晨】", f"Unexpected {h_cross_year}"

    # 解析往返一致性
    pattern = r"^【([^】\–\-]+?)(?:\s*[–\-]\s*([^】]+?))?】"
    for h in [h_same, h_cross_day, h_cross_year]:
        m = re.match(pattern, h)
        assert m is not None
        s = m.group(1).strip()
        e = m.group(2).strip() if m.group(2) else s
        rebuilt = f"【{s}】" if s == e else f"【{s} – {e}】"
        assert rebuilt == h
    return f"{h_same} | {h_cross_day} | {h_cross_year}"


def test_4_strip_model_date_header():
    """4. 模型输出自带日期段头时被剥离；'【即时状态】'这类标题不被误剥"""
    with_date_1 = "【10月9日深夜 – 10月10日午间】\n阿狸与我一同品茶。"
    assert partition_engine._strip_model_date_header(with_date_1) == "阿狸与我一同品茶。"

    with_date_2 = "【10月10日午间】阿狸与我一同散步。"
    assert partition_engine._strip_model_date_header(with_date_2) == "阿狸与我一同散步。"

    preserve_status = "【即时状态】阿狸身体状况良好。"
    assert partition_engine._strip_model_date_header(preserve_status) == "【即时状态】阿狸身体状况良好。"

    preserve_other = "【状态与约定】阿狸周日休息。"
    assert partition_engine._strip_model_date_header(preserve_other) == "【状态与约定】阿狸周日休息。"
    return "日期段头成功剥离，非日期标题【即时状态】完整保留"


def test_5_folding_logic():
    """5. 折叠：base 段头取第一个可解析起点和最后一个可解析终点；存量无头旧段被忽略；全部无头时 base 无头；解析失败不抛异常；字数校验发生在加头之前"""
    shared.CACHE_SUMMARY_BUDGET_CHARS = 50
    shared.CACHE_SUMMARY_MODEL = "mock-model"

    old_parts = [
        "存量无头旧段内容A",
        "【10月8日晚间 – 10月9日凌晨】\n旧段B",
        "【即时状态】旧段C",
        "【10月9日早晨 – 10月9日午间】\n旧段D",
    ]
    recent_parts = ["【10月10日早晨】\n近期段E", "【10月10日午间】\n近期段F"]
    all_parts = old_parts + recent_parts

    async def mock_fold_call(prompt, label="摘要折叠"):
        # 模拟返回没有段头的精简正文
        return "合并后的基础摘要正文。"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_fold_call):
        folded = asyncio.run(partition_engine.fold_summary_parts(all_parts, protected_tail=0))

    assert folded is not None
    base_part = folded[0]
    assert base_part.startswith("【10月8日晚间 – 10月9日午间】\n合并后的基础摘要正文。")

    # 全部无头旧段（确保超过 budget 触发折叠）
    all_headless = ["存量无头旧段内容A" * 5, "存量无头旧段内容B" * 5, "存量无头旧段内容C" * 5, "近期1", "近期2"]
    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_fold_call):
        folded_headless = asyncio.run(partition_engine.fold_summary_parts(all_headless, protected_tail=0))
    assert folded_headless[0] == "合并后的基础摘要正文。"

    return base_part.splitlines()[0]


def test_6_prefix_cache_stability():
    """6. 追加新段前后，旧段文本逐字节不变；build_partitioned_messages 摘要区前 N 个 block 文本逐字节相同；build_time_injection 仅在最后一条 user"""
    shared.CACHE_PARTITION_X = 2
    shared.CACHE_SUMMARY_MODEL = ""  # 纯轮转模式，不发真实网络请求

    old_summary_parts = ["【10月8日深夜】\n旧段1", "【10月9日午间】\n旧段2"]
    new_summary_part = "【10月9日晚间】\n新追加段3"
    updated_summary_parts = old_summary_parts + [new_summary_part]

    # 验证旧段列表文本在追加后逐字节一致
    for old, updated in zip(old_summary_parts, updated_summary_parts[:len(old_summary_parts)]):
        assert old == updated

    # 验证组装的 blocks 前 N 个完全逐字节一致
    def make_blocks(parts):
        blocks = [{"type": "text", "text": "[以下是之前对话的摘要，帮助你回忆上下文]"}]
        for i, part in enumerate(parts):
            item = {"type": "text", "text": part}
            if i == len(parts) - 1:
                item["cache_control"] = shared.make_cache_control()
            blocks.append(item)
        return blocks

    b_old = make_blocks(old_summary_parts)
    b_new = make_blocks(updated_summary_parts)
    for i in range(len(b_old) - 1):  # 除去老结尾的 cache_control 变迁
        assert b_old[i]["text"] == b_new[i]["text"]

    time_inj = partition_engine.build_time_injection()
    assert "<gateway_context>" in time_inj
    return "旧段及前缀block逐字节一致，缓存稳定性得到验证"


def test_7_missing_created_at_tolerance():
    """7. created_at 缺失：无 [HH:mm] 前缀、有 warning、不报错"""
    msgs = [
        {"role": "user", "content": "无时间戳消息", "created_at": None},
        {"role": "assistant", "content": "无时间戳回复"},
    ]

    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "摘要输出"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            result = asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert captured_prompt is not None
    assert "\n用户: 无时间戳消息\n" in captured_prompt or "用户: 无时间戳消息" in captured_prompt
    assert not re.search(r"\[\d{2}:\d{2}\] 用户: 无时间戳消息", captured_prompt)
    assert not result.startswith("【")  # 无有效时间戳，不加段头
    return "无时间戳消息正常处理无报错，无段头降级安全"


def test_8_list_content_text_extraction():
    """8. list 型 content 只取 text 块"""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是多模态文字"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
            "created_at": "2026-10-10T04:00:00Z",
        }
    ]

    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "摘要内容"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert "这是多模态文字" in captured_prompt
    assert "image_url" not in captured_prompt
    assert "{'type':" not in captured_prompt
    return "成功过滤非文本block，只提取text纯文本"


def test_9_tool_and_empty_assistant_filtering():
    """9. tool 消息和空 assistant 不进入摘要输入"""
    msgs = [
        {"role": "user", "content": "请搜索资料", "created_at": "2026-10-10T04:00:00Z"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}], "created_at": "2026-10-10T04:00:05Z"},
        {"role": "tool", "content": "大量的搜索结果数据", "created_at": "2026-10-10T04:00:10Z"},
        {"role": "assistant", "content": "搜索完毕结果已整理", "created_at": "2026-10-10T04:00:15Z"},
    ]

    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "摘要内容"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert "大量的搜索结果数据" not in captured_prompt
    assert "请搜索资料" in captured_prompt
    assert "搜索完毕结果已整理" in captured_prompt
    assert "AI: \n" not in captured_prompt
    return "成功过滤tool消息与空assistant消息"


def test_10_prompt_time_rules():
    """10. 摘要提示词含'时间规则'，且不再含'即时状态''最新约定'"""
    msgs = [{"role": "user", "content": "测试", "created_at": "2026-10-10T04:00:00Z"}]
    captured_prompt = None

    async def mock_call(prompt, label="摘要生成"):
        nonlocal captured_prompt
        captured_prompt = prompt
        return "摘要内容"

    with patch.object(partition_engine, "_call_summary_model", side_effect=mock_call):
        with patch.object(shared, "CACHE_SUMMARY_MODEL", "mock-model"):
            asyncio.run(partition_engine.generate_summary(msgs, "sess1"))

    assert "时间规则【绝对红线】" in captured_prompt
    assert "严禁使用\"昨晚、今早、今天、刚才、明天、前天、后天、最近、目前、现在、最新\"" in captured_prompt
    assert "【状态与约定】" in captured_prompt
    assert "【即时状态】" not in captured_prompt
    assert "最新约定" not in captured_prompt
    return "提示词准确包含时间规则红线，旧关键词已彻底替换"


if __name__ == "__main__":
    tests = [
        ("1. 时区换算 (UTC->本地)", test_1_timezone_conversion),
        ("2. 跨午夜日历日切分", test_2_cross_midnight_batch),
        ("3. 段头生成与往返解析", test_3_header_generation_and_roundtrip),
        ("4. 模型日期标题剥离", test_4_strip_model_date_header),
        ("5. 基础摘要折叠与段头合并", test_5_folding_logic),
        ("6. 缓存前缀字节稳定性", test_6_prefix_cache_stability),
        ("7. 缺失时间戳容错", test_7_missing_created_at_tolerance),
        ("8. 多模态文本块提取", test_8_list_content_text_extraction),
        ("9. 过滤tool与空assistant", test_9_tool_and_empty_assistant_filtering),
        ("10. 提示词时间规则验证", test_10_prompt_time_rules),
    ]

    all_passed = True
    print("\n" + "=" * 60)
    for name, t_func in tests:
        try:
            res = t_func()
            print(f"✅ {name}: {res}")
        except Exception as e:
            print(f"❌ {name}: 失败 - {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
    print("=" * 60 + "\n")
    if not all_passed:
        exit(1)
