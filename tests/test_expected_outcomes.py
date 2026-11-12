"""预期战果抽取 / 归一化 / 合并 / 渲染的单元测试。"""

import pytest

from missionorch_lc.core.expected_outcomes import (
    coerce_outcomes,
    extract_from_mission,
    format_target,
    merge_outcomes,
    normalize_outcome,
    outcomes_to_markdown,
    outcomes_to_prompt,
    parse_target,
    table_rows_to_outcomes,
)
from missionorch_lc.schemas.coa import ExpectedOutcome


@pytest.mark.parametrize(
    "text, expected",
    [
        ("击落 J-16 ≥ 4 架", (">=", "4", "架")),
        ("我方损失不超过1架", ("<=", "1", "架")),
        ("摧毁 6 架以上", (">=", "6", "架")),
        ("压制率达到 80%", (">=", "80", "%")),
        ("任务完成时间 30~45 分钟", ("range", "30-45", "分钟")),
        ("压制持续 20 至 30 分钟", ("range", "20-30", "分钟")),
        ("时间 ~30 分钟", ("≈", "30", "分钟")),
    ],
)
def test_parse_target(text, expected):
    assert parse_target(text)[:3] == expected


def test_parse_target_without_number():
    assert parse_target("我方零损失") == ("", "", "", -1)
    assert parse_target("") == ("", "", "", -1)


def test_format_target_roundtrip():
    assert format_target({"comparator": ">=", "target_value": "4", "unit": "架"}) == "≥ 4 架"
    assert format_target({"comparator": "range", "target_value": "30-45", "unit": "分钟"}) == "30-45 分钟"
    assert format_target({"target_value": ""}) == ""


def test_normalize_from_text_defaults():
    o = normalize_outcome("击落敌机 ≥ 4 架", source="user")
    assert o["metric"] == "击落敌机"
    assert (o["comparator"], o["target_value"], o["unit"]) == (">=", "4", "架")
    assert o["side"] == "RED"
    assert o["source"] == "user"
    # 预留给仿真回填的字段
    assert o["actual_value"] is None
    assert o["status"] == "PENDING"
    assert set(o) == set(ExpectedOutcome.model_fields)


def test_normalize_dict_aliases_and_target_string():
    o = normalize_outcome(
        {"id": "EO-9", "desc": "压制雷达", "target": "≥ 80 %", "来源": "方案", "effects": "E1；E2"}
    )
    assert o["outcome_id"] == "EO-9"
    assert (o["comparator"], o["target_value"], o["unit"]) == (">=", "80", "%")
    assert o["source"] == "planner"
    assert o["linked_effects"] == ["E1", "E2"]


def test_normalize_keeps_own_source():
    o = normalize_outcome({"description": "x ≥ 1 个", "source": "输入"}, source="user")
    assert o["source"] == "mission_input"


def test_coerce_outcomes_variants():
    assert coerce_outcomes(None) == []
    assert coerce_outcomes("") == []
    items = coerce_outcomes("A ≥ 1 个；B ≤ 2 架", source="user")
    assert [o["outcome_id"] for o in items] == ["EO-1", "EO-2"]
    assert all(o["source"] == "user" for o in items)
    mixed = coerce_outcomes(["击落 4 架", {"description": "压制雷达 ≥ 2 部"}], source="user")
    assert len(mixed) == 2 and mixed[1]["target_value"] == "2"


def test_extract_heading_list():
    mission = (
        "任务：压制防空\n\n## 预期战果\n- 摧毁 S-400 阵地 ≥ 2 个\n"
        "- 击落 J-16 ≥ 4 架；我方损失 ≤ 1 架\n\n## 其他\n正文不应被抽取 ≥ 9 个\n"
    )
    items = extract_from_mission(mission)
    assert [o["description"] for o in items] == [
        "摧毁 S-400 阵地 ≥ 2 个",
        "击落 J-16 ≥ 4 架",
        "我方损失 ≤ 1 架",
    ]
    assert [o["outcome_id"] for o in items] == ["EO-1", "EO-2", "EO-3"]
    assert all(o["source"] == "mission_input" for o in items)
    assert items[2]["side"] == "BLUE"


@pytest.mark.parametrize("alias", ["预期效果", "预计目标", "作战指标", "Expected Outcomes"])
def test_extract_inline_aliases(alias):
    items = extract_from_mission(f"背景说明。\n{alias}：雷达压制率≥80%；我方零损失\n后续正文 ≥ 7 架")
    assert [o["description"] for o in items] == ["雷达压制率≥80%", "我方零损失"]


def test_extract_table():
    mission = (
        "## 预期效果\n"
        "| 预期战果 | 衡量指标 | 目标值 |\n| --- | --- | --- |\n"
        "| 摧毁雷达 | 雷达毁伤数 | ≥ 3 部 |\n"
    )
    items = extract_from_mission(mission)
    assert len(items) == 1
    assert (items[0]["metric"], items[0]["comparator"], items[0]["target_value"]) == (
        "雷达毁伤数", ">=", "3"
    )


def test_extract_none():
    assert extract_from_mission("") == []
    assert extract_from_mission("## 任务\n压制敌防空 ≥ 2 个") == []


def test_table_rows_skip_empty():
    rows = [{"预期战果": "", "目标值": ""}, {"预期战果": "压制", "目标值": "≥ 1 个"}]
    out = table_rows_to_outcomes(rows, source="planner")
    assert len(out) == 1 and out[0]["source"] == "planner"


def test_merge_keeps_every_required_item():
    planner = [
        {"outcome_id": "EO-1", "description": "摧毁 S-400 阵地", "metric": "SAM阵地毁伤数",
         "comparator": ">=", "target_value": "2", "unit": "个", "source": "planner",
         "phase_id": "Phase_4"},
    ]
    required = coerce_outcomes(["摧毁 S-400 阵地 ≥ 2 个", "击落 J-16 ≥ 4 架"], source="user")
    merged = merge_outcomes(planner, required)
    assert len(merged) == 2
    # 匹配到的条目：保留方案补充的字段，来源以输入为准
    assert merged[0]["phase_id"] == "Phase_4"
    assert merged[0]["source"] == "user"
    # 未匹配到的条目原样追加并重新编号
    assert merged[1]["description"] == "击落 J-16 ≥ 4 架"
    assert merged[1]["outcome_id"] == "EO-2"


def test_merge_keep_primary_source():
    primary = coerce_outcomes("击落 J-16 ≥ 4 架", source="user")
    extra = coerce_outcomes("击落 J-16 ≥ 4 架", source="mission_input")
    merged = merge_outcomes(primary, extra, keep_primary_source=True)
    assert len(merged) == 1 and merged[0]["source"] == "user"
    assert merge_outcomes(primary, extra)[0]["source"] == "mission_input"


def test_merge_does_not_touch_actual_fields():
    primary = [{"description": "压制雷达 ≥ 2 部", "actual_value": "3", "status": "MET"}]
    merged = merge_outcomes(primary, [{"description": "压制雷达 ≥ 2 部", "status": "PENDING"}])
    assert merged[0]["actual_value"] == "3" and merged[0]["status"] == "MET"


def test_outcomes_to_markdown():
    items = coerce_outcomes("击落 J-16 ≥ 4 架", source="user")
    md = outcomes_to_markdown(items)
    assert "| 战果ID |" in md and "≥ 4 架" in md and "| 用户 |" in md
    assert "仿真实测" not in md
    full = outcomes_to_markdown(items, with_actual=True)
    assert "仿真实测" in full and "待仿真" in full
    assert outcomes_to_markdown([]) == ""


def test_outcomes_to_prompt():
    assert outcomes_to_prompt([]) == ""
    text = outcomes_to_prompt(coerce_outcomes("击落 J-16 ≥ 4 架", source="user"))
    assert "第八节" in text and "J-16" in text
