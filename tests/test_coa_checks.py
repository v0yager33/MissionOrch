"""确定性结构检查（coa_checks）与章节补丁合并（coa_sections）的单元测试。"""

import pytest

from missionorch_lc.core.coa_checks import check_coa
from missionorch_lc.core.coa_parser import COATableParser
from missionorch_lc.core.coa_sections import (
    is_full_document,
    merge_sections,
    section_key,
    split_sections,
)


@pytest.fixture
def parser():
    return COATableParser()


def _check(parser, text):
    return check_coa(parser.parse(text), text)


# ── parser 对完整模板的解析 ──


def test_parser_reads_full_template(parser, full_coa_text):
    coa = parser.parse(full_coa_text)
    assert [u.unit_id for u in coa.units] == ["B_EW_01", "B_ST_01", "B_AEW_01"]
    assert len(coa.phases) == 5
    # 模板里的列名是"达成实体"
    assert [e.achieved_by for e in coa.effects_chain] == [["B_EW_01"], ["B_ST_01"], ["B_AEW_01"]]
    sources = [o.source for o in coa.expected_outcomes]
    assert sources == ["mission_input", "planner"]
    first = coa.expected_outcomes[0]
    assert (first.comparator, first.target_value, first.unit) == (">=", "2", "个")
    assert first.linked_effects == ["E2"]
    assert first.actual_value is None and first.status == "PENDING"


# ── check_coa ──


def test_full_template_passes(parser, full_coa_text):
    report = _check(parser, full_coa_text)
    assert report.passed, report.issues
    assert report.metrics["expected_outcomes"] == 2
    assert report.metrics["phases"] == 5 and report.metrics["units"] == 3
    md = report.to_markdown()
    assert "必须修正：无" in md and "预期战果 2" in md


def test_unparseable_coa():
    report = check_coa(None, "随便写的文字")
    assert not report.passed
    assert "无法解析" in report.issues[0]


def test_missing_manifest_section(parser, full_coa_text):
    head, _, tail = full_coa_text.partition("## 六、实体清单（Entity Manifest）")
    text = head + "## 七、地理参考" + tail.split("## 七、地理参考", 1)[1]
    report = _check(parser, text)
    assert any("实体清单" in issue for issue in report.issues)


def test_fixed_time_trigger_is_issue(parser, full_coa_text):
    text = full_coa_text.replace("*(集结完成 →)*", "*(H+2 →)*")
    report = _check(parser, text)
    assert any("固定时间" in issue for issue in report.issues)


def test_first_and_last_phase(parser, full_coa_text):
    text = (
        full_coa_text.replace("阶段1: 起飞集结", "阶段1: 侦察监视")
        .replace("*(完成编队集结)*", "*(建立监视)*")
        .replace("阶段5: 脱离返航", "阶段5: 持续封锁")
        .replace("*(安全返航)*", "*(维持封锁)*")
    )
    issues = _check(parser, text).issues
    assert any("首阶段" in i for i in issues)
    assert any("末阶段" in i for i in issues)


def test_report_as_dict(parser, full_coa_text):
    data = _check(parser, full_coa_text).as_dict()
    assert data["passed"] is True and data["issues"] == []


# ── coa_sections ──


def test_section_key():
    assert section_key("## 六、实体清单（Entity Manifest）") == "实体清单"
    assert section_key("## 2. COA矩阵") == "coa矩阵"
    assert section_key("## 八、预期战果 (Expected Outcomes)") == "预期战果"


def test_is_full_document(full_coa_text):
    assert is_full_document(full_coa_text)
    assert not is_full_document("## 一、方案概述\n改写后的概述")
    assert not is_full_document("")


def test_merge_replaces_only_patched_section(full_coa_text):
    patch = "## 一、方案概述\n改为夜间突防，先压制后打击。\n"
    merged, touched = merge_sections(full_coa_text, patch)
    assert touched == ["一、方案概述"]
    assert "改为夜间突防" in merged
    assert "以电子战先行压制" not in merged

    _, before = split_sections(full_coa_text)
    _, after = split_sections(merged)
    assert [h for h, _ in after] == [h for h, _ in before]
    # 其余章节逐字保留
    for (h1, b1), (h2, b2) in zip(before[1:], after[1:]):
        assert h1 == h2 and b1.rstrip() == b2.rstrip()
    assert merged.startswith("# COA方案：X海域防空压制")


def test_merge_matches_heading_variants(full_coa_text):
    patch = "## 三、效果链（Effects Chain）\n| 效果ID | 效果描述 | 衡量指标 | 达成实体 |\n"
    merged, touched = merge_sections(full_coa_text, patch)
    assert touched == ["三、效果链"]
    assert merged.count("## 三、效果链") == 1


def test_merge_appends_unknown_section(full_coa_text):
    merged, touched = merge_sections(full_coa_text, "## 九、补充说明\n新增内容\n")
    assert touched == ["九、补充说明"]
    assert merged.rstrip().endswith("新增内容")


def test_merge_without_sections_is_noop(full_coa_text):
    merged, touched = merge_sections(full_coa_text, "没有任何章节标题的回复")
    assert merged == full_coa_text and touched == []


def test_merge_replaces_title(full_coa_text):
    merged, _ = merge_sections(full_coa_text, "# COA方案：新标题\n\n## 一、方案概述\n新的概述\n")
    assert merged.startswith("# COA方案：新标题")
    assert "X海域防空压制" not in merged
