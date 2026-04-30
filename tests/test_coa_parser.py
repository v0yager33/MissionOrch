"""COATableParser 单元测试。"""

import pytest

from missionorch_lc.core.coa_parser import COAParseError, COATableParser


@pytest.fixture
def parser():
    return COATableParser()


SAMPLE_COA_TABLE = """# COA方案：联合防空作战方案

## 方案概述
对东部海域进行联合防空，确保区域制空权。

## COA矩阵

| 作战单元 | 阶段1：部署展开 *(战场态势明确 → 触发)* | 阶段2：前沿拦截 *(敌方空中威胁侦测 → 触发)* | 阶段3：纵深防御 *(敌突破前沿 → 触发)* |
| :--- | :--- | :--- | :--- |
| 空军战斗机群 (AF/FG) *(主要打击力量)* | 1. 完成战备等级转进; 2. 前出至预定巡逻空域 | 1. 对敌机实施拦截; 2. 掩护地面防空部队展开 | 1. 实施追击歼灭; 2. 为增援力量提供掩护 |
| 海军舰艇编队 (NV/SF) *(海上防空支援)* | 1. 进入指定海域; 2. 展开区域防空雷达 | 1. 提供中远程防空火力; 2. 对低空目标实施拦截 | 1. 加强要地防空; 2. 实施电子干扰 |
| 陆基防空部队 (AD/GF) *(要地防护)* | 1. 完成阵地配置; 2. 雷达开机进入战斗状态 | 1. 对中高空目标射击; 2. 与空军协调射击区域 | 1. 转入近程防御; 2. 掩护战略要地 |

## 效果链

| 效果ID | 效果描述 | 衡量指标 | 达成单元 |
| :--- | :--- | :--- | :--- |
| E1 | 夺取局部制空权 | 敌机击落比≥3:1 | AF/FG; NV/SF |
| E2 | 保护战略要地 | 要地受损率＜5% | AD/GF; NV/SF |

## 决策点

| 决策点ID | 所属阶段 | 触发条件 | 选项 |
| :--- | :--- | :--- | :--- |
| DP1 | 阶段2 | 敌方投入第二梯队 | 若敌增兵则请求增援；若敌退缩则转入追击 |
| DP2 | 阶段3 | 弹药消耗超过60% | 若弹药充足则持续防御；若不足则梯次后撤 |

## 关键风险

| 风险类别 | 风险描述 | 缓解措施 |
| :--- | :--- | :--- |
| 作战 | 电子干扰导致雷达失效 | 部署多源探测网络 |
| 后勤 | 远程投送弹药补给困难 | 预置前沿弹药库 |
"""


class TestCOATableParser:
    """COATableParser 解析能力测试。"""

    def test_parse_success(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        assert coa.name == "联合防空作战方案"
        assert "联合防空" in coa.description

    def test_phases_extracted(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        assert len(coa.phases) == 3
        assert coa.phases[0].name == "部署展开"
        assert coa.phases[1].name == "前沿拦截"
        assert coa.phases[2].name == "纵深防御"

    def test_units_extracted(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        assert len(coa.units) == 3
        unit_names = [u.name for u in coa.units]
        assert any("空军" in name for name in unit_names)
        assert any("海军" in name for name in unit_names)
        assert any("陆基" in name or "防空" in name for name in unit_names)

    def test_matrix_cells(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        # 3 units × 3 phases = 9 cells
        assert len(coa.matrix) == 9
        # 每个 cell 至少有 1 个 action
        for action in coa.matrix:
            assert len(action.actions) >= 1

    def test_effects_chain(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        assert len(coa.effects_chain) == 2
        assert coa.effects_chain[0].effect_id == "E1"
        assert "制空权" in coa.effects_chain[0].description

    def test_decision_points(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        assert len(coa.decision_points) == 2
        assert coa.decision_points[0].dp_id == "DP1"
        assert "第二梯队" in coa.decision_points[0].condition

    def test_critical_risks(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        assert len(coa.critical_risks) >= 2
        categories = [r.get("category", "") for r in coa.critical_risks]
        assert "作战" in categories

    def test_empty_input_raises(self, parser: COATableParser):
        with pytest.raises(COAParseError):
            parser.parse("")

    def test_short_input_raises(self, parser: COATableParser):
        with pytest.raises(COAParseError):
            parser.parse("short text")

    def test_helper_get_actions(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        if coa.units and coa.phases:
            actions = coa.get_actions(coa.units[0].unit_id, coa.phases[0].phase_id)
            assert isinstance(actions, list)

    def test_helper_get_unit_actions(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        if coa.units:
            unit_actions = coa.get_unit_actions(coa.units[0].unit_id)
            assert isinstance(unit_actions, dict)
            assert len(unit_actions) == len(coa.phases)

    def test_helper_get_phase_actions(self, parser: COATableParser):
        coa = parser.parse(SAMPLE_COA_TABLE)
        if coa.phases:
            phase_actions = coa.get_phase_actions(coa.phases[0].phase_id)
            assert isinstance(phase_actions, dict)
            assert len(phase_actions) == len(coa.units)
