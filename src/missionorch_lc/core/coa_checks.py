"""COA 确定性结构检查 —— 不调用 LLM，毫秒级完成。

借鉴 nanobot 的做法：能用规则判定的问题（阶段数、起飞/返航、固定时间窗口、空单元格、
效果链实体引用、必需章节……）直接用代码查，不再花一次 LLM 调用。检查报告有三个用途：

1. 喂给 Judge，让它把注意力放在规则查不出的战术合理性上；
2. 喂给 Planner 作为修订指令的一部分；
3. 作为 Validator 的默认实现（``validator.use_llm: false``）。

预期战果只统计数量写进 ``metrics``，不产生 issue —— 它不参与评分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas.coa import COA

_TAKEOFF_RE = re.compile(r"起飞|集结|出动|编队|准备|takeoff|launch|form", re.IGNORECASE)
_RECOVER_RE = re.compile(r"返航|回收|着陆|撤离|撤出|脱离|recover|rtb|egress|land", re.IGNORECASE)
_FIXED_TIME_RE = re.compile(r"(?<![A-Za-z])[HT]\s*[+＋]\s*\d+", re.IGNORECASE)


@dataclass
class CheckReport:
    """确定性检查结果。``issues`` 是必须修改的问题，``warnings`` 是建议。"""

    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.issues

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }

    def to_markdown(self) -> str:
        m = self.metrics
        lines = [
            f"- 规模：{m.get('units', 0)} 个实体 × {m.get('phases', 0)} 个阶段，"
            f"{m.get('cells', 0)} 个单元格（空 {m.get('empty_cells', 0)}，"
            f"单薄 {m.get('thin_cells', 0)}）；效果 {m.get('effects', 0)}，"
            f"决策点 {m.get('decision_points', 0)}，风险 {m.get('risks', 0)}，"
            f"预期战果 {m.get('expected_outcomes', 0)}",
        ]
        if self.issues:
            lines.append("- 必须修正：")
            lines.extend(f"  {i}. {text}" for i, text in enumerate(self.issues, 1))
        else:
            lines.append("- 必须修正：无")
        if self.warnings:
            lines.append("- 建议：")
            lines.extend(f"  - {text}" for text in self.warnings)
        return "\n".join(lines)


def _unit_tokens(coa: COA) -> List[str]:
    tokens: List[str] = []
    for unit in coa.units:
        tokens.append(unit.unit_id.lower())
        tokens.append(unit.unit_id.replace("_", "-").lower())
        tokens.append(unit.name.lower())
    return [t for t in tokens if t]


def _ref_matches_unit(ref: str, tokens: List[str]) -> bool:
    ref_norm = ref.strip().strip("[]()（）").lower()
    if not ref_norm:
        return True
    ref_alt = ref_norm.replace("-", "_")
    return any(ref_norm in t or ref_alt in t or (len(t) > 3 and t in ref_norm) for t in tokens)


def check_coa(
    coa: Optional[COA],
    raw_text: str = "",
    *,
    min_phases: int = 5,
    min_units: int = 3,
    min_actions_per_cell: int = 2,
    min_effects: int = 3,
    min_decision_points: int = 2,
) -> CheckReport:
    """对解析后的 COA 和原始 Markdown 做结构检查。

    ``coa`` 为 None 表示解析失败，此时只能给出"格式不可解析"的 issue。
    """
    report = CheckReport()
    text = raw_text or ""

    if coa is None:
        report.issues.append("COA 矩阵无法解析：请严格按模板输出\"## 二、COA矩阵\"的 Markdown 表格")
        report.metrics.update(units=0, phases=0, cells=0)
        return report

    phases, units = coa.phases, coa.units
    cells = coa.matrix
    empty = [c for c in cells if not c.actions]
    thin = [c for c in cells if 0 < len(c.actions) < min_actions_per_cell]

    report.metrics.update(
        units=len(units),
        phases=len(phases),
        cells=len(cells),
        empty_cells=len(empty),
        thin_cells=len(thin),
        effects=len(coa.effects_chain),
        decision_points=len(coa.decision_points),
        risks=len(coa.critical_risks),
        expected_outcomes=len(coa.expected_outcomes),
    )

    if len(phases) < min_phases:
        report.issues.append(
            f"阶段数 {len(phases)} < {min_phases}：需覆盖起飞→出航→进入→交战→返航全周期"
        )
    if len(units) < min_units:
        report.issues.append(f"作战实体行数 {len(units)} < {min_units}：每个独立编队应单独占一行")

    if phases:
        if not _TAKEOFF_RE.search(f"{phases[0].name} {phases[0].objective}"):
            report.issues.append(f"首阶段\"{phases[0].name}\"不是起飞/集结：在最左侧补充起飞与编队集结阶段")
        if not _RECOVER_RE.search(f"{phases[-1].name} {phases[-1].objective}"):
            report.issues.append(f"末阶段\"{phases[-1].name}\"不是脱离/返航：在最右侧补充脱离与返航阶段")

    fixed_time = [p.phase_id for p in phases if _FIXED_TIME_RE.search(p.transition_trigger)]
    if fixed_time:
        report.issues.append(
            f"阶段转换使用了固定时间窗口（{', '.join(fixed_time)}）：改为条件/事件驱动触发"
        )
    missing_trigger = [p.phase_id for p in phases[:-1] if not p.transition_trigger]
    if missing_trigger:
        report.warnings.append(f"阶段缺少转换触发条件：{', '.join(missing_trigger[:6])}")

    unit_names = {u.unit_id: u.name for u in units}
    if empty:
        where = ", ".join(f"{unit_names.get(c.unit_id, c.unit_id)}@{c.phase_id}" for c in empty[:6])
        report.issues.append(f"{len(empty)} 个空单元格（{where}{' 等' if len(empty) > 6 else ''}）：每格需 2~5 个行动")
    if thin:
        where = ", ".join(f"{unit_names.get(c.unit_id, c.unit_id)}@{c.phase_id}" for c in thin[:6])
        message = f"{len(thin)} 个单元格行动少于 {min_actions_per_cell} 个（{where}{' 等' if len(thin) > 6 else ''}）"
        if cells and len(thin) / len(cells) > 0.3:
            report.issues.append(message)
        else:
            report.warnings.append(message)

    if len(coa.effects_chain) < min_effects:
        report.issues.append(f"效果链只有 {len(coa.effects_chain)} 个效果，至少 {min_effects} 个")
    tokens = _unit_tokens(coa)
    dangling = sorted({
        ref for effect in coa.effects_chain for ref in effect.achieved_by
        if not _ref_matches_unit(ref, tokens)
    })
    if dangling:
        report.warnings.append(f"效果链\"达成实体\"引用了矩阵中不存在的实体：{', '.join(dangling[:6])}")

    if len(coa.decision_points) < min_decision_points:
        report.issues.append(
            f"决策点只有 {len(coa.decision_points)} 个，至少 {min_decision_points} 个且每个 ≥2 个选项"
        )
    single_option = [dp.dp_id for dp in coa.decision_points if len(dp.options) < 2]
    if single_option:
        report.warnings.append(f"决策点选项少于 2 个：{', '.join(single_option)}")
    if not coa.critical_risks:
        report.warnings.append("缺少关键风险表")

    if text:
        if not re.search(r"^##\s*.*实体清单", text, re.MULTILINE):
            report.issues.append("缺少\"## 六、实体清单\"：仿真脚本生成依赖该表")
        if not re.search(r"^##\s*.*地理参考", text, re.MULTILINE):
            report.issues.append("缺少\"## 七、地理参考\"：所有 <TYPE:ID> 占位需在此登记")

    return report
