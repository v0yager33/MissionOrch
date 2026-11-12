"""COA 转换器 —— 将 COA 对象转换为 JSON / YAML / Markdown / 扁平矩阵 / 压缩格式。"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List

import yaml

from ..schemas.coa import COA
from .expected_outcomes import outcomes_to_markdown

_OUTCOME_NOTE = (
    "> 本节只记录预期战果，不参与方案评分；想定脚本仿真后回填\"仿真实测/状态\"，"
    "用于效果比对与战报生成。"
)


def _as_texts(value: Any) -> List[str]:
    """str / list[str|dict] → list[str]（dict 取 name/description）。"""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    out: List[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name") or item.get("description") or json.dumps(item, ensure_ascii=False)
        if str(item).strip():
            out.append(str(item).strip())
    return out


class COATransformer:
    """COA 格式转换工具。"""

    @staticmethod
    def coa_to_json(coa: COA) -> str:
        return json.dumps(coa.model_dump(mode="json"), indent=2, ensure_ascii=False)

    @staticmethod
    def coa_to_yaml(coa: COA) -> str:
        return yaml.dump(coa.model_dump(), default_flow_style=False, allow_unicode=True)

    # ── Markdown 结构化输出 ──

    @staticmethod
    def coa_to_markdown(
        coa: COA,
        *,
        mission_input: str = "",
        analysis: Dict[str, Any] | None = None,
        score: float = 0.0,
        iterations: int = 0,
        validation: Dict[str, Any] | None = None,
        raw_coa_table: str = "",
        best_iteration: int = 0,
        stop_reason: str = "",
    ) -> str:
        """将 COA 对象渲染为结构化 Markdown 文档。

        输出格式与 scenario1_sample_output.txt 对齐，
        同时保留可被下游 COA 解析器（如 ScenClaw coa_parser）直接消费的规范表格。

        Args:
            coa: 已解析的 COA 对象
            mission_input: 原始任务描述
            analysis: Analyst 阶段产出的任务分析字典
            score: 最终评分
            iterations: 迭代次数
            validation: Validator 阶段产出
            raw_coa_table: Planner 原始 Markdown 文本（含 Entity Manifest / Geo Reference
                           等尚未进入 COA schema 的附加段落），作为回退渲染

        Returns:
            完整的 Markdown 字符串
        """
        parts: List[str] = []

        # ── 标题 + 概述 ──
        parts.append(f"# COA方案：{coa.name}")
        parts.append("")
        if coa.description:
            parts.extend(["## 一、方案概述", "", coa.description, ""])

        # ── 任务分析（如有） ──
        if analysis:
            parts.extend(COATransformer._render_analysis_section(analysis))

        # ── COA 矩阵 ──
        section_idx = 3 if analysis else 2
        cn_num = ["二", "三", "四", "五", "六", "七", "八", "九", "十"]
        idx = section_idx - 2
        cn = cn_num[idx] if idx < len(cn_num) else str(section_idx)
        parts.extend([f"## {cn}、COA矩阵", ""])

        if coa.phases and coa.units and coa.matrix:
            parts.append(COATransformer._render_coa_matrix_table(coa))
        elif raw_coa_table:
            # 回退：使用 Planner 原始文本中的矩阵段落
            matrix_section = COATransformer._extract_raw_section(raw_coa_table, "COA矩阵", "矩阵")
            parts.append(matrix_section or raw_coa_table)
        parts.append("")

        # ── 效果链 ──
        idx += 1
        cn = cn_num[idx] if idx < len(cn_num) else str(section_idx + 1)
        if coa.effects_chain:
            parts.extend([f"## {cn}、效果链", ""])
            parts.append(COATransformer._render_effects_table(coa))
            parts.append("")
            idx += 1

        # ── 决策点 ──
        cn = cn_num[idx] if idx < len(cn_num) else str(section_idx + 2)
        if coa.decision_points:
            parts.extend([f"## {cn}、决策点", ""])
            parts.append(COATransformer._render_decision_points_table(coa))
            parts.append("")
            idx += 1

        # ── 关键风险 ──
        cn = cn_num[idx] if idx < len(cn_num) else str(section_idx + 3)
        if coa.critical_risks:
            parts.extend([f"## {cn}、关键风险", ""])
            parts.append(COATransformer._render_risks_table(coa))
            parts.append("")
            idx += 1

        # ── 从原始 COA 文本提取 Entity Manifest / Geo Reference（如有）──
        if raw_coa_table:
            for keywords, section_cn in [
                (("实体清单", "Entity Manifest"), "实体清单（Entity Manifest）"),
                (("地理参考", "Geo Reference"), "地理参考（Geo Reference）"),
            ]:
                raw_section = COATransformer._extract_raw_section(raw_coa_table, *keywords)
                if raw_section:
                    cn = cn_num[idx] if idx < len(cn_num) else str(idx + 2)
                    parts.extend([f"## {cn}、{section_cn}", "", raw_section, ""])
                    idx += 1

        # ── 预期战果（只记录，预留仿真回填列）──
        if coa.expected_outcomes:
            cn = cn_num[idx] if idx < len(cn_num) else str(idx + 2)
            parts.extend([f"## {cn}、预期战果（Expected Outcomes）", "", _OUTCOME_NOTE, ""])
            parts.append(
                outcomes_to_markdown(
                    [o.model_dump() for o in coa.expected_outcomes], with_actual=True
                )
            )
            parts.append("")
            idx += 1

        # ── 元数据 ──
        parts.extend(["---", "", "## 生成元数据", ""])
        parts.append(f"- **迭代次数**: {iterations}")
        parts.append(f"- **最终得分**: {score}")
        if best_iteration:
            parts.append(f"- **采用版本**: 第 {best_iteration} 轮")
        if stop_reason:
            parts.append(f"- **停止原因**: {stop_reason}")
        parts.append(f"- **阶段数量**: {len(coa.phases)}")
        parts.append(f"- **作战单元**: {len(coa.units)}")
        parts.append(f"- **矩阵单元格**: {len(coa.matrix)}")
        parts.append(f"- **效果链**: {len(coa.effects_chain)}")
        parts.append(f"- **决策点**: {len(coa.decision_points)}")
        parts.append(f"- **预期战果**: {len(coa.expected_outcomes)}")
        parts.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if validation:
            is_valid = validation.get("is_valid", False)
            parts.append(f"- **验证结果**: {'通过' if is_valid else '未通过'}")
        parts.append("")

        return "\n".join(parts)

    # ── Markdown 渲染子方法 ──

    @staticmethod
    def _render_analysis_section(analysis: Dict[str, Any]) -> List[str]:
        """渲染 Analyst 阶段的任务分析结果。"""
        lines = ["## 任务分析", ""]
        intent = analysis.get("mission_intent", "")
        if intent:
            lines.append(f"**任务意图**: {intent}")
            lines.append("")

        # analyst 的 LLM 输出不一定严格符合 schema：条目可能是 str 也可能是 dict
        objectives = analysis.get("objectives") or []
        if objectives:
            lines.append(f"**目标 ({len(objectives)})**:")
            for obj in objectives:
                if isinstance(obj, dict):
                    lines.append(
                        f"- [{obj.get('id', '')}] ({obj.get('priority', '')}) "
                        f"{obj.get('description', '')}"
                    )
                else:
                    lines.append(f"- {obj}")
            lines.append("")

        constraints = analysis.get("constraints") or []
        if constraints:
            lines.append(f"**约束**: {'; '.join(_as_texts(constraints))}")
            lines.append("")

        entities = analysis.get("key_entities") or {}
        if isinstance(entities, dict) and entities:
            blue = _as_texts(entities.get("blue_forces"))
            red = _as_texts(entities.get("red_forces"))
            areas = _as_texts(entities.get("areas"))
            if blue:
                lines.append(f"**蓝方力量**: {', '.join(blue)}")
            if red:
                lines.append(f"**红方力量**: {', '.join(red)}")
            if areas:
                lines.append(f"**关键区域**: {', '.join(areas)}")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    @staticmethod
    def _render_coa_matrix_table(coa: COA) -> str:
        """渲染 COA 矩阵 Markdown 表格。"""
        # 表头：| 作战实体 | 阶段1 | 阶段2 | ... |
        phase_headers = [
            f"阶段{i+1}: {p.name}" for i, p in enumerate(coa.phases)
        ]
        header_row = "| 作战实体 | " + " | ".join(phase_headers) + " |"
        separator = "| :--- | " + " | ".join([":---" for _ in coa.phases]) + " |"

        data_rows: List[str] = []
        for unit in coa.units:
            cells = [f"**{unit.name}** ({unit.unit_id})" + (f" *({unit.role})*" if unit.role else "")]
            for phase in coa.phases:
                actions = coa.get_actions(unit.unit_id, phase.phase_id)
                cell_text = "<br>".join(
                    f"{i+1}. {a}" for i, a in enumerate(actions)
                ) if actions else "-"
                cells.append(cell_text)
            data_rows.append("| " + " | ".join(cells) + " |")

        return "\n".join([header_row, separator] + data_rows)

    @staticmethod
    def _render_effects_table(coa: COA) -> str:
        """渲染效果链表格。"""
        lines = [
            "| 效果ID | 效果描述 | 衡量指标 | 达成实体 |",
            "|--------|----------|----------|----------|",
        ]
        for effect in coa.effects_chain:
            measures = "; ".join(effect.measures) if effect.measures else "-"
            achieved = ", ".join(effect.achieved_by) if effect.achieved_by else "-"
            lines.append(
                f"| {effect.effect_id} | {effect.description} | {measures} | {achieved} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_decision_points_table(coa: COA) -> str:
        """渲染决策点表格。"""
        lines = [
            "| 决策点ID | 所属阶段 | 触发条件 | 选项 |",
            "|----------|----------|----------|------|",
        ]
        for dp in coa.decision_points:
            options_text = "; ".join(
                f"{opt.get('if', '')}: {opt.get('then', '')}" for opt in dp.options
            ) if dp.options else "-"
            lines.append(f"| {dp.dp_id} | {dp.phase_id} | {dp.condition} | {options_text} |")
        return "\n".join(lines)

    @staticmethod
    def _render_risks_table(coa: COA) -> str:
        """渲染关键风险表格。"""
        lines = [
            "| 风险类别 | 风险描述 | 缓解措施 |",
            "|----------|----------|----------|",
        ]
        for risk in coa.critical_risks:
            lines.append(
                f"| {risk.get('category', 'GENERAL')} | "
                f"{risk.get('description', '')} | "
                f"{risk.get('mitigation', '')} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_raw_section(raw_text: str, *keywords: str) -> str:
        """从 Planner 原始 Markdown 文本中按标题关键字提取段落。"""
        if not raw_text:
            return ""
        sections: Dict[str, str] = {}
        current_key = "header"
        current_lines: List[str] = []

        for line in raw_text.split("\n"):
            heading_match = re.match(r"^##\s+(.+)", line)
            if heading_match:
                if current_lines:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = heading_match.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections[current_key] = "\n".join(current_lines).strip()

        for key, value in sections.items():
            for keyword in keywords:
                if keyword.lower() in key.lower():
                    return value
        return ""

    @staticmethod
    def coa_to_flat_matrix(coa: COA) -> Dict[str, Any]:
        """扁平矩阵格式：`matrix[unit_id][phase_id] -> [action, ...]`。"""
        flat_matrix: Dict[str, Dict[str, List[str]]] = {}
        for unit in coa.units:
            flat_matrix[unit.unit_id] = {
                phase.phase_id: coa.get_actions(unit.unit_id, phase.phase_id)
                for phase in coa.phases
            }
        return {
            "phases": [p.model_dump() for p in coa.phases],
            "units": [u.model_dump() for u in coa.units],
            "matrix": flat_matrix,
            "effects": [e.model_dump() for e in coa.effects_chain],
            "decision_points": [dp.model_dump() for dp in coa.decision_points],
            "risks": coa.critical_risks,
            "expected_outcomes": [o.model_dump() for o in coa.expected_outcomes],
        }

    @staticmethod
    def coa_to_condensed_format(coa: COA) -> Dict[str, Any]:
        phase_summary = []
        for phase in coa.phases:
            unit_actions: Dict[str, List[str]] = {}
            for unit in coa.units:
                actions = coa.get_actions(unit.unit_id, phase.phase_id)
                if actions:
                    unit_actions[unit.unit_id] = actions
            phase_summary.append({
                "phase_id": phase.phase_id,
                "name": phase.name,
                "transition_trigger": phase.transition_trigger,
                "objective": phase.objective,
                "unit_actions": unit_actions,
            })

        return {
            "coa_id": coa.coa_id,
            "name": coa.name,
            "description": coa.description,
            "key_metrics": {
                "total_phases": len(coa.phases),
                "total_units": len(coa.units),
                "total_matrix_cells": len(coa.matrix),
                "total_effects": len(coa.effects_chain),
                "total_decision_points": len(coa.decision_points),
                "total_risks": len(coa.critical_risks),
                "total_expected_outcomes": len(coa.expected_outcomes),
            },
            "phases": phase_summary,
            "units": [
                {"unit_id": u.unit_id, "name": u.name, "role": u.role}
                for u in coa.units
            ],
            "risk_summary": COATransformer._summarize_risks(coa),
            "expected_outcomes": [
                {
                    "outcome_id": o.outcome_id,
                    "description": o.description,
                    "metric": o.metric,
                    "comparator": o.comparator,
                    "target_value": o.target_value,
                    "unit": o.unit,
                    "observable": o.observable,
                    "source": o.source,
                }
                for o in coa.expected_outcomes
            ],
        }

    @staticmethod
    def _summarize_risks(coa: COA) -> Dict[str, Any]:
        categories: Dict[str, int] = {}
        for risk in coa.critical_risks:
            cat = risk.get("category", "GENERAL")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_risks": len(coa.critical_risks),
            "risk_categories": categories,
        }

    @staticmethod
    def coa_from_dict(coa_dict: Dict[str, Any]) -> COA:
        return COA.model_validate(coa_dict)
