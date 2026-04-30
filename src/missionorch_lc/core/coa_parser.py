"""COA 矩阵解析器 —— 将 Planner 输出的自然语言 Markdown 表格解析为 COA 对象。"""

import logging
import re
from typing import Dict, List, Optional

from ..schemas.coa import COA, Action, DecisionPoint, Effect, Phase, Unit

logger = logging.getLogger(__name__)


class COAParseError(Exception):
    """COA 表格解析错误。"""


class COATableParser:
    """将自然语言 Markdown COA 矩阵表格解析为 COA 对象。"""

    def parse(self, table_text: str) -> COA:
        if not table_text or len(table_text.strip()) < 50:
            raise COAParseError("COA table text is empty or too short")

        logger.info("Parsing COA matrix (%d chars)", len(table_text))
        sections = self._split_sections(table_text)

        coa = COA(
            name=self._parse_name(sections),
            description=self._parse_description(sections),
            metadata={"source": "natural_language_matrix"},
        )
        coa.phases, coa.units, coa.matrix = self._parse_coa_matrix(sections)
        coa.effects_chain = self._parse_effects_chain(sections)
        coa.decision_points = self._parse_decision_points(sections)
        coa.critical_risks = self._parse_critical_risks(sections)

        logger.info(
            "Parsed COA: %d phases, %d units, %d matrix cells, %d effects, %d decision points",
            len(coa.phases), len(coa.units), len(coa.matrix),
            len(coa.effects_chain), len(coa.decision_points),
        )
        return coa

    # ── 分段 ──
    def _split_sections(self, text: str) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        current_key = "header"
        current_lines: List[str] = []

        for line in text.split("\n"):
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
        return sections

    def _find_section(self, sections: Dict[str, str], *keywords: str) -> Optional[str]:
        for key, value in sections.items():
            key_lower = key.lower()
            for keyword in keywords:
                if keyword.lower() in key_lower:
                    return value
        return None

    # ── 基础字段 ──
    def _parse_name(self, sections: Dict[str, str]) -> str:
        header = sections.get("header", "")
        title_match = re.search(r"#\s+COA方案[：:]\s*(.+)", header)
        if title_match:
            return title_match.group(1).strip()
        return "Generated COA"

    def _parse_description(self, sections: Dict[str, str]) -> str:
        overview = self._find_section(sections, "方案概述", "概述", "overview")
        return overview.strip() if overview else ""

    # ── 核心矩阵 ──
    def _parse_coa_matrix(self, sections: Dict[str, str]):
        content = self._find_section(sections, "COA矩阵", "coa matrix", "矩阵", "COA Matrix")
        if not content:
            logger.warning("COA matrix section not found")
            return [], [], []

        table_lines = [
            line.strip() for line in content.split("\n") if line.strip() and "|" in line
        ]
        if len(table_lines) < 3:
            return [], [], []

        header_cells = self._split_table_row(table_lines[0])
        phases = self._parse_phase_headers(header_cells[1:])

        data_lines = [
            line for line in table_lines[1:]
            if not self._is_separator_row(self._split_table_row(line))
        ]

        units: List[Unit] = []
        matrix: List[Action] = []
        for line in data_lines:
            cells = self._split_table_row(line)
            if not cells:
                continue

            unit = self._parse_unit_cell(cells[0], len(units))
            if not unit:
                continue
            units.append(unit)

            action_cells = cells[1:]
            for idx, phase in enumerate(phases):
                cell_text = action_cells[idx].strip() if idx < len(action_cells) else ""
                matrix.append(Action(
                    unit_id=unit.unit_id,
                    phase_id=phase.phase_id,
                    actions=self._parse_action_cell(cell_text),
                ))

        return phases, units, matrix

    def _is_separator_row(self, cells: List[str]) -> bool:
        separator_pattern = re.compile(r"^[-:\s]*$")
        return all(separator_pattern.match(cell) for cell in cells)

    def _parse_phase_headers(self, header_cells: List[str]) -> List[Phase]:
        phases: List[Phase] = []
        for idx, cell in enumerate(header_cells):
            cell_clean = cell.replace("<br>", "\n").replace("<BR>", "\n")

            name_match = re.search(
                r"阶段\s*\d+\s*[:：]\s*(.+?)(?:\s*\(|$|\n|\*)", cell_clean
            )
            phase_name = (
                name_match.group(1).strip() if name_match else cell_clean.split("\n")[0].strip()
            )

            paren_contents = re.findall(r"[*（(]\s*([^*()）]+?)\s*[*）)]", cell_clean)
            transition_trigger = ""
            objective = ""
            for paren_content in paren_contents:
                content = paren_content.strip()
                is_english_only = re.match(r"^[A-Za-z\s&/]+$", content)
                has_time_marker = re.search(r"H[+-]", content)
                has_trigger_arrow = "→" in content or "->" in content

                if has_trigger_arrow or has_time_marker:
                    transition_trigger = content
                elif not is_english_only and not objective:
                    objective = content

            phases.append(Phase(
                phase_id=f"Phase_{idx + 1}",
                name=phase_name,
                transition_trigger=transition_trigger,
                objective=objective,
            ))
        return phases

    def _parse_unit_cell(self, cell_text: str, unit_index: int) -> Optional[Unit]:
        cell_clean = cell_text.replace("<br>", "\n").replace("<BR>", "\n").strip()
        if not cell_clean:
            return None

        first_line = cell_clean.split("\n")[0].strip()
        unit_name = re.sub(r"\s*\*\(.+?\)\*\s*$", "", first_line).strip()

        role_match = re.search(r"\*\(\s*(.+?)\s*\)\*", cell_clean)
        role = role_match.group(1).strip() if role_match else ""

        unit_id_match = re.search(r"[（(]\s*([A-Za-z0-9\-/]+)\s*[）)]", unit_name)
        if unit_id_match:
            unit_id = unit_id_match.group(1).replace("/", "_").replace("-", "_")
        else:
            unit_id = f"Unit_{unit_index + 1}"

        return Unit(unit_id=unit_id, name=unit_name, role=role)

    def _parse_action_cell(self, cell_text: str) -> List[str]:
        if not cell_text or cell_text.strip() in ("-", "—", "N/A", ""):
            return []

        text = cell_text.replace("<br>", "\n").replace("<BR>", "\n")

        numbered_items = re.split(r"\n?\s*\d+\.\s+", text)
        numbered_items = [item.strip() for item in numbered_items if item.strip()]
        if numbered_items:
            return numbered_items

        if ";" in text or "；" in text:
            return [item.strip() for item in re.split(r"[;；]+", text) if item.strip()]

        if "\n" in text:
            return [item.strip() for item in text.split("\n") if item.strip()]

        return [text.strip()] if text.strip() else []

    # ── 效果链 / 决策点 / 风险 ──
    def _parse_effects_chain(self, sections: Dict[str, str]) -> List[Effect]:
        content = self._find_section(sections, "效果链", "effects")
        if not content:
            return []

        effects: List[Effect] = []
        for row in self._parse_markdown_table(content):
            effect_id = self._get_cell(row, "效果ID", "effect_id", "ID")
            if not effect_id:
                continue
            effects.append(Effect(
                effect_id=effect_id,
                description=self._get_cell(row, "效果描述", "description", "描述") or "",
                measures=self._split_list_value(
                    self._get_cell(row, "衡量指标", "measures", "指标")
                ),
                achieved_by=self._split_list_value(
                    self._get_cell(row, "达成单元", "达成任务", "achieved_by", "单元")
                ),
            ))
        return effects

    def _parse_decision_points(self, sections: Dict[str, str]) -> List[DecisionPoint]:
        content = self._find_section(sections, "决策点", "decision")
        if not content:
            return []

        decision_points: List[DecisionPoint] = []
        for row in self._parse_markdown_table(content):
            dp_id = self._get_cell(row, "决策点ID", "dp_id", "ID")
            if not dp_id:
                continue
            decision_points.append(DecisionPoint(
                dp_id=dp_id,
                phase_id=self._get_cell(row, "所属阶段", "phase", "阶段") or "",
                condition=self._get_cell(row, "触发条件", "condition", "条件") or "",
                options=self._parse_decision_options(
                    self._get_cell(row, "选项", "options") or ""
                ),
            ))
        return decision_points

    def _parse_critical_risks(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        content = self._find_section(sections, "关键风险", "risk", "风险")
        if not content:
            return []

        risks: List[Dict[str, str]] = []
        for row in self._parse_markdown_table(content):
            description = self._get_cell(row, "风险描述", "description", "描述")
            if not description:
                continue
            risks.append({
                "category": self._get_cell(row, "风险类别", "category", "类别") or "GENERAL",
                "description": description,
                "mitigation": self._get_cell(row, "缓解措施", "mitigation", "措施") or "",
            })
        return risks

    # ── Markdown 表格工具 ──
    def _parse_markdown_table(self, text: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        table_lines = [line for line in lines if "|" in line]
        if len(table_lines) < 3:
            return []

        headers = self._split_table_row(table_lines[0])
        data_lines = [
            line for line in table_lines[1:]
            if not self._is_separator_row(self._split_table_row(line))
        ]

        rows: List[Dict[str, str]] = []
        for line in data_lines:
            cells = self._split_table_row(line)
            rows.append({
                header: cells[idx].strip() if idx < len(cells) else ""
                for idx, header in enumerate(headers)
            })
        return rows

    def _split_table_row(self, line: str) -> List[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def _get_cell(self, row: Dict[str, str], *candidate_keys: str) -> Optional[str]:
        for key in candidate_keys:
            if key in row:
                value = row[key].strip()
                if value and value not in ("-", "—", "N/A"):
                    return value
            for row_key, row_value in row.items():
                if key.lower() in row_key.lower():
                    value = row_value.strip()
                    if value and value not in ("-", "—", "N/A"):
                        return value
        return None

    def _split_list_value(self, value: Optional[str]) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in re.split(r"[;；,，]+", value) if item.strip()]

    def _parse_decision_options(self, options_text: str) -> List[Dict[str, str]]:
        options: List[Dict[str, str]] = []
        for part in re.split(r"[;；]+", options_text):
            part = part.strip()
            if not part:
                continue

            if_then_match = re.match(
                r"(?:若|如果|if)\s*[（(]?\s*(.+?)\s*[）)]?"
                r"\s*(?:则|那么|then)\s*[（(]?\s*(.+?)\s*[）)]?\s*$",
                part, re.IGNORECASE,
            )
            if if_then_match:
                options.append({
                    "if": if_then_match.group(1).strip(),
                    "then": if_then_match.group(2).strip(),
                })
                continue

            colon_match = re.match(r"(.+?)\s*[:：]\s*(.+)", part)
            if colon_match:
                options.append({
                    "if": colon_match.group(1).strip(),
                    "then": colon_match.group(2).strip(),
                })
            else:
                options.append({"if": "default", "then": part})
        return options
