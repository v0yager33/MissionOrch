"""
COA 矩阵解析器

将 Planner 输出的自然语言 COA 矩阵表格解析为结构化的 COA 对象。
这是整个架构中唯一将自然语言转换为结构化数据的模块。

表格格式约定（与 planner.txt 提示词模板一致）：
- ## 一、方案概述          -> COA.description
- ## 二、COA矩阵           -> COA.phases + COA.units + COA.matrix
- ## 三、效果链            -> COA.effects_chain
- ## 四、决策点            -> COA.decision_points
- ## 五、关键风险          -> COA.critical_risks
"""

import logging
import re
from typing import List, Dict, Optional

from ..schemas.coa import COA, Phase, Unit, Action, Effect, DecisionPoint

logger = logging.getLogger(__name__)


class COAParseError(Exception):
    """COA 表格解析错误"""
    pass


class COATableParser:
    """将自然语言 Markdown COA 矩阵表格解析为 COA 对象"""

    def parse(self, table_text: str) -> COA:
        """
        解析完整的 COA 矩阵文本，返回 COA 对象。

        Args:
            table_text: Planner 输出的 COA 矩阵（Markdown 格式）

        Returns:
            解析后的 COA 对象

        Raises:
            COAParseError: 解析失败时抛出
        """
        if not table_text or len(table_text.strip()) < 50:
            raise COAParseError("COA table text is empty or too short")

        logger.info("Parsing COA matrix (%d chars)", len(table_text))

        sections = self._split_sections(table_text)

        name = self._parse_name(sections)
        description = self._parse_description(sections)
        phases, units, matrix = self._parse_coa_matrix(sections)
        effects_chain = self._parse_effects_chain(sections)
        decision_points = self._parse_decision_points(sections)
        critical_risks = self._parse_critical_risks(sections)

        coa = COA(
            name=name,
            description=description,
            phases=phases,
            units=units,
            matrix=matrix,
            effects_chain=effects_chain,
            decision_points=decision_points,
            critical_risks=critical_risks,
            metadata={"source": "natural_language_matrix"},
        )

        logger.info(
            "Parsed COA: %d phases, %d units, %d matrix cells, "
            "%d effects, %d decision points",
            len(coa.phases), len(coa.units), len(coa.matrix),
            len(coa.effects_chain), len(coa.decision_points),
        )
        return coa

    # ── 分段 ──

    def _split_sections(self, text: str) -> Dict[str, str]:
        """将 COA 表格按二级标题分段"""
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

        logger.debug("Split into %d sections: %s", len(sections), list(sections.keys()))
        return sections

    def _find_section(self, sections: Dict[str, str], *keywords: str) -> Optional[str]:
        """根据关键词查找对应的 section 内容"""
        for key, value in sections.items():
            key_lower = key.lower()
            for keyword in keywords:
                if keyword.lower() in key_lower:
                    return value
        return None

    # ── 解析各部分 ──

    def _parse_name(self, sections: Dict[str, str]) -> str:
        """解析方案名称"""
        header = sections.get("header", "")
        title_match = re.search(r"#\s+COA方案[：:]\s*(.+)", header)
        if title_match:
            return title_match.group(1).strip()
        return "Generated COA"

    def _parse_description(self, sections: Dict[str, str]) -> str:
        """解析方案概述"""
        overview = self._find_section(sections, "方案概述", "概述", "overview")
        if overview:
            return overview.strip()
        return ""

    def _parse_coa_matrix(self, sections: Dict[str, str]):
        """
        解析 COA 矩阵表格，提取 phases、units 和 matrix。

        矩阵格式：
        - 第一列：作战单元 / 职能
        - 后续列：各阶段（列标题包含阶段名、时间窗口、目标）
        - 单元格：该单元在该阶段的行动列表
        """
        content = self._find_section(
            sections, "COA矩阵", "coa matrix", "矩阵", "COA Matrix"
        )
        if not content:
            logger.warning("COA matrix section not found")
            return [], [], []

        # 提取表格行
        table_lines = [
            line.strip() for line in content.split("\n")
            if line.strip() and "|" in line
        ]

        if len(table_lines) < 3:
            logger.warning("COA matrix table has fewer than 3 lines")
            return [], [], []

        # 解析表头行 -> 提取阶段信息
        header_cells = self._split_table_row(table_lines[0])
        phase_headers = header_cells[1:]
        phases = self._parse_phase_headers(phase_headers)

        # 跳过分隔行，解析数据行
        data_lines = []
        for line in table_lines[1:]:
            cells = self._split_table_row(line)
            if self._is_separator_row(cells):
                continue
            data_lines.append(line)

        # 解析每一行 -> 提取单元和行动
        units: List[Unit] = []
        matrix: List[Action] = []

        for line in data_lines:
            cells = self._split_table_row(line)
            if not cells:
                continue

            unit_cell = cells[0]
            unit = self._parse_unit_cell(unit_cell, len(units))
            if not unit:
                continue
            units.append(unit)

            action_cells = cells[1:]
            for phase_idx, phase in enumerate(phases):
                cell_text = action_cells[phase_idx].strip() if phase_idx < len(action_cells) else ""
                actions_list = self._parse_action_cell(cell_text)
                matrix.append(Action(
                    unit_id=unit.unit_id,
                    phase_id=phase.phase_id,
                    actions=actions_list,
                ))

        logger.debug(
            "Parsed matrix: %d phases, %d units, %d cells",
            len(phases), len(units), len(matrix),
        )
        return phases, units, matrix

    def _is_separator_row(self, cells: List[str]) -> bool:
        """判断是否为 Markdown 表格分隔行"""
        separator_pattern = re.compile(r"^[-:\s]*$")
        return all(separator_pattern.match(cell) for cell in cells)

    def _parse_phase_headers(self, header_cells: List[str]) -> List[Phase]:
        """
        从表头单元格解析阶段信息。

        支持格式（条件驱动）：
        - "阶段1: 突防与压制 *(敌防空网络被压制后→)* *(瘫痪敌态势感知)*"
        - "阶段 1: 突防与压制 (Ingress & SEAD)<br>*(敌雷达被致盲后→)*<br>*(突破防线)*"

        兼容旧格式（时间窗口）：
        - "阶段1: 突防与压制 *(H-30min ~ H+05min)* *(突破防线)*"
        """
        phases: List[Phase] = []

        for idx, cell in enumerate(header_cells):
            cell_clean = cell.replace("<br>", "\n").replace("<BR>", "\n")

            # 提取阶段名称
            name_match = re.search(
                r"阶段\s*\d+\s*[:：]\s*(.+?)(?:\s*\(|$|\n|\*)",
                cell_clean,
            )
            phase_name = name_match.group(1).strip() if name_match else cell_clean.split("\n")[0].strip()

            # 提取所有括号/星号内的内容
            paren_contents = re.findall(
                r"[*（(]\s*([^*()）]+?)\s*[*）)]",
                cell_clean,
            )

            transition_trigger = ""
            objective = ""

            for paren_content in paren_contents:
                content = paren_content.strip()
                is_english_only = re.match(r"^[A-Za-z\s&/]+$", content)
                has_time_marker = re.search(r"H[+-]", content)
                has_trigger_arrow = "→" in content or "->" in content

                if has_trigger_arrow:
                    # 包含箭头的是转换触发条件
                    transition_trigger = content
                elif has_time_marker:
                    # 兼容旧格式：时间窗口也作为转换条件
                    transition_trigger = content
                elif not is_english_only and not objective:
                    # 非英文纯文本、非时间标记的是阶段目标
                    objective = content

            phase_id = "Phase_%d" % (idx + 1)
            phases.append(Phase(
                phase_id=phase_id,
                name=phase_name,
                transition_trigger=transition_trigger,
                objective=objective,
            ))

        return phases

    def _parse_unit_cell(self, cell_text: str, unit_index: int) -> Optional[Unit]:
        """
        解析作战单元单元格。

        支持格式：
        - "电子战飞机 (EA-18G) *(电磁掩护)*"
        - "电子战飞机 (EA-18G)<br>*(电磁掩护)*"
        """
        cell_clean = cell_text.replace("<br>", "\n").replace("<BR>", "\n").strip()
        if not cell_clean:
            return None

        first_line = cell_clean.split("\n")[0].strip()
        # 去掉尾部的 *(角色)* 部分
        unit_name = re.sub(r"\s*\*\(.+?\)\*\s*$", "", first_line).strip()

        # 提取职能角色
        role_match = re.search(r"\*\(\s*(.+?)\s*\)\*", cell_clean)
        role = role_match.group(1).strip() if role_match else ""

        # 生成 unit_id
        unit_id_match = re.search(r"[（(]\s*([A-Za-z0-9\-/]+)\s*[）)]", unit_name)
        if unit_id_match:
            unit_id = unit_id_match.group(1).replace("/", "_").replace("-", "_")
        else:
            unit_id = "Unit_%d" % (unit_index + 1)

        return Unit(
            unit_id=unit_id,
            name=unit_name,
            role=role,
        )

    def _parse_action_cell(self, cell_text: str) -> List[str]:
        """
        解析行动单元格，提取行动列表。

        支持格式：
        - "1. 行动A<br>2. 行动B<br>3. 行动C"
        - "1. 行动A 2. 行动B 3. 行动C"
        - "行动A; 行动B; 行动C"
        """
        if not cell_text or cell_text.strip() in ("-", "—", "N/A", ""):
            return []

        text = cell_text.replace("<br>", "\n").replace("<BR>", "\n")

        # 按编号列表解析
        numbered_items = re.split(r"\n?\s*\d+\.\s+", text)
        numbered_items = [item.strip() for item in numbered_items if item.strip()]
        if numbered_items:
            return numbered_items

        # 按分号分隔
        if ";" in text or "；" in text:
            items = re.split(r"[;；]+", text)
            return [item.strip() for item in items if item.strip()]

        # 按换行分隔
        if "\n" in text:
            items = text.split("\n")
            return [item.strip() for item in items if item.strip()]

        return [text.strip()] if text.strip() else []

    def _parse_effects_chain(self, sections: Dict[str, str]) -> List[Effect]:
        """解析效果链表格"""
        content = self._find_section(sections, "效果链", "effects")
        if not content:
            logger.warning("Effects chain section not found")
            return []

        rows = self._parse_markdown_table(content)
        effects: List[Effect] = []

        for row in rows:
            effect_id = self._get_cell(row, "效果ID", "effect_id", "ID")
            description = self._get_cell(row, "效果描述", "description", "描述")
            measures_raw = self._get_cell(row, "衡量指标", "measures", "指标")
            achieved_by_raw = self._get_cell(
                row, "达成单元", "达成任务", "achieved_by", "单元",
            )

            if not effect_id:
                continue

            measures = self._split_list_value(measures_raw)
            achieved_by = self._split_list_value(achieved_by_raw)

            effects.append(Effect(
                effect_id=effect_id,
                description=description or "",
                measures=measures,
                achieved_by=achieved_by,
            ))

        logger.debug("Parsed %d effects", len(effects))
        return effects

    def _parse_decision_points(self, sections: Dict[str, str]) -> List[DecisionPoint]:
        """解析决策点"""
        content = self._find_section(sections, "决策点", "decision")
        if not content:
            return []

        rows = self._parse_markdown_table(content)
        decision_points: List[DecisionPoint] = []

        for row in rows:
            dp_id = self._get_cell(row, "决策点ID", "dp_id", "ID")
            phase_id = self._get_cell(row, "所属阶段", "phase", "阶段")
            condition = self._get_cell(row, "触发条件", "condition", "条件")
            options_raw = self._get_cell(row, "选项", "options")

            if not dp_id:
                continue

            options = self._parse_decision_options(options_raw or "")

            decision_points.append(DecisionPoint(
                dp_id=dp_id,
                phase_id=phase_id or "",
                condition=condition or "",
                options=options,
            ))

        logger.debug("Parsed %d decision points", len(decision_points))
        return decision_points

    def _parse_critical_risks(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """解析关键风险"""
        content = self._find_section(sections, "关键风险", "risk", "风险")
        if not content:
            return []

        rows = self._parse_markdown_table(content)
        risks: List[Dict[str, str]] = []

        for row in rows:
            category = self._get_cell(row, "风险类别", "category", "类别")
            description = self._get_cell(row, "风险描述", "description", "描述")
            mitigation = self._get_cell(row, "缓解措施", "mitigation", "措施")

            if description:
                risks.append({
                    "category": category or "GENERAL",
                    "description": description,
                    "mitigation": mitigation or "",
                })

        logger.debug("Parsed %d risks", len(risks))
        return risks

    # ── Markdown 表格解析工具 ──

    def _parse_markdown_table(self, text: str) -> List[Dict[str, str]]:
        """解析标准 Markdown 表格（非矩阵），返回字典列表。"""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        table_lines = [line for line in lines if "|" in line]
        if len(table_lines) < 3:
            return []

        header_line = table_lines[0]
        headers = self._split_table_row(header_line)

        data_lines = []
        for line in table_lines[1:]:
            cells = self._split_table_row(line)
            if self._is_separator_row(cells):
                continue
            data_lines.append(line)

        rows: List[Dict[str, str]] = []
        for line in data_lines:
            cells = self._split_table_row(line)
            row_dict: Dict[str, str] = {}
            for idx, header in enumerate(headers):
                cell_value = cells[idx].strip() if idx < len(cells) else ""
                row_dict[header] = cell_value
            rows.append(row_dict)

        return rows

    def _split_table_row(self, line: str) -> List[str]:
        """将 Markdown 表格行拆分为单元格列表"""
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    def _get_cell(self, row: Dict[str, str], *candidate_keys: str) -> Optional[str]:
        """从行字典中按候选 key 名查找单元格值"""
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
        """将分号/逗号分隔的值拆分为列表"""
        if not value:
            return []
        items = re.split(r"[;；,，]+", value)
        return [item.strip() for item in items if item.strip()]

    def _parse_decision_options(self, options_text: str) -> List[Dict[str, str]]:
        """解析决策点选项文本"""
        options: List[Dict[str, str]] = []

        parts = re.split(r"[;；]+", options_text)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if_then_pattern = (
                r"(?:若|如果|if)\s*[（(]?\s*(.+?)\s*[）)]?"
                r"\s*(?:则|那么|then)\s*[（(]?\s*(.+?)\s*[）)]?\s*$"
            )
            match = re.match(if_then_pattern, part, re.IGNORECASE)
            if match:
                options.append({
                    "if": match.group(1).strip(),
                    "then": match.group(2).strip(),
                })
            else:
                colon_match = re.match(r"(.+?)\s*[:：]\s*(.+)", part)
                if colon_match:
                    options.append({
                        "if": colon_match.group(1).strip(),
                        "then": colon_match.group(2).strip(),
                    })
                elif part:
                    options.append({"if": "default", "then": part})

        return options
