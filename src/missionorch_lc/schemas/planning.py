"""规划服务契约；完整 Markdown 保持为内容源。"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import quote
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..core.coa_checks import _ref_matches_unit, _unit_tokens, check_coa
from ..core.coa_parser import COAParseError, COATableParser

MAX_CONTENT_BYTES = 2 * 1024 * 1024


class PlanningError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    mission: str = Field(min_length=1, max_length=65536)


class PrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    markdown: str = Field(min_length=1)
    selected_task_ids: list[str] | None = None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_unique(values: list[str], label: str) -> None:
    if any(not value for value in values) or len(set(values)) != len(values):
        raise PlanningError("invalid_references", f"{label}存在空 ID 或重复 ID")


def prepare_coa(markdown: str, selected_task_ids: list[str] | None = None) -> dict[str, Any]:
    """解析完整稿；选择快照只替换矩阵，原始附加章节不丢失。"""
    if len(markdown.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise PlanningError("payload_too_large", "规划内容超过 2 MiB", 413)
    parser = COATableParser()
    sections = parser._split_sections(markdown)
    matrix_section = parser._find_section(sections, "COA矩阵", "COA Matrix", "矩阵")
    matrix_lines = [line for line in (matrix_section or "").splitlines() if "|" in line]
    if len(matrix_lines) < 3:
        raise PlanningError("unparseable_plan", "无法解析完整 COA 矩阵")
    width = len(parser._split_table_row(matrix_lines[0]))
    if width < 2 or any(len(parser._split_table_row(line)) != width for line in matrix_lines[1:]):
        raise PlanningError("unparseable_plan", "矩阵列数不一致，拒绝截断行动")
    try:
        coa = parser.parse(markdown)
    except (COAParseError, ValueError) as exc:
        raise PlanningError("unparseable_plan", "无法解析 COA Markdown") from exc
    if not coa.phases or not coa.units or not any(cell.actions for cell in coa.matrix):
        raise PlanningError("unparseable_plan", "未解析到阶段、单元及行动，请检查 COA 矩阵")
    phases = [phase.phase_id for phase in coa.phases]
    units = [unit.unit_id for unit in coa.units]
    _check_unique(phases, "阶段")
    _check_unique(units, "单元")
    phase_set, unit_set = set(phases), set(units)
    cells = [(cell.phase_id, cell.unit_id) for cell in coa.matrix]
    if len(set(cells)) != len(cells):
        raise PlanningError("invalid_references", "矩阵存在重复阶段／单元格")
    if any(phase not in phase_set or unit not in unit_set for phase, unit in cells):
        raise PlanningError("invalid_references", "矩阵引用了不存在的阶段或单元")
    for row in [*coa.decision_points, *coa.expected_outcomes]:
        if row.phase_id and row.phase_id not in phase_set:
            raise PlanningError("invalid_references", f"阶段引用不存在：{row.phase_id}")
    _check_unique([effect.effect_id for effect in coa.effects_chain], "效果")
    _check_unique([row.dp_id for row in coa.decision_points], "决策点")
    _check_unique([row.outcome_id for row in coa.expected_outcomes], "战果")
    effects = {effect.effect_id for effect in coa.effects_chain}
    for row in coa.expected_outcomes:
        if any(effect not in effects for effect in row.linked_effects):
            raise PlanningError("invalid_references", "预期战果引用了不存在的效果")
    tokens = _unit_tokens(coa)
    if any(not _ref_matches_unit(ref, tokens) for effect in coa.effects_chain for ref in effect.achieved_by):
        raise PlanningError("invalid_references", "效果链引用了不存在的执行单元")
    manifest = parser._find_section(sections, "实体清单", "Entity Manifest")
    if manifest:
        rows = parser._parse_markdown_table(manifest)
        if rows:
            entity_ids = [parser._get_cell(row, "unit_id", "实体ID", "ID") or "" for row in rows]
            _check_unique(entity_ids, "实体清单")
            normalized = {value.replace("/", "_").replace("-", "_") for value in entity_ids}
            if not unit_set.issubset(normalized):
                raise PlanningError("invalid_references", "矩阵单元未列入实体清单")
    geo = parser._find_section(sections, "地理参考", "Geo Reference")
    geo_rows = parser._parse_markdown_table(geo or "")
    geo_ids = [parser._get_cell(row, "ref_id", "参考ID", "ID") or "" for row in geo_rows]
    _check_unique(geo_ids, "地理参考")
    explicit_geo_refs = set(re.findall(r"<(?:BASE|WAYPOINT|TARGET|AREA|ORBIT|POSITION):[^<>\n]+>", markdown))
    if explicit_geo_refs - {value.strip("`") for value in geo_ids}:
        raise PlanningError("invalid_references", "规划包含未定义的地理占位引用")
    names = {unit.unit_id: unit.name for unit in coa.units}
    phase_names = {phase.phase_id: phase.name for phase in coa.phases}
    tasks = []
    for cell in coa.matrix:
        for index, action in enumerate(cell.actions):
            if not action.strip():
                raise PlanningError("invalid_action", "行动正文不能为空")
            tasks.append({
                "id": f"{quote(cell.phase_id, safe='')}:{quote(cell.unit_id, safe='')}:{index}",
                "phase_id": cell.phase_id, "phase_name": phase_names[cell.phase_id],
                "unit_id": cell.unit_id, "unit_name": names[cell.unit_id],
                "action_index": index, "text": action,
            })
    ids = {task["id"] for task in tasks}
    chosen = ids if selected_task_ids is None else set(selected_task_ids)
    if not chosen or not chosen.issubset(ids) or (
        selected_task_ids is not None and len(chosen) != len(selected_task_ids)
    ):
        raise PlanningError("invalid_selection", "请选择有效且不重复的行动")
    snapshot = markdown
    selection_note = "文本依赖未自动扩展；如需要未选前置行动，请在想定会话中确认。"
    if selected_task_ids is not None:
        selected_phases = {task["phase_id"] for task in tasks if task["id"] in chosen}
        reference_text = "\n".join(
            [task["text"] for task in tasks if task["id"] in chosen]
            + [phase.transition_trigger for phase in coa.phases if phase.phase_id in selected_phases]
        )
        # 只报告可核对的显式 ID 引用，不推断或补全自然语言依赖图。
        phase_refs = set(re.findall(r"\bPhase_\d+\b", reference_text, re.ASCII)) - selected_phases
        action_refs = set(re.findall(
            r"\bPhase_\d+:[A-Za-z0-9_%~.-]+:\d+\b", reference_text, re.ASCII,
        )) & (ids - chosen)
        warnings = [f"阶段 {phase.phase_id}（{phase.name}）" for phase in coa.phases if phase.phase_id in phase_refs]
        warnings += [f'行动 {task["id"]}' for task in tasks if task["id"] in action_refs]
        if warnings:
            selection_note = "选中文本显式提及未选范围：" + "、".join(warnings[:20]) + "。" + selection_note
        heading = re.search(
            r"^##[^\S\n]+[^\n]*(?:COA矩阵|COA Matrix|矩阵)[^\n]*\n",
            markdown, re.MULTILINE | re.IGNORECASE,
        )
        if not heading:
            raise PlanningError("unparseable_plan", "找不到 COA 矩阵章节")
        next_heading = re.search(r"^##\s+", markdown[heading.end():], re.MULTILINE)
        end = heading.end() + next_heading.start() if next_heading else len(markdown)
        # 保留原表头的触发条件、目标和实体列，只替换行动单元格。
        section_lines = markdown[heading.end():end].splitlines(keepends=True)
        table_indices = [i for i, line in enumerate(section_lines) if "|" in line]
        parser = COATableParser()
        data_indices = [i for i in table_indices[1:]
                        if not parser._is_separator_row(parser._split_table_row(section_lines[i]))]
        if len(data_indices) != len(coa.units):
            raise PlanningError("unparseable_plan", "矩阵行数不一致")
        for line_index, unit in zip(data_indices, coa.units):
            original_cells = parser._split_table_row(section_lines[line_index])
            cells = [original_cells[0]]
            for phase in coa.phases:
                actions = [task["text"] for task in tasks if task["id"] in chosen
                           and task["phase_id"] == phase.phase_id and task["unit_id"] == unit.unit_id]
                cells.append("<br>".join(f"{i + 1}. {text}" for i, text in enumerate(actions)) or "-")
            section_lines[line_index] = "| " + " | ".join(cells) + " |\n"
        snapshot = markdown[:heading.end()] + "".join(section_lines) + markdown[end:]
        participants = sorted({task["unit_id"] for task in tasks if task["id"] in chosen})
        scope_note = (
            "\n> 执行范围：仅执行本矩阵中非空的已选行动；空单元格不执行。\n"
            f"> 执行单元：{', '.join(participants)}。其他实体仅为背景／目标，不增加其行动。\n"
            "> 保留的阶段、实体、地理、效果和其他章节是上下文；不得据此恢复未选择行动。\n"
            f"> 依赖提示：{selection_note}\n"
        )
        first_heading = re.search(r"^##\s+", snapshot, re.MULTILINE)
        pos = first_heading.start() if first_heading else len(snapshot)
        snapshot = snapshot[:pos] + scope_note + "\n" + snapshot[pos:]
    return {
        "markdown": markdown,
        "content_hash": content_hash(markdown),
        "coa": coa.model_dump(mode="json"),
        "tasks": tasks,
        "validation": check_coa(coa, markdown).as_dict(),
        "snapshot_markdown": snapshot,
        "snapshot_hash": content_hash(snapshot),
        "selected_task_ids": [task["id"] for task in tasks if task["id"] in chosen],
        "selection_note": selection_note,
    }
