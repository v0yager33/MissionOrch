"""Validator Agent —— 验证 COA 矩阵格式并提取仿真矩阵数据。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from ..schemas.coa import COA
from ..schemas.validation_result import ValidationResult
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseAgent):
    """验证智能体。"""

    prompt_variables = ("coa_json", "mission", "validation_rules")

    def __init__(self) -> None:
        super().__init__("validator")
        self.validation_rules: List[str] = self.get_config_value(
            "validation_rules",
            ["matrix_completeness", "phase_progression", "unit_coordination", "effect_coverage"],
        )

        prompt = self._build_prompt()
        self._structured_chain, self._fallback_chain = self.build_structured_chain_pair(
            ValidationResult, prompt
        )
        logger.info(f"ValidatorAgent initialized with rules: {self.validation_rules}")

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "请严格按 JSON 格式输出验证结果。"),
            ],
        )

    async def validate_and_extract_matrix(
        self,
        coa: COA,
        mission_desc: str = "",
        *,
        config: Optional[RunnableConfig] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """返回 `(is_valid, feedback, extras_dict)`。"""
        rules_translated = "\n".join(
            f"- {rule}" for rule in self._translate_rules(self.validation_rules)
        )
        coa_json_str = json.dumps(coa.model_dump(mode="json"), indent=2, ensure_ascii=False)
        inputs = {
            "coa_json": coa_json_str,
            "mission": mission_desc,
            "validation_rules": rules_translated,
        }
        merged_config = self.merge_config(config, run_name="validator.validate")

        try:
            result_dict = await self.invoke_structured(
                ValidationResult,
                self._structured_chain,
                self._fallback_chain,
                inputs,
                config=merged_config,
            )
        except Exception as validate_error:
            logger.error(
                f"Validator failed, falling back to basic validation: {validate_error}"
            )
            return (
                self._basic_validate(coa),
                f"Validator failed: {validate_error}",
                {
                    "corrected_coa_data": coa.model_dump(),
                    "pure_matrix_data": self._extract_basic_matrix(coa),
                    "issues_found": [str(validate_error)],
                    "validation_feedback": "回退到基本验证与矩阵提取",
                },
            )

        is_valid = bool(result_dict.get("is_valid", False))
        feedback = str(result_dict.get("validation_feedback", ""))
        corrected_coa_data = result_dict.get("corrected_coa") or coa.model_dump()
        pure_matrix_data = result_dict.get("pure_matrix_data") or self._extract_basic_matrix(coa)

        if not is_valid:
            logger.warning("COA validation failed, attempting automatic fixes")
            corrected_coa = self._attempt_fix_coa(coa, result_dict.get("issues_found", []))
            corrected_coa_data = corrected_coa.model_dump()

        self.log_interaction(
            "validate_and_extract_matrix",
            self.system_prompt_raw,
            f"[coa_json ({len(coa_json_str)} chars)]",
            str(result_dict),
        )

        return is_valid, feedback, {
            "corrected_coa_data": corrected_coa_data,
            "pure_matrix_data": pure_matrix_data,
            "issues_found": result_dict.get("issues_found", []),
            "validation_feedback": feedback,
        }

    # ── 工具方法 ──
    _RULE_TRANSLATIONS = {
        "matrix_completeness": "矩阵完整性：每个单元在每个阶段都应有明确行动",
        "phase_progression": "阶段递进：转换条件应合理且由事件/条件驱动",
        "unit_coordination": "单元协同：各单元在同一阶段的行动应协调一致",
        "effect_coverage": "效果覆盖：行动应能有效达成预期战略效果",
        "format_compliance": "格式合规：矩阵格式符合系统解析要求",
        "logical_consistency": "逻辑一致：行动之间无矛盾冲突",
        "resource_feasibility": "资源可行：资源分配不冲突",
        "decision_point_clarity": "决策点明确：决策点条件和选项清晰",
    }

    def _translate_rules(self, rules: List[str]) -> List[str]:
        translated = [self._RULE_TRANSLATIONS.get(rule, rule) for rule in rules]
        return translated if translated else list(self._RULE_TRANSLATIONS.values())[:4]

    def _attempt_fix_coa(self, coa: COA, issues: List[str]) -> COA:
        logger.debug(f"Attempting to fix COA issues: {issues}")
        coa_dict = coa.model_dump()

        if not coa_dict.get("description"):
            coa_dict["description"] = "Generated COA"

        seen_ids: set = set()
        for idx, unit in enumerate(coa_dict.get("units", [])):
            unit_id = unit.get("unit_id", f"Unit_{idx + 1}")
            original = unit_id
            counter = 1
            while unit_id in seen_ids:
                unit_id = f"{original}_{counter}"
                counter += 1
            seen_ids.add(unit_id)
            coa_dict["units"][idx]["unit_id"] = unit_id

        existing_cells = {
            (cell["unit_id"], cell["phase_id"]) for cell in coa_dict.get("matrix", [])
        }
        for unit in coa_dict.get("units", []):
            for phase in coa_dict.get("phases", []):
                key = (unit["unit_id"], phase["phase_id"])
                if key not in existing_cells:
                    coa_dict["matrix"].append({
                        "unit_id": unit["unit_id"],
                        "phase_id": phase["phase_id"],
                        "actions": [],
                    })

        return COA.model_validate(coa_dict)

    def _basic_validate(self, coa: COA) -> bool:
        return all([
            len(coa.phases) >= 2,
            len(coa.units) >= 2,
            len(coa.matrix) > 0,
            len(coa.effects_chain) > 0,
            bool(coa.description),
        ])

    def _extract_basic_matrix(self, coa: COA) -> Dict[str, Any]:
        flat: Dict[str, Dict[str, List[str]]] = {}
        for unit in coa.units:
            flat[unit.unit_id] = {
                phase.phase_id: coa.get_actions(unit.unit_id, phase.phase_id)
                for phase in coa.phases
            }

        return {
            "units": [u.unit_id for u in coa.units],
            "phases": [p.phase_id for p in coa.phases],
            "matrix": flat,
            "phase_transitions": {
                p.phase_id: p.transition_trigger for p in coa.phases if p.transition_trigger
            },
            "effects": [
                {
                    "effect_id": e.effect_id,
                    "description": e.description,
                    "measures": e.measures,
                    "achieved_by": e.achieved_by,
                }
                for e in coa.effects_chain
            ],
            "unit_details": {
                u.unit_id: {"name": u.name, "role": u.role} for u in coa.units
            },
            "decision_points": [
                {
                    "dp_id": dp.dp_id,
                    "phase_id": dp.phase_id,
                    "condition": dp.condition,
                    "options": dp.options,
                }
                for dp in coa.decision_points
            ],
            "risks": coa.critical_risks,
        }
