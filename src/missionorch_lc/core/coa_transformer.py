"""COA 转换器 —— 将 COA 对象转换为 JSON / YAML / 扁平矩阵 / 压缩格式。"""

import json
from typing import Any, Dict, List

import yaml

from ..schemas.coa import COA


class COATransformer:
    """COA 格式转换工具。"""

    @staticmethod
    def coa_to_json(coa: COA) -> str:
        return json.dumps(coa.model_dump(mode="json"), indent=2, ensure_ascii=False)

    @staticmethod
    def coa_to_yaml(coa: COA) -> str:
        return yaml.dump(coa.model_dump(), default_flow_style=False, allow_unicode=True)

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
            },
            "phases": phase_summary,
            "units": [
                {"unit_id": u.unit_id, "name": u.name, "role": u.role}
                for u in coa.units
            ],
            "risk_summary": COATransformer._summarize_risks(coa),
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
