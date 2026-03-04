"""
COA 转换模块 - 将 COA 矩阵转换为不同格式

适配新的"作战单元 x 阶段"矩阵架构。
"""

from typing import Dict, Any, List
from ..schemas.coa import COA
import json
import yaml


class COATransformer:
    """COA 转换器"""

    @staticmethod
    def coa_to_json(coa: COA) -> str:
        """将 COA 转换为 JSON 格式"""
        return json.dumps(coa.model_dump(mode="json"), indent=2, ensure_ascii=False)

    @staticmethod
    def coa_to_yaml(coa: COA) -> str:
        """将 COA 转换为 YAML 格式"""
        coa_dict = coa.model_dump()
        return yaml.dump(coa_dict, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def coa_to_condensed_format(coa: COA) -> Dict[str, Any]:
        """将 COA 转换为压缩格式，便于传输和概览"""
        phase_summary = []
        for phase in coa.phases:
            phase_info = {
                "phase_id": phase.phase_id,
                "name": phase.name,
                "transition_trigger": phase.transition_trigger,
                "objective": phase.objective,
                "unit_actions": {},
            }
            for unit in coa.units:
                actions = coa.get_actions(unit.unit_id, phase.phase_id)
                if actions:
                    phase_info["unit_actions"][unit.unit_id] = actions
            phase_summary.append(phase_info)

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
    def coa_to_flat_matrix(coa: COA) -> Dict[str, Any]:
        """
        将 COA 转换为扁平矩阵格式，便于下游系统消费。

        返回格式：
        {
            "phases": [...],
            "units": [...],
            "matrix": {
                "unit_id": {
                    "phase_id": ["action1", "action2", ...]
                }
            }
        }
        """
        flat_matrix: Dict[str, Dict[str, List[str]]] = {}
        for unit in coa.units:
            flat_matrix[unit.unit_id] = {}
            for phase in coa.phases:
                flat_matrix[unit.unit_id][phase.phase_id] = coa.get_actions(
                    unit.unit_id, phase.phase_id
                )

        return {
            "phases": [p.model_dump() for p in coa.phases],
            "units": [u.model_dump() for u in coa.units],
            "matrix": flat_matrix,
            "effects": [e.model_dump() for e in coa.effects_chain],
            "decision_points": [dp.model_dump() for dp in coa.decision_points],
            "risks": coa.critical_risks,
        }

    @staticmethod
    def _summarize_risks(coa: COA) -> Dict[str, Any]:
        """汇总风险信息"""
        categories: Dict[str, int] = {}
        for risk in coa.critical_risks:
            category = risk.get("category", "GENERAL")
            categories[category] = categories.get(category, 0) + 1
        return {
            "total_risks": len(coa.critical_risks),
            "risk_categories": categories,
        }

    @staticmethod
    def coa_from_dict(coa_dict: Dict[str, Any]) -> COA:
        """从字典创建 COA 对象"""
        return COA.model_validate(coa_dict)