"""
MissionOrch - 多智能体任务编排系统

通过规划-评估-反思的迭代循环生成战役级行动方案（COA）。
COA 以"作战单元 × 阶段"矩阵为核心，聚焦 Who/What/When/Where。
整个循环中 Agent 之间传递自然语言 COA 矩阵表格，
只在最终输出阶段由 COATableParser 转换为结构化格式（JSON/YAML）。
"""

__version__ = "3.0.0"

from missionorch.schemas.coa import COA, Phase, Unit, Action, Effect, DecisionPoint
from missionorch.core.orchestrator import COAOrchestrator
from missionorch.core.coa_transformer import COATransformer
from missionorch.core.coa_parser import COATableParser

__all__ = [
    "COAOrchestrator",
    "COATransformer",
    "COATableParser",
    "COA",
    "Phase",
    "Unit",
    "Action",
    "Effect",
    "DecisionPoint",
]