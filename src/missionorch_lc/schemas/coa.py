"""COA 数据模型 —— 作战单元(Unit) × 阶段(Phase) 的二维矩阵。"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Phase(BaseModel):
    """作战阶段。"""

    phase_id: str = Field(description="阶段标识，如 Phase_1")
    name: str = Field(description="阶段名称")
    transition_trigger: str = Field(
        default="",
        description="阶段转换触发条件（条件/事件驱动，禁止固定时间）",
    )
    objective: str = Field(default="", description="阶段目标")


class Unit(BaseModel):
    """作战单元。"""

    unit_id: str = Field(description="单元标识")
    name: str = Field(description="单元名称")
    role: str = Field(default="", description="职能角色")


class Action(BaseModel):
    """矩阵单元格：某作战单元在某阶段的行动列表。"""

    unit_id: str
    phase_id: str
    actions: List[str] = Field(default_factory=list)


class Effect(BaseModel):
    """战略效果。"""

    effect_id: str = Field(default="effect_default")
    description: str = Field(default="")
    measures: List[str] = Field(default_factory=list)
    achieved_by: List[str] = Field(default_factory=list)


class DecisionPoint(BaseModel):
    """决策点。"""

    dp_id: str = Field(default="dp_default")
    phase_id: str = Field(default="")
    condition: str = Field(default="")
    options: List[Dict[str, str]] = Field(default_factory=list)


class ExpectedOutcome(BaseModel):
    """预期战果（Expected Outcome / MOE）。

    只记录、透传，不参与评分与流程决策。预留给后续"想定脚本 → 仿真 → 实测比对 → 战报"：
    仿真结束后回填 ``actual_value`` 与 ``status``，即可逐条比对方案的预期与实际效果。
    """

    outcome_id: str = Field(default="EO-1", description="战果标识，如 EO-1")
    description: str = Field(default="", description="预期战果描述")
    metric: str = Field(default="", description="衡量指标，如 击落敌机数量")
    comparator: str = Field(
        default="", description="比较符：>= <= == > < ≈ range；空表示定性目标"
    )
    target_value: str = Field(default="", description="目标值，如 6 或 6-8")
    unit: str = Field(default="", description="单位，如 架 / % / 分钟")
    side: str = Field(default="", description="统计对象所属方：BLUE / RED / NEUTRAL")
    phase_id: str = Field(default="", description="关联阶段")
    priority: str = Field(default="", description="优先级：高/中/低")
    linked_effects: List[str] = Field(default_factory=list, description="关联效果ID")
    observable: str = Field(default="", description="仿真中可统计的观测量")
    source: str = Field(
        default="", description="来源：user / mission_input / analyst / planner"
    )

    # ── 预留：仿真回填，当前流程不写入 ──
    actual_value: Optional[str] = Field(default=None, description="仿真实测值（预留）")
    status: str = Field(
        default="PENDING", description="比对结论（预留）：PENDING / MET / PARTIAL / NOT_MET"
    )


class COA(BaseModel):
    """Course of Action —— 战役级行动方案。"""

    coa_id: str = Field(
        default_factory=lambda: f"COA-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    name: str = Field(default="Generated COA")
    description: str = Field(default="")

    phases: List[Phase] = Field(default_factory=list)
    units: List[Unit] = Field(default_factory=list)
    matrix: List[Action] = Field(default_factory=list)

    effects_chain: List[Effect] = Field(default_factory=list)
    decision_points: List[DecisionPoint] = Field(default_factory=list)
    critical_risks: List[Dict[str, str]] = Field(default_factory=list)
    expected_outcomes: List[ExpectedOutcome] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ── 辅助查询 ──
    def get_actions(self, unit_id: str, phase_id: str) -> List[str]:
        for action in self.matrix:
            if action.unit_id == unit_id and action.phase_id == phase_id:
                return action.actions
        return []

    def get_unit_actions(self, unit_id: str) -> Dict[str, List[str]]:
        return {a.phase_id: a.actions for a in self.matrix if a.unit_id == unit_id}

    def get_phase_actions(self, phase_id: str) -> Dict[str, List[str]]:
        return {a.unit_id: a.actions for a in self.matrix if a.phase_id == phase_id}
