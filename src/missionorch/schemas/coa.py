"""
COA Schema - 战役级行动方案数据模型

核心结构：作战单元(Unit) × 阶段(Phase) 的二维矩阵
每个单元格包含该作战单元在该阶段的行动列表(Actions)

设计原则：
- 聚焦 Who/What/When/Where 四要素
- 战役级抽象，不涉及战术细节
- 后续交付给其他系统 agent 转化为想定
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class Phase(BaseModel):
    """作战阶段"""
    phase_id: str = Field(description="阶段标识，如 Phase_1")
    name: str = Field(description="阶段名称，如 '突防与压制'")
    transition_trigger: str = Field(
        default="",
        description="阶段转换触发条件，如 '敌防空网络被压制后→转入下一阶段'。"
                    "战役级阶段转换由条件/事件驱动，而非固定时间。"
    )
    objective: str = Field(default="", description="阶段目标，如 '突破防线，致盲敌雷达'")


class Unit(BaseModel):
    """作战单元"""
    unit_id: str = Field(description="单元标识，如 EA-18G_EW")
    name: str = Field(description="单元名称，如 '电子战飞机 (EA-18G)'")
    role: str = Field(default="", description="职能角色，如 '电磁掩护'")


class Action(BaseModel):
    """单元格行动 - 某个作战单元在某个阶段的具体行动"""
    unit_id: str = Field(description="所属作战单元ID")
    phase_id: str = Field(description="所属阶段ID")
    actions: List[str] = Field(default_factory=list, description="行动描述列表")


class Effect(BaseModel):
    """战略效果"""
    effect_id: str = Field(default="effect_default", description="效果ID")
    description: str = Field(default="", description="效果描述")
    measures: List[str] = Field(default_factory=list, description="衡量指标")
    achieved_by: List[str] = Field(default_factory=list, description="达成该效果的单元ID列表")


class DecisionPoint(BaseModel):
    """决策点"""
    dp_id: str = Field(default="dp_default", description="决策点ID")
    phase_id: str = Field(default="", description="所属阶段")
    condition: str = Field(default="", description="触发条件")
    options: List[Dict[str, str]] = Field(default_factory=list, description="选项：if/then")


class COA(BaseModel):
    """
    行动方案（Course of Action）

    核心是"作战单元 × 阶段"的二维矩阵，
    每个单元格描述该单元在该阶段的具体行动。
    """
    coa_id: str = Field(default_factory=lambda: f"COA-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    name: str = Field(default="Generated COA", description="方案名称")
    description: str = Field(default="", description="方案概述")

    # ── 矩阵核心三要素 ──
    phases: List[Phase] = Field(default_factory=list, description="阶段列表（横轴）")
    units: List[Unit] = Field(default_factory=list, description="作战单元列表（纵轴）")
    matrix: List[Action] = Field(default_factory=list, description="矩阵单元格：每个单元在每个阶段的行动")

    # ── 辅助要素 ──
    effects_chain: List[Effect] = Field(default_factory=list, description="效果链")
    decision_points: List[DecisionPoint] = Field(default_factory=list, description="决策点")
    critical_risks: List[Dict[str, str]] = Field(default_factory=list, description="关键风险")

    # ── 元数据 ──
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def get_actions(self, unit_id: str, phase_id: str) -> List[str]:
        """获取指定单元在指定阶段的行动列表"""
        for action in self.matrix:
            if action.unit_id == unit_id and action.phase_id == phase_id:
                return action.actions
        return []

    def get_unit_actions(self, unit_id: str) -> Dict[str, List[str]]:
        """获取指定单元在所有阶段的行动"""
        result: Dict[str, List[str]] = {}
        for action in self.matrix:
            if action.unit_id == unit_id:
                result[action.phase_id] = action.actions
        return result

    def get_phase_actions(self, phase_id: str) -> Dict[str, List[str]]:
        """获取指定阶段所有单元的行动"""
        result: Dict[str, List[str]] = {}
        for action in self.matrix:
            if action.phase_id == phase_id:
                result[action.unit_id] = action.actions
        return result