"""Judge Agent 输出 Schema —— 用于 LangChain 的 PydanticOutputParser。"""

from typing import List

from pydantic import BaseModel, Field


class DimensionScores(BaseModel):
    """各维度评分。"""

    feasibility: float = Field(default=0.0, description="可行性")
    completeness: float = Field(default=0.0, description="完整性")
    synchronization: float = Field(default=0.0, description="协同性")
    phase_progression: float = Field(default=0.0, description="阶段递进")
    effect_achievement: float = Field(default=0.0, description="效果达成")


class JudgeResult(BaseModel):
    """Judge Agent 的结构化评估结果。"""

    overall_score: float = Field(description="总分（0-10）")
    dimension_scores: DimensionScores = Field(
        default_factory=DimensionScores, description="各维度得分"
    )
    verdict: str = Field(default="REVISE", description="判定：ACCEPT / REVISE / REJECT")
    feedback: str = Field(default="", description="详细反馈")
    critical_issues: List[str] = Field(default_factory=list, description="关键问题")
    improvement_suggestions: List[str] = Field(default_factory=list, description="改进建议")
