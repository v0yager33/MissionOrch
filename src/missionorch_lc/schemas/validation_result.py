"""Validator Agent 输出 Schema。"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Validator Agent 的结构化验证结果。"""

    is_valid: bool = Field(default=False, description="COA 是否有效")
    validation_feedback: str = Field(default="", description="验证反馈")
    issues_found: List[str] = Field(default_factory=list, description="发现的问题")
    corrected_coa: Dict[str, Any] = Field(default_factory=dict, description="修正后的 COA")
    pure_matrix_data: Dict[str, Any] = Field(
        default_factory=dict, description="提取的仿真矩阵数据"
    )
