"""COA 语法校验工具（LangChain Tool）。"""

from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.coa_parser import COAParseError, COATableParser


class COASyntaxInput(BaseModel):
    markdown_text: str = Field(..., description="需要检查的 COA Markdown 文本")


class COASyntaxValidatorTool(BaseTool):
    """检查 COA Markdown 表格的格式是否合法。"""

    name: str = "coa_syntax_validator"
    description: str = (
        "检查 COA Markdown 表格的格式是否合法、能否被系统解析。"
        "如果格式错误，会返回具体的错误信息；格式正确时返回统计信息和潜在警告。"
    )
    args_schema: Type[BaseModel] = COASyntaxInput

    parser: COATableParser = Field(default_factory=COATableParser)

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, markdown_text: str) -> Dict[str, Any]:
        return self._check(markdown_text)

    async def _arun(self, markdown_text: str) -> Dict[str, Any]:
        return self._check(markdown_text)

    def _check(self, markdown_text: str) -> Dict[str, Any]:
        try:
            coa = self.parser.parse(markdown_text)
        except COAParseError as parse_error:
            return {
                "valid": False,
                "message": f"格式解析失败: {parse_error}",
                "error_type": "COAParseError",
            }
        except Exception as unexpected_error:
            return {
                "valid": False,
                "message": f"未知错误: {unexpected_error}",
                "error_type": type(unexpected_error).__name__,
            }

        stats = {
            "phases": len(coa.phases),
            "units": len(coa.units),
            "matrix_cells": len(coa.matrix),
            "effects": len(coa.effects_chain),
        }
        warnings = []
        if stats["phases"] < 3:
            warnings.append("阶段数量少于 3 个，建议增加")
        if stats["units"] < 3:
            warnings.append("作战单元数量少于 3 个，建议增加")
        empty_cells = sum(1 for action in coa.matrix if not action.actions)
        if empty_cells > 0:
            warnings.append(f"有 {empty_cells} 个矩阵单元格为空")

        return {
            "valid": True,
            "message": "格式解析成功",
            "stats": stats,
            "warnings": warnings,
        }
