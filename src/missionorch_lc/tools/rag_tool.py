"""RAG 检索工具（LangChain Tool）。"""

from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.rag_manager import RAGManager


class RAGSearchInput(BaseModel):
    query: str = Field(..., description="查询关键词或问题")


class RAGSearchTool(BaseTool):
    """检索军事战术、武器装备性能、历史战例等相关知识。"""

    name: str = "rag_search"
    description: str = (
        "检索军事战术、武器装备性能、历史战例等相关知识。"
        "当规划/反思过程中不确定某些军事常识时，请调用此工具获取参考信息。"
    )
    args_schema: Type[BaseModel] = RAGSearchInput

    rag_manager: RAGManager = Field(default_factory=RAGManager)

    # 允许 RAGManager 这种非 pydantic 字段
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str) -> str:  # pragma: no cover - LangChain 同步入口兜底
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self._arun(query))

    async def _arun(self, query: str) -> str:
        context = await self.rag_manager.retrieve(query)
        if not context:
            return "未找到相关信息。"
        return f"检索结果:\n{context}"
