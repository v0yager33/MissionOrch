"""RAG 工具集 —— 按知识源拆分为多个 LangChain Tool。

为什么按 source 拆？
- 单一 `rag_search` 工具会让 Agent 失去"该查哪里"的判断能力
- 拆开后，Agent 在 ReAct 推理时可以根据问题类型主动选择：
    - 想看作战条例 → `rag_doctrine_search`
    - 想看战例对比 → `rag_historical_search`
    - 想看地形/地图 → `rag_map_search`
    - 不确定术语    → `rag_glossary_search`
- 每个工具的 description 详尽，便于 LLM 通过 function calling 自动选择

返回结构统一为：检索文本 + 来源标注，便于 Planner 引用追溯。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.rag_manager import RAGManager

logger = logging.getLogger(__name__)


# ── 共享输入 schema ──


class _RAGQueryInput(BaseModel):
    query: str = Field(
        ...,
        description="自然语言查询，应该具体而非笼统（如『装甲集群在城市作战中的协同要点』）",
    )
    mode: Optional[str] = Field(
        default=None,
        description="检索模式：hybrid（默认）/local（实体邻居）/global（关系图）/naive（朴素向量）",
    )


# ── 工具基类 ──


class _SourceRAGTool(BaseTool):
    """绑定到固定 source 的 RAG 工具。子类只需指定 name / description / source_name。"""

    args_schema: Type[BaseModel] = _RAGQueryInput
    rag_manager: RAGManager = Field(default_factory=RAGManager)
    source_name: str = ""
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str, mode: Optional[str] = None) -> str:  # pragma: no cover
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self._arun(query, mode=mode))

    async def _arun(self, query: str, mode: Optional[str] = None) -> str:
        if not self.rag_manager.is_enabled():
            return f"[RAG 未启用] 工具 {self.name} 无法返回真实结果。"
        if self.source_name not in self.rag_manager.list_sources():
            return (
                f"[知识源 '{self.source_name}' 未配置] 当前可用："
                f"{', '.join(self.rag_manager.list_sources()) or '无'}"
            )
        text = await self.rag_manager.retrieve(query, source=self.source_name, mode=mode)
        if not text:
            return f"[未在 {self.source_name} 中找到相关信息] query={query!r}"
        return f"[来源: {self.source_name}]\n{text}"


# ── 具体工具 ──


class DoctrineSearchTool(_SourceRAGTool):
    """检索作战条令、战术手册、训练大纲等规范性文献。"""

    name: str = "rag_doctrine_search"
    description: str = (
        "在『作战条令与战术手册』知识库中检索。"
        "适用问题示例：装甲集群机动原则、特种作战指挥关系、空地协同程序等。"
        "当需要权威规范、概念定义或标准流程时优先调用。"
    )
    source_name: str = "doctrines"


class MapSearchTool(_SourceRAGTool):
    """检索作战地图、地形态势、关键地物等空间信息。"""

    name: str = "rag_map_search"
    description: str = (
        "在『作战地图与地形』知识库中检索（含图片解析结果）。"
        "适用问题示例：某区域地形特征、关键节点位置、敌我态势分布等。"
        "当规划需要空间/地形参考时调用。"
    )
    source_name: str = "maps"


class HistoricalSearchTool(_SourceRAGTool):
    """检索历史战例、战后报告、经验教训。"""

    name: str = "rag_historical_search"
    description: str = (
        "在『历史战例与战后报告』知识库中检索。"
        "适用问题示例：类似任务的历史经验、过往失败原因、成功要素提炼等。"
        "当需要类比、教训提炼或风险预警时调用。"
    )
    source_name: str = "historical"


class GlossarySearchTool(_SourceRAGTool):
    """检索专业术语、缩略语、装备代号。"""

    name: str = "rag_glossary_search"
    description: str = (
        "在『术语与代号』知识库中检索精确定义。"
        "适用问题示例：某缩略语含义、装备型号参数、专业概念边界等。"
        "当遇到不确定的术语时调用，确保理解一致。"
    )
    source_name: str = "glossary"


# ── 一键工厂 ──


def build_rag_tools(
    rag_manager: Optional[RAGManager] = None,
    enabled_sources: Optional[List[str]] = None,
) -> List[BaseTool]:
    """根据已配置的 source 自动构建对应工具列表。

    Args:
        rag_manager: 复用已有的 RAGManager；None 则按默认配置实例化
        enabled_sources: 白名单；None 则根据 RAGManager 已构建的 source 自动选择

    Returns:
        BaseTool 列表，可直接 ``bind_tools`` 给 LLM
    """
    manager = rag_manager or RAGManager()
    if not manager.is_enabled():
        logger.warning("RAGManager 未启用，build_rag_tools 返回空列表")
        return []

    available = set(manager.list_sources())
    candidate_classes: dict[str, Type[_SourceRAGTool]] = {
        "doctrines": DoctrineSearchTool,
        "maps": MapSearchTool,
        "historical": HistoricalSearchTool,
        "glossary": GlossarySearchTool,
    }
    if enabled_sources is None:
        targets = [s for s in candidate_classes if s in available]
    else:
        targets = [s for s in enabled_sources if s in candidate_classes and s in available]

    tools: List[BaseTool] = []
    for source in targets:
        tool_cls = candidate_classes[source]
        tools.append(tool_cls(rag_manager=manager))
    logger.info(f"build_rag_tools: {len(tools)} 个工具就绪 ({[t.name for t in tools]})")
    return tools
