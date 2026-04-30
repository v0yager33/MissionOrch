"""LangChain Tool 注册表。"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """LangChain Tool 管理中心。

    - 注册 / 查询工具
    - 为 Agent 提供 `bind_tools` 所需的工具列表
    - 提供 Function Calling Schema 导出
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_definitions(self) -> List[Dict[str, Any]]:
        """返回 OpenAI Function Calling 格式的 schema 列表。"""
        schemas: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            args_schema = tool.args_schema
            params: Dict[str, Any] = {"type": "object", "properties": {}}
            if args_schema is not None:
                try:
                    params = args_schema.model_json_schema()
                    params.pop("title", None)
                except Exception:
                    pass
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": params,
            })
        return schemas

    def clear(self) -> None:
        self._tools.clear()
