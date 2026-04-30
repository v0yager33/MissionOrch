"""Mission Analyst Agent —— 任务结构化分析。

输出严格 JSON（含 mission_intent / objectives / key_entities / constraints /
uncertainties / research_queries），为 Researcher 与 Planner 提供基础。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from .base import BaseAgent

logger = logging.getLogger(__name__)


class MissionAnalystAgent(BaseAgent):
    """任务分析智能体：把自然语言任务转成结构化要素。"""

    prompt_variables = ("mission_input",)

    def __init__(self) -> None:
        super().__init__("analyst")
        self._chain: Optional[Runnable] = None
        logger.info("MissionAnalystAgent initialized")

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "请严格按 JSON 格式输出任务分析结果，不要包含任何解释文字。"),
            ],
        )

    def _get_chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self._build_prompt() | self.bind_model() | JsonOutputParser()
        return self._chain

    async def analyze(
        self,
        mission_input: str,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """分析任务，返回结构化字典。失败时返回带兜底字段的字典。"""
        merged_config = self.merge_config(config, run_name="analyst.analyze")
        chain = self._get_chain()

        logger.info(f"Analyst processing mission ({len(mission_input)} chars)")
        try:
            result = await chain.ainvoke(
                {"mission_input": mission_input}, config=merged_config
            )
            if not isinstance(result, dict):
                raise TypeError(f"Analyst expected dict, got {type(result).__name__}")
        except Exception as analyze_error:
            logger.error(f"Analyst failed: {analyze_error}", exc_info=True)
            result = self._fallback_analysis(mission_input, str(analyze_error))

        # 强制保证关键字段存在
        result.setdefault("mission_intent", mission_input[:120])
        result.setdefault("objectives", [])
        result.setdefault("key_entities", {})
        result.setdefault("constraints", [])
        result.setdefault("uncertainties", [])
        result.setdefault("research_queries", [])

        self.log_interaction(
            "analyze",
            self.system_prompt_raw,
            f"[mission ({len(mission_input)} chars)]",
            json.dumps(result, ensure_ascii=False),
        )
        logger.info(
            f"Analyst done: {len(result.get('objectives', []))} objectives, "
            f"{len(result.get('research_queries', []))} research queries"
        )
        return result

    @staticmethod
    def _fallback_analysis(mission_input: str, error: str) -> Dict[str, Any]:
        return {
            "mission_intent": mission_input[:120],
            "objectives": [
                {"id": "OBJ-1", "description": mission_input[:200], "priority": "primary"}
            ],
            "key_entities": {
                "blue_forces": [],
                "red_forces": [],
                "areas": [],
                "equipments": [],
            },
            "constraints": [],
            "uncertainties": [f"Analyst LLM 调用失败: {error}"],
            "research_queries": [mission_input[:120]],
        }
