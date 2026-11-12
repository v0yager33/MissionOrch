"""Mission Analyst Agent —— 任务结构化分析。

输出严格 JSON（含 mission_intent / objectives / key_entities / constraints /
uncertainties / research_queries / expected_outcomes），为 Researcher 与 Planner 提供基础。

预期战果先用规则从任务文本的"## 预期战果"等段落里确定性抽取，再合并 LLM 抽到的条目；
LLM 调用失败时规则抽取的结果仍然保留。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from ..core.expected_outcomes import coerce_outcomes, extract_from_mission, merge_outcomes
from .base import BaseAgent

logger = logging.getLogger(__name__)


def _text(value: Any) -> str:
    """列表条目 → 文本；dict 取 name / description。"""
    if isinstance(value, dict):
        value = value.get("name") or value.get("description") or json.dumps(value, ensure_ascii=False)
    return str(value).strip() if value is not None else ""


class MissionAnalystAgent(BaseAgent):
    """任务分析智能体：把自然语言任务转成结构化要素。"""

    prompt_variables = ("mission_input",)

    def __init__(self, profile: Optional[str] = None) -> None:
        super().__init__("analyst", profile=profile)
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

        # 强制保证关键字段存在且类型正确（LLM 可能给 null / 字符串列表）
        self._normalize(result, mission_input)
        # 规则抽取优先（数值精确），LLM 只补充规则没抽到的条目
        result["expected_outcomes"] = merge_outcomes(
            extract_from_mission(mission_input),
            coerce_outcomes(result.get("expected_outcomes"), source="analyst"),
            keep_primary_source=True,
        )

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
    def _normalize(result: Dict[str, Any], mission_input: str) -> None:
        """就地修正字段类型，下游（transformer / memory / webui）可以直接 .get()。"""
        if not result.get("mission_intent"):
            result["mission_intent"] = mission_input[:120]
        for key in ("constraints", "uncertainties", "research_queries"):
            value = result.get(key)
            if isinstance(value, str):
                value = [value]
            result[key] = [_text(v) for v in (value or []) if _text(v)]

        objectives = []
        raw_objectives = result.get("objectives") or []
        if isinstance(raw_objectives, (str, dict)):
            raw_objectives = [raw_objectives]
        for idx, obj in enumerate(raw_objectives, start=1):
            if isinstance(obj, dict):
                obj.setdefault("id", f"OBJ-{idx}")
                obj.setdefault("priority", "")
                obj.setdefault("description", "")
                objectives.append(obj)
            elif str(obj).strip():
                objectives.append({"id": f"OBJ-{idx}", "priority": "", "description": str(obj)})
        result["objectives"] = objectives

        entities = result.get("key_entities")
        if not isinstance(entities, dict):
            entities = {}
        for key, value in list(entities.items()):
            if isinstance(value, str):
                entities[key] = [value]
            elif isinstance(value, list):
                entities[key] = [_text(v) for v in value if _text(v)]
            else:
                entities[key] = []
        result["key_entities"] = entities

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
