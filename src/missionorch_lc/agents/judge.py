"""Judge Agent —— 评估 COA 质量，输出结构化 JudgeResult。

优先使用 `model.with_structured_output(JudgeResult)`（底层走 function calling / JSON mode，
可靠性远高于 JsonOutputParser）；若模型不支持，自动回退到 JsonOutputParser 兜底链。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from ..schemas.judge_result import JudgeResult
from .base import BaseAgent

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """评估智能体 —— 直接评估自然语言 COA 表格。"""

    # judge.txt 的占位符白名单
    prompt_variables = ("coa_table", "mission", "criteria")

    def __init__(self) -> None:
        super().__init__("judge")
        self.criteria = self.agent_cfg.get("criteria", [])
        self._structured_chain: Optional[Runnable] = None
        self._fallback_chain: Optional[Runnable] = None
        self._structured_supported: Optional[bool] = None  # None=未尝试, True=支持, False=不支持
        logger.info(f"JudgeAgent initialized with criteria: {self.criteria}")

    # ── Chain 构造（懒加载） ──
    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "请按要求输出评估结果 JSON。"),
            ],
        )

    def _get_structured_chain(self) -> Optional[Runnable]:
        if self._structured_supported is False:
            return None
        if self._structured_chain is not None:
            return self._structured_chain
        try:
            structured_model = self.bind_structured_output(JudgeResult)
            self._structured_chain = self._build_prompt() | structured_model
            self._structured_supported = True
            logger.info("Judge using with_structured_output(JudgeResult)")
            return self._structured_chain
        except Exception as build_error:
            logger.warning(
                f"with_structured_output unavailable, fallback to JsonOutputParser: {build_error}"
            )
            self._structured_supported = False
            return None

    def _get_fallback_chain(self) -> Runnable:
        if self._fallback_chain is None:
            parser = JsonOutputParser(pydantic_object=JudgeResult)
            self._fallback_chain = self._build_prompt() | self.bind_model() | parser
        return self._fallback_chain

    async def evaluate(
        self,
        coa_text: str,
        mission_desc: str,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """评估 COA，返回 `(overall_score, feedback, full_result_dict)`。"""
        criteria_text = ", ".join(self.criteria) if self.criteria else "feasibility"
        inputs = {
            "coa_table": coa_text,
            "mission": mission_desc,
            "criteria": criteria_text,
        }
        merged_config = self.merge_config(config, run_name="judge.evaluate")

        logger.info(f"Judge evaluating COA ({len(coa_text)} chars)")

        result_dict: Dict[str, Any]
        structured_chain = self._get_structured_chain()
        try:
            if structured_chain is not None:
                result_obj = await structured_chain.ainvoke(inputs, config=merged_config)
                result_dict = (
                    result_obj.model_dump()
                    if hasattr(result_obj, "model_dump")
                    else dict(result_obj)
                )
            else:
                result_dict = await self._get_fallback_chain().ainvoke(
                    inputs, config=merged_config
                )
                if not isinstance(result_dict, dict):
                    result_dict = dict(result_dict)
        except Exception as evaluate_error:
            logger.error(f"Judge evaluation failed: {evaluate_error}")
            result_dict = {
                "overall_score": 5.0,
                "feedback": f"Evaluation failed: {evaluate_error}",
                "verdict": "REVISE",
            }

        score = float(result_dict.get("overall_score", 5.0))
        feedback = str(result_dict.get("feedback", ""))

        self.log_interaction(
            "evaluate",
            self.system_prompt_raw,
            f"[coa_table ({len(coa_text)} chars)] | mission ({len(mission_desc)} chars)",
            str(result_dict),
        )
        logger.info(
            f"Judge score: {score} | verdict: {result_dict.get('verdict', 'UNKNOWN')}"
        )
        return score, feedback, result_dict
