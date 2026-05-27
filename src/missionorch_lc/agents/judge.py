"""Judge Agent —— 评估 COA 质量，输出结构化 JudgeResult。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from ..schemas.judge_result import JudgeResult
from .base import BaseAgent

logger = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """评估智能体 —— 直接评估自然语言 COA 表格。"""

    prompt_variables = ("coa_table", "mission", "criteria")

    def __init__(self) -> None:
        super().__init__("judge")
        self.criteria = self.agent_cfg.get("criteria", [])

        prompt = self._build_prompt()
        self._structured_chain, self._fallback_chain = self.build_structured_chain_pair(
            JudgeResult, prompt
        )
        logger.info(f"JudgeAgent initialized with criteria: {self.criteria}")

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "请按要求输出评估结果 JSON。"),
            ],
        )

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

        try:
            result_dict = await self.invoke_structured(
                JudgeResult,
                self._structured_chain,
                self._fallback_chain,
                inputs,
                config=merged_config,
            )
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
