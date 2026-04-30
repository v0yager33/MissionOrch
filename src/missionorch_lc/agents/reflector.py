"""Reflector Agent —— 基于 Judge 反馈生成改进建议。"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReflectorAgent(BaseAgent):
    """反思智能体 —— 输出自然语言改进建议。"""

    # reflector.txt 的占位符白名单
    prompt_variables = ("coa_table", "judge_feedback", "iteration")

    def __init__(self) -> None:
        super().__init__("reflector")
        self.use_reasoning = bool(self.agent_cfg.get("use_reasoning_chain", False))
        self.output_parser = StrOutputParser()
        logger.info(f"ReflectorAgent initialized (use_reasoning={self.use_reasoning})")

    async def reflect(
        self,
        coa_text: str,
        feedback: str,
        iteration: int,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """基于 COA 与 Judge feedback 生成反思。"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "请输出结构化的修改建议，足够具体以便直接重写 COA 矩阵。"),
            ],
        )

        # 推理型模型使用略高温度增强多样性
        extra_bind = {"temperature": 0.6} if self.use_reasoning else None
        model = self.bind_model(extra_bind=extra_bind)

        chain = prompt | model | self.output_parser
        merged_config = self.merge_config(
            config, run_name=f"reflector.reflect.iter{iteration}"
        )

        logger.info(f"Reflector generating reflection (iter={iteration})")
        response = await chain.ainvoke(
            {
                "coa_table": coa_text,
                "judge_feedback": feedback,
                "iteration": str(iteration),
            },
            config=merged_config,
        )

        self.log_interaction(
            "reflect",
            self.system_prompt_raw,
            f"[coa_table ({len(coa_text)} chars)] | feedback ({len(feedback)} chars) | iter={iteration}",
            response,
        )
        logger.info(f"Reflection generated ({len(response)} chars)")
        return response
