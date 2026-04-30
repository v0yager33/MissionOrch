"""Planner Agent —— 生成 COA 矩阵。

LCEL 组合：
    ChatPromptTemplate | ChatModel | StrOutputParser

模板说明：
- `planner.txt` 中仅 `{rag_context}` 是真正的占位符，其余花括号（Markdown/JSON 示例）
  由 `prompt_loader.escape_prompt_text` 自动转义为字面量。
- User 消息采用 LangChain 默认 f-string 格式，占位符只有显式声明的几个。
- 支持工具调用：传入 tools 时使用 `bind_tools`；进一步 ReAct 循环通过
  `generate_coa_with_tools` 实现。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .base import BaseAgent

logger = logging.getLogger(__name__)

# User 消息模板：变量白名单见 `_USER_VARIABLES`
_USER_TEMPLATE = """基于以下任务描述生成COA（行动方案）矩阵：

【任务描述】
{mission_desc}
{reflection_section}{previous_coa_section}
【要求】
1. 战役级抽象：定义Who/What/When/Where，不涉及How
2. 以"作战单元 × 阶段"矩阵为核心输出
3. 至少3个作战单元（纵轴）、至少3个阶段（横轴）
4. 每个单元格至少2个行动描述
5. 阶段转换必须有明确的触发条件（条件/事件驱动，禁止使用固定时间窗口）
6. 效果链至少2个效果
7. 包含决策点和关键风险

请严格按照系统提示词中的COA矩阵表格模板格式输出，使用Markdown表格，不要输出JSON。
"""

_USER_VARIABLES = ("mission_desc", "reflection_section", "previous_coa_section")


class PlannerAgent(BaseAgent):
    """规划智能体 —— 输出自然语言 COA 矩阵。"""

    prompt_variables = ("rag_context",)

    def __init__(self) -> None:
        super().__init__("planner")
        self.output_parser = StrOutputParser()
        logger.info("PlannerAgent initialized")

    # ── Prompt 构建 ──
    def _build_prompt(self) -> ChatPromptTemplate:
        # system 通过 base.system_prompt 自动转义，仅保留白名单变量 `rag_context`
        # user 由我们完全控制，用 f-string 默认风格
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", _USER_TEMPLATE),
            ],
        )

    def _build_inputs(
        self,
        mission_desc: str,
        knowledge: str,
        reflection: str,
        previous_coa_text: Optional[str],
    ) -> Dict[str, Any]:
        reflection_section = (
            f"\n【改进建议（请根据以下反馈改进）】\n{reflection}\n" if reflection else ""
        )
        previous_coa_section = (
            f"\n【上一版COA表格（请在此基础上改进）】\n{previous_coa_text}\n"
            if previous_coa_text
            else ""
        )
        return {
            "rag_context": knowledge or "（未提供 RAG 上下文）",
            "mission_desc": mission_desc,
            "reflection_section": reflection_section,
            "previous_coa_section": previous_coa_section,
        }

    # ── 业务方法 ──
    async def generate_coa(
        self,
        mission_desc: str,
        knowledge: str = "",
        reflection: str = "",
        previous_coa_text: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """生成 COA（自然语言 Markdown 表格）。

        Args:
            tools: 若提供，模型将以 tool-calling 模式工作；此时返回结果若包含
                `tool_calls` 则应改用 `generate_coa_with_tools` 走 ReAct 循环。
            config: RunnableConfig，用于注入 callbacks / tags / run_name 等。
        """
        prompt = self._build_prompt()
        model = self.bind_model(tools=tools)
        chain = prompt | model | self.output_parser

        inputs = self._build_inputs(mission_desc, knowledge, reflection, previous_coa_text)
        merged_config = self.merge_config(config, run_name="planner.generate_coa")

        logger.info(f"Planner generating COA (mission len={len(mission_desc)})")
        response = await chain.ainvoke(inputs, config=merged_config)

        if not response or len(response.strip()) < 50:
            raise ValueError("COA generation returned empty/too-short response")

        self.log_interaction(
            "generate_coa",
            self.system_prompt_raw,
            f"mission({len(mission_desc)})+reflection({len(reflection)})+"
            f"previous({len(previous_coa_text or '')})",
            response,
        )
        logger.info(f"Planner returned COA table ({len(response)} chars)")
        return response

    async def generate_coa_with_tools(
        self,
        mission_desc: str,
        tools: List[BaseTool],
        *,
        knowledge: str = "",
        reflection: str = "",
        previous_coa_text: Optional[str] = None,
        max_tool_iterations: int = 4,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """ReAct 风格：允许 Planner 先用工具调研，再生成 COA。

        手动驱动 tool-calling loop；生产级可改用 LangGraph ToolNode。
        """
        if not tools:
            raise ValueError("generate_coa_with_tools requires at least one tool")

        tool_map = {t.name: t for t in tools}
        prompt = self._build_prompt()
        model = self.bind_model(tools=tools)
        inputs = self._build_inputs(mission_desc, knowledge, reflection, previous_coa_text)

        messages = prompt.format_messages(**inputs)
        merged_config = self.merge_config(
            config, run_name="planner.generate_coa_with_tools"
        )

        for iteration in range(max_tool_iterations):
            ai_msg = await model.ainvoke(messages, config=merged_config)
            if not isinstance(ai_msg, AIMessage):
                raise TypeError(
                    f"Planner expected AIMessage from tool-calling model, got {type(ai_msg).__name__}"
                )
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                final_text = (
                    ai_msg.content
                    if isinstance(ai_msg.content, str)
                    else str(ai_msg.content)
                )
                if not final_text or len(final_text.strip()) < 50:
                    raise ValueError("COA generation returned empty/too-short response")
                self.log_interaction(
                    "generate_coa_with_tools",
                    self.system_prompt_raw,
                    f"[tool_iterations={iteration}]",
                    final_text,
                )
                return final_text

            # 执行每个工具调用，把结果追加到消息历史
            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args") or {}
                tool_id = call.get("id", "")
                tool = tool_map.get(tool_name)
                if tool is None:
                    content = f"[错误] 工具 {tool_name} 未注册"
                else:
                    try:
                        result = await tool.ainvoke(tool_args)
                        content = str(result)
                    except Exception as tool_error:
                        content = f"[工具执行失败] {tool_error}"
                messages.append(ToolMessage(content=content, tool_call_id=tool_id))

        raise RuntimeError(
            f"Planner tool-calling loop exceeded {max_tool_iterations} iterations"
            " without final answer"
        )