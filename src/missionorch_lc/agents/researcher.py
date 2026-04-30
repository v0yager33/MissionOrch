"""Knowledge Researcher Agent —— ReAct 风格的 RAG 调研。

接收 MissionAnalystAgent 的分析结果（含 research_queries），
通过 LangChain 工具调用循环主动检索多个知识源，最终输出结构化「研究简报」。

设计要点：
- 工具调用次数硬上限：避免无限循环
- 每轮工具调用后把 ToolMessage 追加进消息历史，让 LLM 自决何时收敛
- 没有任何 RAG 工具可用时，优雅降级为「直接给一段研究简报模板说『未启用 RAG』」
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .base import BaseAgent

logger = logging.getLogger(__name__)


class KnowledgeResearcherAgent(BaseAgent):
    """ReAct 风格的研究员，主动调用 RAG 工具收集证据。"""

    prompt_variables = ("mission_analysis", "research_queries", "max_tool_calls")

    def __init__(self) -> None:
        super().__init__("researcher")
        self.max_tool_calls: int = int(self.agent_cfg.get("max_tool_calls", 6))
        logger.info(
            f"KnowledgeResearcherAgent initialized (max_tool_calls={self.max_tool_calls})"
        )

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "请开始研究，需要时直接调用工具；调研充分后输出最终的研究简报 Markdown。"),
            ],
        )

    async def research(
        self,
        mission_analysis: Dict[str, Any],
        tools: List[BaseTool],
        *,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """执行研究，返回 Markdown 研究简报。"""
        merged_config = self.merge_config(config, run_name="researcher.research")
        analysis_json = json.dumps(mission_analysis, ensure_ascii=False, indent=2)
        queries_text = "\n".join(
            f"- {q}" for q in mission_analysis.get("research_queries") or []
        ) or "（无明确建议查询，请基于任务分析自行决定检索方向）"

        prompt_inputs = {
            "mission_analysis": analysis_json,
            "research_queries": queries_text,
            "max_tool_calls": str(self.max_tool_calls),
        }

        if not tools:
            logger.warning(
                "Researcher: 没有任何 RAG 工具可用，跳过 ReAct 循环，返回降级简报"
            )
            return self._fallback_brief(mission_analysis, reason="无可用 RAG 工具")

        prompt = self._build_prompt()
        try:
            model = self.bind_model(tools=tools)
        except RuntimeError as bind_error:
            logger.error(
                f"Researcher: 当前模型不支持 tool calling: {bind_error}; 降级"
            )
            return self._fallback_brief(mission_analysis, reason=str(bind_error))

        tool_map = {t.name: t for t in tools}
        messages = prompt.format_messages(**prompt_inputs)

        tool_calls_used = 0
        for iteration in range(self.max_tool_calls + 1):
            ai_msg = await model.ainvoke(messages, config=merged_config)
            if not isinstance(ai_msg, AIMessage):
                raise TypeError(
                    f"Researcher: 预期 AIMessage，实际 {type(ai_msg).__name__}"
                )
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                # 模型主动收敛 → 直接拿正文
                final_text = (
                    ai_msg.content
                    if isinstance(ai_msg.content, str)
                    else str(ai_msg.content)
                )
                if not final_text or len(final_text.strip()) < 30:
                    final_text = self._fallback_brief(
                        mission_analysis, reason="LLM 未输出有效研究简报"
                    )
                self.log_interaction(
                    "research",
                    self.system_prompt_raw,
                    f"[tool_calls={tool_calls_used}, iterations={iteration}]",
                    final_text,
                )
                logger.info(
                    f"Researcher 收敛: iterations={iteration}, tool_calls={tool_calls_used}"
                )
                return final_text

            # 还有工具调用 → 检查是否超额
            if tool_calls_used >= self.max_tool_calls:
                logger.warning(
                    f"Researcher: 达到工具调用上限 {self.max_tool_calls}，强制收敛"
                )
                # 把"请马上输出"作为新一轮 user 提示，强制收敛
                from langchain_core.messages import HumanMessage

                messages.append(
                    HumanMessage(
                        content=(
                            "已达到工具调用上限，请基于已有证据立即输出最终的"
                            "『研究简报』Markdown，不要再调用任何工具。"
                        )
                    )
                )
                # 用不带工具的 model 再来一发
                close_model = self.bind_model()
                close_msg = await close_model.ainvoke(messages, config=merged_config)
                final_text = (
                    close_msg.content
                    if isinstance(close_msg.content, str)
                    else str(close_msg.content)
                )
                if not final_text or len(final_text.strip()) < 30:
                    final_text = self._fallback_brief(
                        mission_analysis, reason="超额收敛后 LLM 未输出有效简报"
                    )
                return final_text

            # 执行工具
            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args") or {}
                tool_id = call.get("id", "")
                tool = tool_map.get(tool_name)
                if tool is None:
                    content = f"[错误] 工具 {tool_name} 未注册，请使用：{list(tool_map)}"
                else:
                    try:
                        result = await tool.ainvoke(tool_args)
                        content = str(result)
                    except Exception as tool_error:
                        content = f"[工具执行失败] {tool_error}"
                tool_calls_used += 1
                messages.append(ToolMessage(content=content, tool_call_id=tool_id))

        # 走到这里说明 for 完整跑完仍未收敛
        return self._fallback_brief(
            mission_analysis, reason="ReAct 循环未在限定轮次内收敛"
        )

    @staticmethod
    def _fallback_brief(mission_analysis: Dict[str, Any], reason: str) -> str:
        """降级简报：把 analyst 的输出原样回填，保证流程不中断。"""
        objs = mission_analysis.get("objectives") or []
        constraints = mission_analysis.get("constraints") or []
        uncertainties = mission_analysis.get("uncertainties") or []
        return (
            "## 研究简报（降级版本）\n\n"
            f"**说明**：{reason}。以下内容来自任务分析，未经 RAG 检索增强。\n\n"
            "### 关键事实\n"
            + ("\n".join(f"- {o.get('description', '')}" for o in objs) or "- （无）")
            + "\n\n### 适用条令 / 标准\n- （未检索，请 Planner 基于通识推理）\n"
            "\n### 历史经验 / 类似战例\n- （未检索）\n"
            "\n### 关键术语澄清\n- （未检索）\n"
            "\n### 仍存在的不确定性\n"
            + ("\n".join(f"- {u}" for u in uncertainties) or "- （无）")
            + (
                "\n- 约束未在 RAG 中验证：\n"
                + "\n".join(f"  - {c}" for c in constraints)
                if constraints
                else ""
            )
        )
