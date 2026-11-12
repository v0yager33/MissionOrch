"""Planner Agent —— 生成 COA 矩阵（自然语言 Markdown 表格）。

LCEL 组合：
    ChatPromptTemplate | ChatModel   （直接拿 AIMessage，以便检查 finish_reason）

模板说明：
- `planner.txt` 中仅 `{rag_context}` 是真正的占位符，其余花括号（Markdown/JSON 示例）
  由 `prompt_loader.escape_prompt_text` 自动转义为字面量。
- User 消息采用 LangChain 默认 f-string 格式，占位符只有 `_USER_VARIABLES` 中的几个。

修订模式（``workflow.revision_mode``）：
- ``full``：每轮输出完整方案；
- ``section_patch``（默认）：有上一版时只输出需要修改的章节，再用
  `coa_sections.merge_sections` 合并回上一版。输出 token 大幅减少，修订轮更快。

预期战果：输入侧给定的条目通过 ``expected_outcomes`` 传入，要求原样写进第八节；
它们只记录、不参与评分。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from ..core.coa_sections import is_full_document, merge_sections
from ..core.expected_outcomes import outcomes_to_prompt
from .base import BaseAgent

logger = logging.getLogger(__name__)

# User 消息模板：变量白名单见 `_USER_VARIABLES`
_USER_TEMPLATE = """基于以下任务描述生成 COA（行动方案）矩阵：

【任务描述】
{mission_desc}
{expected_outcomes_section}{structure_section}{reflection_section}{previous_coa_section}
【要求】
1. 任务级规划：每个独立编队/编组单独占一行，写清每个阶段做什么、在哪里、由什么条件触发
2. 阶段 ≥ 5 个，覆盖起飞集结 → 出航 → 进入 → 交战 → 脱离返航的完整周期
3. 每个单元格 2~5 个行动，优先使用受控动词、平台代号与地理占位语法
4. 阶段转换由条件/事件触发，禁止固定时间窗口
5. 效果链 ≥ 3 个效果；决策点 ≥ 2 个，每个 ≥ 2 个选项；给出关键风险
6. 第六节实体清单、第七节地理参考、第八节预期战果必须齐全
{output_mode_section}"""

_USER_VARIABLES = (
    "mission_desc",
    "expected_outcomes_section",
    "structure_section",
    "reflection_section",
    "previous_coa_section",
    "output_mode_section",
)

_FULL_OUTPUT = (
    "\n请严格按照系统提示词中的模板输出完整的八个章节（Markdown 表格），"
    "不要输出 JSON，不要输出代码块。\n"
)

_PATCH_OUTPUT = """
【输出方式：只输出修改的章节】
- 只输出需要修改的章节；每个章节以上一版中的原标题行开头（如"## 二、COA矩阵"），并给出该章节修改后的**完整内容**
- 没有问题的章节不要输出，系统会自动沿用上一版
- 不要输出解释性文字、JSON 或代码块
"""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content if isinstance(content, str) else str(content or "")


def _finish_reason(message: Any) -> str:
    meta = getattr(message, "response_metadata", None) or {}
    return str(meta.get("finish_reason") or meta.get("stop_reason") or "")


class PlannerAgent(BaseAgent):
    """规划智能体 —— 输出自然语言 COA 矩阵。"""

    prompt_variables = ("rag_context",)

    def __init__(self, profile: Optional[str] = None) -> None:
        super().__init__("planner", profile=profile)
        self.revision_mode: str = str(
            self.agent_cfg.get("revision_mode")
            or self.workflow_cfg.get("revision_mode")
            or "full"
        )
        self.last_finish_reason: str = ""
        self.last_truncated: bool = False
        self.last_patched_sections: List[str] = []
        logger.info(f"PlannerAgent initialized (revision_mode={self.revision_mode})")

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
        *,
        expected_outcomes: Optional[Sequence[Dict[str, Any]]] = None,
        structure_report: str = "",
        patch_mode: bool = False,
    ) -> Dict[str, Any]:
        outcomes_block = outcomes_to_prompt(list(expected_outcomes or []))
        return {
            "rag_context": knowledge or "（未提供 RAG 上下文）",
            "mission_desc": mission_desc,
            "expected_outcomes_section": f"\n{outcomes_block}\n" if outcomes_block else "",
            "structure_section": (
                f"\n【结构检查（代码自动检出，必须修正）】\n{structure_report}\n"
                if structure_report
                else ""
            ),
            "reflection_section": (
                f"\n【修订指令（逐条落实）】\n{reflection}\n" if reflection else ""
            ),
            "previous_coa_section": (
                f"\n【上一版COA（在此基础上修改，未指出问题的内容保持不变）】\n{previous_coa_text}\n"
                if previous_coa_text
                else ""
            ),
            "output_mode_section": _PATCH_OUTPUT if patch_mode else _FULL_OUTPUT,
        }

    def _use_patch(self, previous_coa_text: Optional[str], revision_mode: Optional[str]) -> bool:
        mode = revision_mode or self.revision_mode
        return mode == "section_patch" and is_full_document(previous_coa_text or "")

    # ── 业务方法 ──
    async def generate_coa(
        self,
        mission_desc: str,
        knowledge: str = "",
        reflection: str = "",
        previous_coa_text: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[RunnableConfig] = None,
        *,
        expected_outcomes: Optional[Sequence[Dict[str, Any]]] = None,
        structure_report: str = "",
        revision_mode: Optional[str] = None,
    ) -> str:
        """生成 COA（自然语言 Markdown 表格），返回完整方案文本。

        Args:
            tools: 若提供，模型将以 tool-calling 模式工作；此时返回结果若包含
                `tool_calls` 则应改用 `generate_coa_with_tools` 走 ReAct 循环。
            config: RunnableConfig，用于注入 callbacks / tags / run_name 等。
            expected_outcomes: 输入侧给定的预期战果，要求原样写进第八节。
            structure_report: 确定性结构检查的 Markdown 报告（修订轮传入）。
            revision_mode: 覆盖配置中的 ``revision_mode``（full / section_patch）。
        """
        patch_mode = self._use_patch(previous_coa_text, revision_mode)
        prompt = self._build_prompt()
        model = self.bind_model(tools=tools)
        chain = prompt | model

        inputs = self._build_inputs(
            mission_desc,
            knowledge,
            reflection,
            previous_coa_text,
            expected_outcomes=expected_outcomes,
            structure_report=structure_report,
            patch_mode=patch_mode,
        )
        merged_config = self.merge_config(config, run_name="planner.generate_coa")

        logger.info(
            f"Planner generating COA (mission len={len(mission_desc)}, "
            f"mode={'patch' if patch_mode else 'full'})"
        )
        message = await chain.ainvoke(inputs, config=merged_config)
        response = _message_text(message)
        self.last_finish_reason = _finish_reason(message)
        self.last_truncated = self.last_finish_reason == "length"
        if self.last_truncated:
            logger.warning("Planner output hit max_tokens (finish_reason=length)")

        if not response or len(response.strip()) < 50:
            raise ValueError("COA generation returned empty/too-short response")

        self.last_patched_sections = []
        if patch_mode and not is_full_document(response):
            merged, touched = merge_sections(previous_coa_text or "", response)
            if touched:
                self.last_patched_sections = touched
                logger.info(f"Planner patched sections: {touched}")
                response = merged
            else:
                logger.warning("Planner patch contained no sections; keeping previous COA")
                response = previous_coa_text or response

        self.log_interaction(
            "generate_coa",
            self.system_prompt_raw,
            f"mission({len(mission_desc)})+reflection({len(reflection)})+"
            f"previous({len(previous_coa_text or '')})+patch({patch_mode})",
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
        expected_outcomes: Optional[Sequence[Dict[str, Any]]] = None,
        max_tool_iterations: int = 4,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """ReAct 风格：允许 Planner 先用工具调研，再生成完整 COA。

        手动驱动 tool-calling loop；生产级可改用 LangGraph ToolNode。
        """
        if not tools:
            raise ValueError("generate_coa_with_tools requires at least one tool")

        tool_map = {t.name: t for t in tools}
        prompt = self._build_prompt()
        model = self.bind_model(tools=tools)
        inputs = self._build_inputs(
            mission_desc,
            knowledge,
            reflection,
            previous_coa_text,
            expected_outcomes=expected_outcomes,
        )

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
                final_text = _message_text(ai_msg)
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
