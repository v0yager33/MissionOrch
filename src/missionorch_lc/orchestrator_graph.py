"""LangGraph StateGraph 版 COA Orchestrator —— 完整 6-Agent 流水线。

完整流程：

    ┌──────────┐
    │ analyst  │  分析任务，提取实体 / 约束 / 研究问题
    └────┬─────┘
         ▼
    ┌──────────┐
    │researcher│  ReAct 主动调用 RAG 工具收集证据 → 研究简报
    └────┬─────┘
         ▼
    ┌──────────┐
    │  planner │◄────────────────────┐
    └────┬─────┘                      │
         ▼                            │
    ┌──────────┐  score >= threshold  │
    │  judge   │ ───────────────► ┌──┴───────┐
    └────┬─────┘                  │ finalize │
         │ score < threshold       │  解析+   │
         ▼                         │  验证+   │
    ┌──────────┐                  │ 格式化   │
    │reflector │                  └──────────┘
    └────┬─────┘
         └─────────► planner

终止条件：
1. score >= quality_threshold
2. iteration >= max_iterations

任何前置 Agent 缺失（如未启用 RAG → 跳过 researcher），系统自动降级。

用法::

    from missionorch_lc.orchestrator_graph import run_graph
    result = await run_graph(mission_input="...")
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from .agents import (
    JudgeAgent,
    KnowledgeResearcherAgent,
    MissionAnalystAgent,
    PlannerAgent,
    ReflectorAgent,
    ValidatorAgent,
)
from .core.callbacks import StageTimingCallback, TokenUsageCallback
from .core.coa_parser import COATableParser
from .core.coa_transformer import COATransformer
from .core.rag_manager import RAGManager
from .schemas.coa import COA
from .tools.rag_tools import build_rag_tools

logger = logging.getLogger(__name__)


# ── 状态定义 ──


class COAGraphState(TypedDict, total=False):
    """LangGraph 的全局状态。"""

    # 输入
    mission_input: str

    # 前置阶段产物
    mission_analysis: Dict[str, Any]
    research_brief: str

    # 规划循环
    current_coa_text: str
    iteration: int
    max_iterations: int
    quality_threshold: float
    early_stop: bool
    best_score: float
    history: List[Dict[str, Any]]
    reflection: str

    # 控制
    skip_research: bool

    # 最终输出
    final_coa_obj: Optional[COA]
    parse_success: bool
    validation: Dict[str, Any]
    outputs: Dict[str, Any]
    token_usage: Dict[str, Any]
    timing: List[Dict[str, Any]]
    stage_log: List[Dict[str, Any]]
    error: Optional[str]


# ── 内部辅助 ──


def _append_stage_log(
    state: COAGraphState, entry: Dict[str, Any]
) -> List[Dict[str, Any]]:
    log = list(state.get("stage_log") or [])
    log.append(entry)
    return log


def _compose_planner_knowledge(state: COAGraphState) -> str:
    """合并 analyst + researcher 输出，作为 Planner 的 RAG 上下文。"""
    parts: List[str] = []
    analysis = state.get("mission_analysis") or {}
    if analysis:
        parts.append(
            "## 任务分析（来自 Analyst）\n```json\n"
            + _json.dumps(analysis, ensure_ascii=False, indent=2)
            + "\n```"
        )
    brief = state.get("research_brief")
    if brief:
        parts.append(brief)
    return "\n\n".join(parts)


# ── 节点工厂 ──


def _make_analyst_node(analyst: MissionAnalystAgent):
    async def analyst_node(state: COAGraphState) -> Dict[str, Any]:
        mission = state["mission_input"]
        logger.info(f"[Graph/Analyst] mission len={len(mission)}")
        analysis = await analyst.analyze(mission)
        log = _append_stage_log(
            state,
            {
                "stage": "analyst",
                "objectives": len(analysis.get("objectives", [])),
                "research_queries": len(analysis.get("research_queries", [])),
            },
        )
        return {"mission_analysis": analysis, "stage_log": log}

    return analyst_node


def _make_researcher_node(
    researcher: KnowledgeResearcherAgent, rag_manager: RAGManager
):
    async def researcher_node(state: COAGraphState) -> Dict[str, Any]:
        if state.get("skip_research") or not rag_manager.is_enabled():
            logger.info("[Graph/Researcher] 跳过（skip_research 或 RAG 未启用）")
            return {
                "research_brief": researcher._fallback_brief(
                    state.get("mission_analysis") or {},
                    reason="RAG 未启用 / 显式跳过",
                ),
                "stage_log": _append_stage_log(
                    state, {"stage": "researcher", "skipped": True}
                ),
            }

        analysis = state.get("mission_analysis") or {}
        tools = build_rag_tools(rag_manager=rag_manager)
        logger.info(f"[Graph/Researcher] 工具数={len(tools)}")
        brief = await researcher.research(analysis, tools)
        log = _append_stage_log(
            state,
            {
                "stage": "researcher",
                "tools_available": [t.name for t in tools],
                "brief_len": len(brief),
            },
        )
        return {"research_brief": brief, "stage_log": log}

    return researcher_node


def _make_planner_node(planner: PlannerAgent):
    async def planner_node(state: COAGraphState) -> Dict[str, Any]:
        mission = state["mission_input"]
        reflection = state.get("reflection", "")
        previous_coa = state.get("current_coa_text")
        iteration = state.get("iteration", 0)
        knowledge = _compose_planner_knowledge(state)

        logger.info(
            f"[Graph/Planner] iter={iteration}, knowledge_len={len(knowledge)}, "
            f"reflection_len={len(reflection)}"
        )
        coa_text = await planner.generate_coa(
            mission,
            knowledge=knowledge,
            reflection=reflection,
            previous_coa_text=previous_coa if iteration > 0 else None,
        )
        log = _append_stage_log(
            state,
            {
                "stage": "planner",
                "iteration": iteration,
                "coa_len": len(coa_text),
                "used_knowledge": bool(knowledge),
            },
        )
        return {"current_coa_text": coa_text, "stage_log": log}

    return planner_node


def _make_judge_node(judge: JudgeAgent):
    async def judge_node(state: COAGraphState) -> Dict[str, Any]:
        coa_text = state["current_coa_text"]
        mission = state["mission_input"]
        iteration = state.get("iteration", 0) + 1

        logger.info(f"[Graph/Judge] iter={iteration}")
        try:
            score, feedback, details = await judge.evaluate(coa_text, mission)
        except Exception as eval_error:
            logger.error(f"[Graph/Judge] Evaluation failed: {eval_error}")
            score, feedback, details = 5.0, f"Evaluation failed: {eval_error}", {}

        verdict = details.get("verdict", "UNKNOWN")
        history = list(state.get("history") or [])
        history.append(
            {
                "iteration": iteration,
                "score": score,
                "feedback": feedback,
                "verdict": verdict,
            }
        )
        best_score = max(state.get("best_score", 0.0), score)
        log = _append_stage_log(
            state,
            {
                "stage": "judge",
                "iteration": iteration,
                "score": score,
                "verdict": verdict,
            },
        )

        return {
            "iteration": iteration,
            "history": history,
            "best_score": best_score,
            "reflection": feedback,  # 暂存给 reflector
            "stage_log": log,
        }

    return judge_node


def _make_reflector_node(reflector: ReflectorAgent):
    async def reflector_node(state: COAGraphState) -> Dict[str, Any]:
        coa_text = state["current_coa_text"]
        feedback = state.get("reflection", "")
        iteration = state.get("iteration", 1)

        logger.info(f"[Graph/Reflector] iter={iteration}")
        try:
            reflection = await reflector.reflect(coa_text, feedback, iteration)
        except Exception as reflect_error:
            logger.error(f"[Graph/Reflector] failed: {reflect_error}")
            reflection = f"Previous evaluation feedback: {feedback}"

        log = _append_stage_log(
            state,
            {
                "stage": "reflector",
                "iteration": iteration,
                "reflection_len": len(reflection),
            },
        )
        return {"reflection": reflection, "stage_log": log}

    return reflector_node


def _make_finalize_node(
    parser: COATableParser,
    transformer: COATransformer,
    validator: ValidatorAgent,
    token_callback: TokenUsageCallback,
    timing_callback: StageTimingCallback,
):
    async def finalize_node(state: COAGraphState) -> Dict[str, Any]:
        coa_text = state["current_coa_text"]
        mission = state["mission_input"]

        # 解析
        try:
            coa_obj = parser.parse(coa_text)
            parse_success = True
        except Exception as parse_error:
            logger.error(f"[Graph/Finalize] Parse failed: {parse_error}")
            coa_obj = COA(
                description="Parse error",
                metadata={"parse_error": str(parse_error)},
            )
            parse_success = False

        # 验证
        validation_data: Dict[str, Any] = {}
        if parse_success:
            try:
                is_valid, v_feedback, v_extras = (
                    await validator.validate_and_extract_matrix(coa_obj, mission)
                )
                validation_data = {
                    "is_valid": is_valid,
                    "validation_feedback": v_feedback,
                    **v_extras,
                }
            except Exception as validate_error:
                logger.error(f"[Graph/Finalize] Validation failed: {validate_error}")
                validation_data = {
                    "is_valid": False,
                    "validation_feedback": str(validate_error),
                }

        # 格式化
        outputs: Dict[str, Any] = {}
        if parse_success:
            outputs = {
                "json_format": transformer.coa_to_json(coa_obj),
                "yaml_format": transformer.coa_to_yaml(coa_obj),
                "flat_matrix": transformer.coa_to_flat_matrix(coa_obj),
                "condensed_format": transformer.coa_to_condensed_format(coa_obj),
            }

        log = _append_stage_log(
            state,
            {
                "stage": "finalize",
                "parse_success": parse_success,
                "is_valid": validation_data.get("is_valid"),
            },
        )

        return {
            "final_coa_obj": coa_obj,
            "parse_success": parse_success,
            "validation": validation_data,
            "outputs": outputs,
            "token_usage": {
                "by_model": token_callback.summary(),
                "total_tokens": token_callback.total_tokens(),
            },
            "timing": timing_callback.summary(),
            "stage_log": log,
        }

    return finalize_node


# ── 条件边 ──


def _should_continue(state: COAGraphState) -> str:
    """Judge 之后的路由：质量达标 / 迭代上限 → finalize，否则 → reflector。"""
    history = state.get("history") or []
    if not history:
        return "reflector"

    latest = history[-1]
    score = float(latest.get("score", 0))
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    threshold = state.get("quality_threshold", 8.0)
    early_stop = state.get("early_stop", True)

    if early_stop and score >= threshold:
        logger.info(f"[Graph/Router] Quality met ({score} >= {threshold}), → finalize")
        return "finalize"

    if iteration >= max_iterations:
        logger.info(f"[Graph/Router] Max iterations reached ({iteration}), → finalize")
        return "finalize"

    logger.info(
        f"[Graph/Router] Score {score} < {threshold}, "
        f"iter {iteration}/{max_iterations}, → reflector"
    )
    return "reflector"


# ── 图构建 ──


def _make_disabled_rag_manager(rag_config_path: str) -> RAGManager:
    """构造一个 disabled 的 RAGManager，避免触发模型加载。"""
    manager = RAGManager.__new__(RAGManager)
    manager.config_path = rag_config_path
    manager.config = {}
    manager.rag_cfg = {}
    manager.retrieval_cfg = {}
    manager.enabled = False
    manager._engines = {}
    manager._initialized = {}
    return manager


def build_graph(
    *,
    rag_config_path: str = "config/rag.yaml",
    use_rag: bool = True,
):
    """构建完整 6-Agent LangGraph，并返回
    ``(compiled_graph, token_callback, timing_callback, rag_manager)``。

    Args:
        rag_config_path: RAG 配置文件路径
        use_rag: False 时强制跳过 researcher 节点（仍构建以保持图结构一致）
    """
    analyst = MissionAnalystAgent()
    researcher = KnowledgeResearcherAgent()
    planner = PlannerAgent()
    judge = JudgeAgent()
    reflector = ReflectorAgent()
    validator = ValidatorAgent()
    parser = COATableParser()
    transformer = COATransformer()
    token_callback = TokenUsageCallback()
    timing_callback = StageTimingCallback()

    rag_manager = (
        RAGManager(rag_config_path)
        if use_rag
        else _make_disabled_rag_manager(rag_config_path)
    )

    graph = StateGraph(COAGraphState)

    graph.add_node("analyst", _make_analyst_node(analyst))
    graph.add_node("researcher", _make_researcher_node(researcher, rag_manager))
    graph.add_node("planner", _make_planner_node(planner))
    graph.add_node("judge", _make_judge_node(judge))
    graph.add_node("reflector", _make_reflector_node(reflector))
    graph.add_node(
        "finalize",
        _make_finalize_node(
            parser, transformer, validator, token_callback, timing_callback
        ),
    )

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "judge")
    graph.add_conditional_edges(
        "judge",
        _should_continue,
        {"reflector": "reflector", "finalize": "finalize"},
    )
    graph.add_edge("reflector", "planner")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info(
        "LangGraph 6-Agent COA orchestrator compiled (use_rag=%s, rag_sources=%s)",
        use_rag,
        rag_manager.list_sources(),
    )

    return compiled, token_callback, timing_callback, rag_manager


# ── 便捷运行函数 ──


async def run_graph(
    mission_input: str,
    *,
    rag_config_path: str = "config/rag.yaml",
    use_rag: bool = True,
    max_iterations: int = 3,
    quality_threshold: float = 8.0,
    early_stop: bool = True,
    extra_callbacks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """一键运行完整流水线。

    Returns:
        与原 ``COAOrchestrator.generate`` 兼容的结果字典（多了
        ``mission_analysis`` / ``research_brief`` / ``stage_log`` 字段）。
    """
    compiled, token_callback, timing_callback, rag_manager = build_graph(
        rag_config_path=rag_config_path, use_rag=use_rag
    )

    callbacks: List[Any] = [token_callback, timing_callback]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    run_config: RunnableConfig = {"callbacks": callbacks}  # type: ignore[assignment]

    initial_state: COAGraphState = {
        "mission_input": mission_input,
        "current_coa_text": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold,
        "early_stop": early_stop,
        "best_score": 0.0,
        "history": [],
        "reflection": "",
        "skip_research": not use_rag,
        "stage_log": [],
    }

    final_state = await compiled.ainvoke(initial_state, config=run_config)

    history = final_state.get("history") or []
    final_score = history[-1]["score"] if history else 0.0

    result: Dict[str, Any] = {
        "coa_table": final_state.get("current_coa_text", ""),
        "final_coa": (
            final_state["final_coa_obj"].model_dump(mode="json")
            if final_state.get("final_coa_obj")
            else {}
        ),
        "parse_success": final_state.get("parse_success", False),
        "iterations": final_state.get("iteration", 0),
        "final_score": final_score,
        "best_score": final_state.get("best_score", 0.0),
        "history": history,
        "validation": final_state.get("validation", {}),
        "outputs": final_state.get("outputs", {}),
        "token_usage": final_state.get("token_usage", {}),
        "timing": final_state.get("timing", []),
        "mission_analysis": final_state.get("mission_analysis", {}),
        "research_brief": final_state.get("research_brief", ""),
        "stage_log": final_state.get("stage_log", []),
        "rag_sources": rag_manager.list_sources(),
        "rag_enabled": rag_manager.is_enabled(),
    }

    logger.info(
        f"LangGraph COA completed: {result['iterations']} iter, "
        f"score={final_score}, "
        f"tokens={result.get('token_usage', {}).get('total_tokens', 0)}, "
        f"rag={result['rag_enabled']}({len(result['rag_sources'])} sources)"
    )
    return result
