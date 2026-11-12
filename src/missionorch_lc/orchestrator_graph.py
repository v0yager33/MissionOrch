"""LangGraph StateGraph 版 COA Orchestrator —— 完整 6-Agent 流水线。

完整流程：

    ┌──────────┐
    │ analyst  │  分析任务，提取实体 / 约束 / 研究问题 / 预期战果
    └────┬─────┘
         ▼
    ┌──────────┐
    │researcher│  ReAct 主动调用 RAG 工具收集证据 → 研究简报
    └────┬─────┘
         ▼
    ┌──────────┐
    │  planner │◄──────────────┬──────────────┐
    └────┬─────┘                │              │
         ▼                      │ use_reflector=false（默认）
    ┌──────────┐  stop_reason   │              │
    │  judge   │ ──────────► finalize       ┌──┴───────┐
    └────┬─────┘  （解析+验证+格式化）        │reflector │
         └──────────── use_reflector=true ──►└──────────┘

judge 节点先跑确定性结构检查（coa_checks，毫秒级），再让 LLM 只评估战术合理性，
并直接给出 revision_instructions；默认不再单独调用 reflector。

终止条件（stop_reason）：
1. quality_met：score >= quality_threshold 且结构检查无必须修正项
2. max_iterations：达到迭代上限
3. plateau：连续 patience 轮分数提升 < min_improvement
4. planner_error：修订轮 planner 调用失败（保留已有最优版本）

最终输出采用历次迭代中得分最高的版本（best），而不是最后一版。

预期战果（expected_outcomes）只透传：输入侧由用户参数 + 任务文本 + analyst 汇总，
输出侧由 planner 第八节给出并与输入条目合并，不参与评分与路由。

用法::

    from missionorch_lc.orchestrator_graph import run_graph
    result = await run_graph(mission_input="...", expected_outcomes=["击落敌机≥4架"])
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
import weakref
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

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
from .agents.base import load_agents_config
from .core.callbacks import StageTimingCallback, TokenUsageCallback
from .core.coa_checks import check_coa
from .core.coa_parser import COATableParser
from .core.coa_transformer import COATransformer
from .core.expected_outcomes import coerce_outcomes, extract_from_mission, merge_outcomes
from .core.memory import save_run_memory
from .core.rag_manager import RAGManager
from .core.settings import get_settings, resolve_path
from .schemas.coa import COA, ExpectedOutcome
from .schemas.result import COAResult
from .tools.rag_tools import build_rag_tools

logger = logging.getLogger(__name__)


# ── 状态定义 ──


class COAGraphState(TypedDict, total=False):
    """LangGraph 的全局状态。"""

    # 输入
    mission_input: str
    expected_outcomes_input: List[Dict[str, Any]]  # 用户参数 + 任务文本 + analyst 汇总

    # 前置阶段产物
    mission_analysis: Dict[str, Any]
    research_brief: str

    # 规划循环
    current_coa_text: str
    revision_base: str            # 下一轮 planner 的修改基准（分数回退时为 best 版本）
    iteration: int
    max_iterations: int
    quality_threshold: float
    early_stop: bool
    use_reflector: bool
    min_improvement: float
    patience: int
    history: List[Dict[str, Any]]
    reflection: str               # 给 planner 的修订指令
    structure_report: str         # 确定性结构检查报告（Markdown）
    structure_issues: List[str]

    # 最优版本
    best_score: float
    best_coa_text: str
    best_iteration: int
    best_reflection: str
    best_structure_report: str
    best_issue_count: int
    no_improve_count: int
    stop_reason: str

    # 控制
    skip_research: bool
    use_analyst: bool
    profile: Optional[str]

    # 最终输出
    final_coa_text: str
    final_coa_obj: Optional[COA]
    parse_success: bool
    validation: Dict[str, Any]
    expected_outcomes: List[Dict[str, Any]]
    outputs: Dict[str, Any]
    token_usage: Dict[str, Any]
    timing: List[Dict[str, Any]]
    node_timing: Dict[str, float]
    stage_log: List[Dict[str, Any]]
    error: Optional[str]


# ── 内部辅助 ──


def _append_stage_log(
    state: COAGraphState, entry: Dict[str, Any]
) -> List[Dict[str, Any]]:
    log = list(state.get("stage_log") or [])
    log.append(entry)
    return log


def _add_timing(state: COAGraphState, stage: str, elapsed: float) -> Dict[str, float]:
    timing = dict(state.get("node_timing") or {})
    timing[stage] = round(timing.get(stage, 0.0) + elapsed, 3)
    return timing


# analysis 里这些字段要么单独传给 planner，要么只对 researcher 有用
_PLANNER_SKIP_KEYS = ("expected_outcomes", "research_queries")


def _compose_planner_knowledge(state: COAGraphState) -> str:
    """合并 analyst + researcher 输出，作为 Planner 的 RAG 上下文。"""
    parts: List[str] = []
    analysis = {
        k: v for k, v in (state.get("mission_analysis") or {}).items()
        if k not in _PLANNER_SKIP_KEYS
    }
    if analysis:
        parts.append(
            "## 任务分析（来自 Analyst）\n```json\n"
            + _json.dumps(analysis, ensure_ascii=False)
            + "\n```"
        )
    brief = state.get("research_brief")
    if brief:
        parts.append(brief)
    return "\n\n".join(parts)


def _rule_only_analysis(mission: str) -> Dict[str, Any]:
    """use_analyst=false 时的零 LLM 分析：只做规则抽取。"""
    return {
        "mission_intent": mission[:120],
        "objectives": [],
        "key_entities": {},
        "constraints": [],
        "uncertainties": [],
        "research_queries": [mission[:200]],
        "expected_outcomes": extract_from_mission(mission),
        "skipped_llm": True,
    }


def _format_instructions(instructions: Sequence[str], feedback: str) -> str:
    """judge 的 revision_instructions → 编号列表；为空时退回 feedback。"""
    items = [str(x).strip() for x in instructions if str(x).strip()]
    if not items:
        return feedback or ""
    return "\n".join(f"{i}. {text}" for i, text in enumerate(items, 1))


def _stop_reason(
    *,
    score: float,
    structure_passed: bool,
    iteration: int,
    max_iterations: int,
    threshold: float,
    early_stop: bool,
    no_improve_count: int,
    patience: int,
) -> str:
    """judge 之后是否结束循环；返回空串表示继续修订。"""
    if early_stop and score >= threshold and structure_passed:
        return "quality_met"
    if iteration >= max_iterations:
        return "max_iterations"
    if early_stop and patience > 0 and no_improve_count >= patience:
        return "plateau"
    return ""


# ── 节点工厂 ──


def _make_analyst_node(analyst: MissionAnalystAgent):
    async def analyst_node(state: COAGraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        mission = state["mission_input"]
        logger.info(f"[Graph/Analyst] mission len={len(mission)}")
        if state.get("use_analyst", True):
            analysis = await analyst.analyze(mission)
        else:
            analysis = _rule_only_analysis(mission)

        # 用户显式给定的条目优先（保留其来源），analyst / 任务文本抽取补充
        outcomes = merge_outcomes(
            state.get("expected_outcomes_input") or [],
            analysis.get("expected_outcomes") or [],
            keep_primary_source=True,
        )
        analysis["expected_outcomes"] = outcomes
        elapsed = time.perf_counter() - started
        log = _append_stage_log(
            state,
            {
                "stage": "analyst",
                "objectives": len(analysis.get("objectives", [])),
                "research_queries": len(analysis.get("research_queries", [])),
                "expected_outcomes": len(outcomes),
                "skipped": not state.get("use_analyst", True),
                "elapsed": round(elapsed, 3),
            },
        )
        return {
            "mission_analysis": analysis,
            "expected_outcomes_input": outcomes,
            "stage_log": log,
            "node_timing": _add_timing(state, "analyst", elapsed),
        }

    return analyst_node


def _make_researcher_node(
    researcher: KnowledgeResearcherAgent, rag_manager: RAGManager
):
    async def researcher_node(state: COAGraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        if state.get("skip_research") or not rag_manager.is_enabled():
            logger.info("[Graph/Researcher] 跳过（skip_research 或 RAG 未启用）")
            elapsed = time.perf_counter() - started
            return {
                "research_brief": researcher._fallback_brief(
                    state.get("mission_analysis") or {},
                    reason="RAG 未启用 / 显式跳过",
                ),
                "stage_log": _append_stage_log(
                    state,
                    {"stage": "researcher", "skipped": True, "elapsed": round(elapsed, 3)},
                ),
                "node_timing": _add_timing(state, "researcher", elapsed),
            }

        analysis = state.get("mission_analysis") or {}
        tools = build_rag_tools(rag_manager=rag_manager)
        logger.info(f"[Graph/Researcher] 工具数={len(tools)}")
        brief = await researcher.research(analysis, tools)
        elapsed = time.perf_counter() - started
        log = _append_stage_log(
            state,
            {
                "stage": "researcher",
                "tools_available": [t.name for t in tools],
                "brief_len": len(brief),
                "elapsed": round(elapsed, 3),
            },
        )
        return {
            "research_brief": brief,
            "stage_log": log,
            "node_timing": _add_timing(state, "researcher", elapsed),
        }

    return researcher_node


def _make_planner_node(planner: PlannerAgent):
    async def planner_node(state: COAGraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        mission = state["mission_input"]
        iteration = state.get("iteration", 0)
        revision_base = state.get("revision_base") or ""
        revising = iteration > 0 and bool(revision_base)
        reflection = state.get("reflection", "") if revising else ""
        structure_report = state.get("structure_report", "") if revising else ""
        knowledge = _compose_planner_knowledge(state)

        logger.info(
            f"[Graph/Planner] iter={iteration}, knowledge_len={len(knowledge)}, "
            f"reflection_len={len(reflection)}, revising={revising}"
        )
        try:
            coa_text = await planner.generate_coa(
                mission,
                knowledge=knowledge,
                reflection=reflection,
                previous_coa_text=revision_base if revising else None,
                expected_outcomes=state.get("expected_outcomes_input") or [],
                structure_report=structure_report,
            )
        except Exception as plan_error:
            if not revising:
                raise
            # 修订轮失败不应丢掉已经评过分的方案：保留 best，直接收尾
            logger.error(f"[Graph/Planner] 修订失败，保留已有最优版本: {plan_error}")
            elapsed = time.perf_counter() - started
            return {
                "current_coa_text": revision_base,
                "stop_reason": "planner_error",
                "stage_log": _append_stage_log(
                    state,
                    {
                        "stage": "planner",
                        "iteration": iteration,
                        "error": f"{type(plan_error).__name__}: {plan_error}",
                        "elapsed": round(elapsed, 3),
                    },
                ),
                "node_timing": _add_timing(state, "planner", elapsed),
            }

        elapsed = time.perf_counter() - started
        log = _append_stage_log(
            state,
            {
                "stage": "planner",
                "iteration": iteration,
                "coa_len": len(coa_text),
                "used_knowledge": bool(knowledge),
                "truncated": bool(getattr(planner, "last_truncated", False)),
                "patched_sections": list(getattr(planner, "last_patched_sections", None) or []),
                "elapsed": round(elapsed, 3),
            },
        )
        return {
            "current_coa_text": coa_text,
            "stage_log": log,
            "node_timing": _add_timing(state, "planner", elapsed),
        }

    return planner_node


def _make_judge_node(judge: JudgeAgent, parser: COATableParser):
    async def judge_node(state: COAGraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        coa_text = state["current_coa_text"]
        mission = state["mission_input"]
        iteration = state.get("iteration", 0) + 1

        # 1) 确定性结构检查：不花 LLM，结果同时喂给 judge 和下一轮 planner
        try:
            parsed: Optional[COA] = parser.parse(coa_text)
        except Exception as parse_error:
            logger.warning(f"[Graph/Judge] 解析失败，按不可解析处理: {parse_error}")
            parsed = None
        report = check_coa(parsed, coa_text)
        structure_md = report.to_markdown()

        # 2) LLM 评估战术合理性，并直接给出修订指令（替代 reflector）
        logger.info(f"[Graph/Judge] iter={iteration}, structure_issues={len(report.issues)}")
        try:
            score, feedback, details = await judge.evaluate(
                coa_text, mission, structure_report=structure_md
            )
        except Exception as eval_error:
            logger.error(f"[Graph/Judge] Evaluation failed: {eval_error}")
            score, feedback, details = 5.0, f"Evaluation failed: {eval_error}", {
                "judge_failed": True
            }
        judge_failed = bool(details.get("judge_failed"))
        verdict = details.get("verdict", "UNKNOWN")
        reflection = _format_instructions(details.get("revision_instructions") or [], feedback)

        # 3) 最优版本追踪 + 平台期计数
        has_best = bool(state.get("best_coa_text"))
        prev_best = float(state.get("best_score", 0.0))
        prev_issue_count = int(state.get("best_issue_count", 0))
        no_improve = int(state.get("no_improve_count", 0))
        min_improvement = float(state.get("min_improvement", 0.0))
        is_best = False
        if not has_best:
            is_best = True
        elif not judge_failed:
            is_best = (score, -len(report.issues)) > (prev_best, -prev_issue_count)
            no_improve = no_improve + 1 if score - prev_best < min_improvement else 0

        update: Dict[str, Any] = {}
        if is_best:
            update.update(
                best_score=score,
                best_coa_text=coa_text,
                best_iteration=iteration,
                best_reflection=reflection,
                best_structure_report=structure_md,
                best_issue_count=len(report.issues),
            )
            revision_base, next_reflection, next_structure = coa_text, reflection, structure_md
        else:
            # 没有改进：回到最优版本上继续改，并提醒避开那次修订引入的问题
            revision_base = state.get("best_coa_text", "")
            next_structure = state.get("best_structure_report", "")
            next_reflection = state.get("best_reflection", "")
            if reflection and not judge_failed:
                next_reflection += (
                    f"\n\n注意：上一次修订后评分 {prev_best} → {score}，没有改进；"
                    f"下列是那次修订后评估指出的问题，本次一并避免：\n{reflection}"
                )

        best_score = update.get("best_score", prev_best)
        stop = _stop_reason(
            score=score,
            structure_passed=report.passed,
            iteration=iteration,
            max_iterations=int(state.get("max_iterations", 3)),
            threshold=float(state.get("quality_threshold", 8.0)),
            early_stop=bool(state.get("early_stop", True)),
            no_improve_count=no_improve,
            patience=int(state.get("patience", 0)),
        )
        logger.info(
            f"[Graph/Judge] iter={iteration} score={score} best={best_score} "
            f"verdict={verdict} stop={stop or '-'}"
        )

        history = list(state.get("history") or [])
        history.append(
            {
                "iteration": iteration,
                "score": score,
                "feedback": feedback,
                "verdict": verdict,
                "structure_issues": list(report.issues),
                "judge_failed": judge_failed,
                "is_best": is_best,
            }
        )
        elapsed = time.perf_counter() - started
        log = _append_stage_log(
            state,
            {
                "stage": "judge",
                "iteration": iteration,
                "score": score,
                "verdict": verdict,
                "best_score": best_score,
                "structure_issues": len(report.issues),
                "stop_reason": stop,
                "elapsed": round(elapsed, 3),
            },
        )
        update.update(
            iteration=iteration,
            history=history,
            reflection=next_reflection,
            structure_report=next_structure,
            structure_issues=list(report.issues),
            revision_base=revision_base,
            no_improve_count=no_improve,
            stop_reason=stop,
            stage_log=log,
            node_timing=_add_timing(state, "judge", elapsed),
        )
        return update

    return judge_node


def _make_reflector_node(reflector: ReflectorAgent):
    async def reflector_node(state: COAGraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        coa_text = state.get("revision_base") or state["current_coa_text"]
        feedback = state.get("reflection", "")
        structure = state.get("structure_report", "")
        if structure:
            feedback = f"{feedback}\n\n【结构检查】\n{structure}"
        iteration = state.get("iteration", 1)

        logger.info(f"[Graph/Reflector] iter={iteration}")
        try:
            reflection = await reflector.reflect(coa_text, feedback, iteration)
        except Exception as reflect_error:
            logger.error(f"[Graph/Reflector] failed: {reflect_error}")
            reflection = state.get("reflection", "")

        elapsed = time.perf_counter() - started
        log = _append_stage_log(
            state,
            {
                "stage": "reflector",
                "iteration": iteration,
                "reflection_len": len(reflection),
                "elapsed": round(elapsed, 3),
            },
        )
        return {
            "reflection": reflection,
            "stage_log": log,
            "node_timing": _add_timing(state, "reflector", elapsed),
        }

    return reflector_node


def _make_finalize_node(
    parser: COATableParser,
    transformer: COATransformer,
    validator: ValidatorAgent,
    token_callback: TokenUsageCallback,
    timing_callback: StageTimingCallback,
):
    async def finalize_node(state: COAGraphState) -> Dict[str, Any]:
        started = time.perf_counter()
        coa_text = state.get("best_coa_text") or state.get("current_coa_text", "")
        mission = state["mission_input"]
        input_outcomes = state.get("expected_outcomes_input") or []

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

        # 预期战果：以方案第八节为主，保证输入侧条目一条不丢（只透传，不参与评分）
        outcomes = merge_outcomes(
            [o.model_dump() for o in coa_obj.expected_outcomes], input_outcomes
        )
        coa_obj.expected_outcomes = [ExpectedOutcome(**o) for o in outcomes]

        # 验证
        validation_data: Dict[str, Any] = {}
        if parse_success:
            try:
                is_valid, v_feedback, v_extras = (
                    await validator.validate_and_extract_matrix(
                        coa_obj, mission, raw_text=coa_text
                    )
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
            # 结构化 Markdown（核心输出，供下游 ScenClaw 等工具直接消费）
            md_text = transformer.coa_to_markdown(
                coa_obj,
                mission_input=mission,
                analysis=state.get("mission_analysis", {}),
                score=state.get("best_score", 0.0),
                iterations=state.get("iteration", 0),
                validation=validation_data,
                raw_coa_table=coa_text,
                best_iteration=state.get("best_iteration", 0),
                stop_reason=state.get("stop_reason", ""),
            )
            outputs = {
                "markdown_format": md_text,
                "json_format": transformer.coa_to_json(coa_obj),
                "yaml_format": transformer.coa_to_yaml(coa_obj),
                "flat_matrix": transformer.coa_to_flat_matrix(coa_obj),
                "condensed_format": transformer.coa_to_condensed_format(coa_obj),
            }

        elapsed = time.perf_counter() - started
        log = _append_stage_log(
            state,
            {
                "stage": "finalize",
                "parse_success": parse_success,
                "is_valid": validation_data.get("is_valid"),
                "expected_outcomes": len(outcomes),
                "best_score": state.get("best_score", 0.0),
                "stop_reason": state.get("stop_reason", ""),
                "elapsed": round(elapsed, 3),
            },
        )

        return {
            "final_coa_text": coa_text,
            "final_coa_obj": coa_obj,
            "parse_success": parse_success,
            "validation": validation_data,
            "expected_outcomes": outcomes,
            "outputs": outputs,
            "token_usage": {
                "by_model": token_callback.summary(),
                "total_tokens": token_callback.total_tokens(),
            },
            "timing": timing_callback.summary(),
            "stage_log": log,
            "node_timing": _add_timing(state, "finalize", elapsed),
        }

    return finalize_node


# ── 条件边 ──


def _after_planner(state: COAGraphState) -> str:
    """planner 之后：修订失败（已设置 stop_reason）直接收尾，否则去 judge。"""
    return "finalize" if state.get("stop_reason") else "judge"


def _should_continue(state: COAGraphState) -> str:
    """judge 之后：有 stop_reason → finalize；否则按 use_reflector 去 reflector / planner。"""
    if state.get("stop_reason"):
        logger.info(f"[Graph/Router] stop_reason={state['stop_reason']} → finalize")
        return "finalize"
    target = "reflector" if state.get("use_reflector") else "planner"
    logger.info(
        f"[Graph/Router] iter {state.get('iteration', 0)}/{state.get('max_iterations', 3)} → {target}"
    )
    return target


# ── 进程级缓存（借鉴 nanobot：Agent / RAG 只构造一次）──

_AGENT_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_RAG_CACHE: Dict[str, Tuple["weakref.ReferenceType[Any] | None", Tuple[Any, ...], RAGManager]] = {}


def _mtime(path: Any) -> int:
    try:
        return resolve_path(path).stat().st_mtime_ns
    except (OSError, TypeError):
        return 0


def _config_signature(cfg: Dict[str, Any]) -> Tuple[int, ...]:
    """agents.yaml / models.yaml / 各 prompt 文件的 mtime，任一变化即重建 Agent。"""
    settings = get_settings()
    paths: List[Any] = [settings.agents_config, settings.models_config]
    for agent_cfg in (cfg.get("agents") or {}).values():
        if isinstance(agent_cfg, dict) and agent_cfg.get("prompt_file"):
            paths.append(agent_cfg["prompt_file"])
    return tuple(_mtime(p) for p in paths)


def get_agents(profile: Optional[str] = None) -> Dict[str, Any]:
    """按 profile 返回缓存的 6 个 Agent；配置或 prompt 文件变更后自动重建。"""
    cfg = load_agents_config(profile=profile)
    name = cfg.get("active_profile")
    key = (name or "", *_config_signature(cfg))
    agents = _AGENT_CACHE.get(key)
    if agents is None:
        agents = {
            "analyst": MissionAnalystAgent(profile=name),
            "researcher": KnowledgeResearcherAgent(profile=name),
            "planner": PlannerAgent(profile=name),
            "judge": JudgeAgent(profile=name),
            "reflector": ReflectorAgent(profile=name),
            "validator": ValidatorAgent(profile=name),
        }
        _AGENT_CACHE[key] = agents
    return agents


def clear_caches() -> None:
    """清空 Agent / RAG 缓存（测试或热更新配置时使用）。"""
    _AGENT_CACHE.clear()
    _RAG_CACHE.clear()


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


def _get_rag_manager(rag_config_path: str, use_rag: bool) -> RAGManager:
    """RAG 引擎初始化很重（嵌入模型 / 索引），同一 event loop 内复用。

    引擎内部的异步资源绑定在创建时的 loop 上，loop 变化或 rag.yaml 改动时重建。
    """
    if not use_rag:
        return _make_disabled_rag_manager(rag_config_path)
    try:
        loop: Any = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    key = str(resolve_path(rag_config_path))
    signature = (_mtime(rag_config_path),)
    cached = _RAG_CACHE.get(key)
    if cached is not None:
        loop_ref, cached_sig, manager = cached
        cached_loop = loop_ref() if loop_ref is not None else None
        if cached_sig == signature and cached_loop is loop:
            return manager
    manager = RAGManager(rag_config_path)
    _RAG_CACHE[key] = (weakref.ref(loop) if loop is not None else None, signature, manager)
    return manager


# ── 图构建 ──


def build_graph(
    *,
    rag_config_path: str | None = None,
    use_rag: bool = True,
    profile: Optional[str] = None,
):
    """构建完整 6-Agent LangGraph，并返回
    ``(compiled_graph, token_callback, timing_callback, rag_manager)``。

    Args:
        rag_config_path: RAG 配置文件路径
        use_rag: False 时强制跳过 researcher 节点（仍构建以保持图结构一致）
        profile: agents.yaml 中的运行档位（fast / balanced / quality）
    """
    if rag_config_path is None:
        rag_config_path = str(get_settings().rag_config)

    agents = get_agents(profile)
    parser = COATableParser()
    transformer = COATransformer()
    token_callback = TokenUsageCallback()
    timing_callback = StageTimingCallback()
    rag_manager = _get_rag_manager(rag_config_path, use_rag)

    graph = StateGraph(COAGraphState)

    graph.add_node("analyst", _make_analyst_node(agents["analyst"]))
    graph.add_node("researcher", _make_researcher_node(agents["researcher"], rag_manager))
    graph.add_node("planner", _make_planner_node(agents["planner"]))
    graph.add_node("judge", _make_judge_node(agents["judge"], parser))
    graph.add_node("reflector", _make_reflector_node(agents["reflector"]))
    graph.add_node(
        "finalize",
        _make_finalize_node(
            parser, transformer, agents["validator"], token_callback, timing_callback
        ),
    )

    graph.set_entry_point("analyst")
    graph.add_edge("analyst", "researcher")
    graph.add_edge("researcher", "planner")
    graph.add_conditional_edges(
        "planner", _after_planner, {"judge": "judge", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "judge",
        _should_continue,
        {"reflector": "reflector", "planner": "planner", "finalize": "finalize"},
    )
    graph.add_edge("reflector", "planner")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info(
        "LangGraph 6-Agent COA orchestrator compiled (use_rag=%s, rag_sources=%s, profile=%s)",
        use_rag,
        rag_manager.list_sources(),
        agents["planner"].profile,
    )

    return compiled, token_callback, timing_callback, rag_manager


# ── 便捷运行函数 ──


def _pick(value: Any, cfg: Dict[str, Any], key: str, default: Any, cast) -> Any:
    return cast(value if value is not None else cfg.get(key, default))


async def run_graph(
    mission_input: str,
    *,
    rag_config_path: str | None = None,
    use_rag: bool = True,
    max_iterations: Optional[int] = None,
    quality_threshold: Optional[float] = None,
    early_stop: Optional[bool] = None,
    extra_callbacks: Optional[List[Any]] = None,
    expected_outcomes: Any = None,
    profile: Optional[str] = None,
) -> COAResult:
    """一键运行完整流水线。

    Args:
        max_iterations / quality_threshold / early_stop: 为 None 时读 agents.yaml
            的 workflow（已应用 profile）。
        expected_outcomes: 用户显式给定的预期战果，str / dict / 列表均可；
            只透传到输出，不参与评分。
        profile: 运行档位，None 时按 MISSIONORCH_PROFILE → workflow.default_profile。

    Returns:
        与原 ``COAOrchestrator.generate`` 兼容的结果字典（多了
        ``mission_analysis`` / ``research_brief`` / ``stage_log`` /
        ``expected_outcomes`` / ``stop_reason`` 等字段）。``coa_table`` 与
        ``final_score`` 对应被采用的最优版本。
    """
    started = time.perf_counter()
    workflow = load_agents_config(profile=profile).get("workflow") or {}

    compiled, token_callback, timing_callback, rag_manager = build_graph(
        rag_config_path=rag_config_path, use_rag=use_rag, profile=profile
    )

    callbacks: List[Any] = [token_callback, timing_callback]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    max_iter = _pick(max_iterations, workflow, "max_iterations", 3, int)
    run_config: RunnableConfig = {  # type: ignore[assignment]
        "callbacks": callbacks,
        # 每轮最多 planner + judge + reflector 三步，外加前后 3 个节点
        "recursion_limit": max(25, 3 * max_iter + 8),
    }

    initial_state: COAGraphState = {
        "mission_input": mission_input,
        "expected_outcomes_input": coerce_outcomes(expected_outcomes, source="user"),
        "current_coa_text": "",
        "revision_base": "",
        "iteration": 0,
        "max_iterations": max_iter,
        "quality_threshold": _pick(quality_threshold, workflow, "quality_threshold", 8.0, float),
        "early_stop": _pick(early_stop, workflow, "early_stop", True, bool),
        "use_reflector": bool(workflow.get("use_reflector", False)),
        "use_analyst": bool(workflow.get("use_analyst", True)),
        "min_improvement": float(workflow.get("min_improvement", 0.0)),
        "patience": int(workflow.get("patience", 0)),
        "best_score": 0.0,
        "best_coa_text": "",
        "best_iteration": 0,
        "no_improve_count": 0,
        "stop_reason": "",
        "history": [],
        "reflection": "",
        "structure_report": "",
        "skip_research": not use_rag,
        "profile": profile,
        "stage_log": [],
        "node_timing": {},
    }

    final_state = await compiled.ainvoke(initial_state, config=run_config)

    history = final_state.get("history") or []
    best_score = float(final_state.get("best_score", 0.0))
    agents = get_agents(profile)
    elapsed = round(time.perf_counter() - started, 3)

    result: COAResult = {
        "coa_table": final_state.get("final_coa_text") or final_state.get("current_coa_text", ""),
        "final_coa": (
            final_state["final_coa_obj"].model_dump(mode="json")
            if final_state.get("final_coa_obj")
            else {}
        ),
        "parse_success": final_state.get("parse_success", False),
        "iterations": final_state.get("iteration", 0),
        "final_score": best_score,
        "best_score": best_score,
        "best_iteration": final_state.get("best_iteration", 0),
        "stop_reason": final_state.get("stop_reason", ""),
        "history": history,
        "validation": final_state.get("validation", {}),
        "expected_outcomes": final_state.get("expected_outcomes", []),
        "input_expected_outcomes": final_state.get("expected_outcomes_input", []),
        "outputs": final_state.get("outputs", {}),
        "markdown": final_state.get("outputs", {}).get("markdown_format", ""),
        "token_usage": final_state.get("token_usage", {}),
        "timing": final_state.get("timing", []),
        "stage_timing": final_state.get("node_timing", {}),
        "elapsed_seconds": elapsed,
        "mission_analysis": final_state.get("mission_analysis", {}),
        "research_brief": final_state.get("research_brief", ""),
        "stage_log": final_state.get("stage_log", []),
        "rag_sources": rag_manager.list_sources(),
        "rag_enabled": rag_manager.is_enabled(),
        "config": {
            "profile": agents["planner"].profile,
            "planner": agents["planner"].model_id,
            "judge": agents["judge"].model_id,
            "analyst": agents["analyst"].model_id,
            "max_iterations": initial_state["max_iterations"],
            "quality_threshold": initial_state["quality_threshold"],
            "early_stop": initial_state["early_stop"],
            "use_reflector": initial_state["use_reflector"],
            "revision_mode": getattr(agents["planner"], "revision_mode", "full"),
        },
    }

    logger.info(
        f"LangGraph COA completed in {elapsed}s: {result['iterations']} iter, "
        f"best={best_score}@iter{result['best_iteration']}, stop={result['stop_reason']}, "
        f"tokens={result.get('token_usage', {}).get('total_tokens', 0)}, "
        f"rag={result['rag_enabled']}({len(result['rag_sources'])} sources)"
    )

    # 持久化为 Markdown 记忆，供后续 RAG 检索
    try:
        save_run_memory(mission_input, result)
    except Exception as mem_err:
        logger.warning(f"[Memory] 保存记忆失败（不影响主流程）: {mem_err}")

    return result
