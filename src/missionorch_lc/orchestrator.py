"""COA Orchestrator —— 规划-评估-反思-验证循环（经典 4-Agent 模式，``--legacy``）。

架构说明：
- 循环中流转的是自然语言 COA 矩阵（str）
- Planner 输出自然语言矩阵 → 结构检查 + Judge 评估 → Reflector 反思 → Planner 修订
- 最终阶段采用历次迭代中得分最高的版本：COATableParser 将矩阵解析为 COA 对象；
  ValidatorAgent 验证并提取仿真矩阵数据；
  COATransformer 输出 Markdown / JSON / YAML / 扁平矩阵 / 压缩格式。
- 预期战果（expected_outcomes）：输入侧 = 用户参数 + 任务文本规则抽取，
  输出侧 = Planner 第八节与输入条目合并；只透传，不参与评分。

可观测性：
- `TokenUsageCallback`：累计各模型 token 消耗
- `StageTimingCallback`：记录各阶段耗时
- 通过 `RunnableConfig` 传递给每个 Agent
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from .agents import JudgeAgent, PlannerAgent, ReflectorAgent, ValidatorAgent
from .agents.base import load_agents_config
from .core.callbacks import StageTimingCallback, TokenUsageCallback
from .core.coa_checks import check_coa
from .core.coa_parser import COATableParser
from .core.coa_transformer import COATransformer
from .core.expected_outcomes import coerce_outcomes, extract_from_mission, merge_outcomes
from .schemas.coa import COA, ExpectedOutcome
from .schemas.result import COAResult

logger = logging.getLogger(__name__)


class COAOrchestrator:
    """规划-评估-反思-验证的多智能体编排器。"""

    def __init__(self, config_path: str | None = None, *, profile: Optional[str] = None) -> None:
        self.planner = PlannerAgent(profile=profile)
        self.judge = JudgeAgent(profile=profile)
        self.reflector = ReflectorAgent(profile=profile)
        self.validator = ValidatorAgent(profile=profile)
        self.parser = COATableParser()
        self.transformer = COATransformer()

        try:
            workflow_cfg = load_agents_config(config_path, profile=profile).get("workflow") or {}
        except Exception as load_error:
            logger.warning(f"Failed to load workflow config: {load_error}; using defaults")
            workflow_cfg = {}

        self.max_iter: int = int(workflow_cfg.get("max_iterations", 3))
        self.threshold: float = float(workflow_cfg.get("quality_threshold", 8.0))
        self.early_stop: bool = bool(workflow_cfg.get("early_stop", True))

        # 可观测性 callbacks
        self.token_callback = TokenUsageCallback()
        self.timing_callback = StageTimingCallback()

        logger.info(
            f"COAOrchestrator initialized: max_iter={self.max_iter}, "
            f"threshold={self.threshold}, early_stop={self.early_stop}"
        )
        logger.info(
            f"Agents: Planner={self.planner.model_id} (RAG={self.planner.use_rag}), "
            f"Judge={self.judge.model_id}, Reflector={self.reflector.model_id}, "
            f"Validator={self.validator.model_id}"
        )

    def _base_config(self) -> RunnableConfig:
        """返回携带所有 callbacks 的基础 RunnableConfig。"""
        return RunnableConfig(
            callbacks=[self.token_callback, self.timing_callback],
        )

    def _structure_report(self, coa_text: str):
        try:
            parsed: Optional[COA] = self.parser.parse(coa_text)
        except Exception:
            parsed = None
        return check_coa(parsed, coa_text)

    async def generate(
        self,
        mission_input: str,
        *,
        config: Optional[RunnableConfig] = None,
        expected_outcomes: Any = None,
    ) -> COAResult:
        """主流程：规划 → 迭代（评估 → 反思 → 规划）→ 解析 → 验证 → 输出。

        Args:
            mission_input: 任务描述文本
            config: 外部传入的 RunnableConfig（如 LangSmith tracing / 额外 callbacks）
            expected_outcomes: 用户显式给定的预期战果（只透传，不参与评分）
        """
        started = time.perf_counter()
        # 合并外部 config 与内部 callbacks
        base = self._base_config()
        if config:
            merged_callbacks = list(base.get("callbacks") or []) + list(
                config.get("callbacks") or []
            )
            run_config: RunnableConfig = {**config, "callbacks": merged_callbacks}  # type: ignore[assignment]
        else:
            run_config = base

        logger.info(f"Starting COA generation (mission len={len(mission_input)})")
        input_outcomes = merge_outcomes(
            coerce_outcomes(expected_outcomes, source="user"),
            extract_from_mission(mission_input),
            keep_primary_source=True,
        )

        iteration = 0
        best_score = 0.0
        best_text = ""
        best_iteration = 0
        stop_reason = ""
        history: List[Dict[str, Any]] = []

        # ── 初始生成 ──
        logger.info("[Iter 0] Initial COA generation")
        current_coa_text = await self.planner.generate_coa(
            mission_input, config=run_config, expected_outcomes=input_outcomes
        )

        # ── 迭代循环 ──
        while iteration < self.max_iter:
            iteration += 1

            report = self._structure_report(current_coa_text)
            structure_md = report.to_markdown()
            logger.info(f"[Iter {iteration}] Judging COA (structure_issues={len(report.issues)}) ...")
            try:
                score, feedback, details = await self.judge.evaluate(
                    current_coa_text, mission_input,
                    structure_report=structure_md, config=run_config,
                )
            except Exception as eval_error:
                logger.error(f"[Iter {iteration}] Evaluation failed: {eval_error}")
                score, feedback, details = 5.0, "Evaluation failed", {"judge_failed": True}

            verdict = details.get("verdict", "UNKNOWN")
            logger.info(f"[Iter {iteration}] Score: {score}/10 | Verdict: {verdict}")

            is_best = not best_text or (score > best_score and not details.get("judge_failed"))
            if is_best:
                best_score, best_text, best_iteration = score, current_coa_text, iteration
            history.append({
                "iteration": iteration,
                "score": score,
                "feedback": feedback,
                "verdict": verdict,
                "structure_issues": list(report.issues),
                "is_best": is_best,
            })

            if self.early_stop and score >= self.threshold and report.passed:
                logger.info(f"Quality threshold met ({score} >= {self.threshold})")
                stop_reason = "quality_met"
                break

            if iteration >= self.max_iter:
                logger.info("Max iterations reached")
                stop_reason = "max_iterations"
                break

            instructions = details.get("revision_instructions") or []
            judge_feedback = feedback
            if instructions:
                judge_feedback += "\n\n修订指令：\n" + "\n".join(
                    f"{i}. {x}" for i, x in enumerate(instructions, 1)
                )
            logger.info(f"[Iter {iteration}] Reflecting ...")
            try:
                reflection = await self.reflector.reflect(
                    best_text, judge_feedback, iteration, config=run_config
                )
            except Exception as reflect_error:
                logger.error(f"[Iter {iteration}] Reflection failed: {reflect_error}")
                reflection = judge_feedback

            logger.info(f"[Iter {iteration}] Regenerating COA ...")
            try:
                current_coa_text = await self.planner.generate_coa(
                    mission_input,
                    reflection=reflection,
                    previous_coa_text=best_text,
                    config=run_config,
                    expected_outcomes=input_outcomes,
                    structure_report=structure_md if is_best else "",
                )
            except Exception as plan_error:
                logger.error(f"[Iter {iteration}] Regeneration failed, keeping best: {plan_error}")
                stop_reason = "planner_error"
                break

        # ── 解析为结构化 COA（采用最优版本）──
        final_text = best_text or current_coa_text
        logger.info(f"[Final] Parsing best COA (iter {best_iteration}) into structured format")
        try:
            final_coa_obj = self.parser.parse(final_text)
            parse_success = True
        except Exception as parse_error:
            logger.error(f"COA table parsing failed: {parse_error}")
            final_coa_obj = COA(
                description="Parse error",
                metadata={"parse_error": str(parse_error)},
            )
            parse_success = False

        outcomes = merge_outcomes(
            [o.model_dump() for o in final_coa_obj.expected_outcomes], input_outcomes
        )
        final_coa_obj.expected_outcomes = [ExpectedOutcome(**o) for o in outcomes]

        # ── 验证并提取仿真矩阵 ──
        validation_data: Dict[str, Any] = {}
        is_valid = False
        if parse_success:
            try:
                is_valid, v_feedback, v_extras = (
                    await self.validator.validate_and_extract_matrix(
                        final_coa_obj, mission_input, raw_text=final_text, config=run_config
                    )
                )
                validation_data = {
                    "is_valid": is_valid,
                    "validation_feedback": v_feedback,
                    **v_extras,
                }
            except Exception as validate_error:
                logger.error(f"Validation failed: {validate_error}")
                validation_data = {
                    "is_valid": False,
                    "validation_feedback": f"Validation error: {validate_error}",
                    "issues_found": [str(validate_error)],
                    "pure_matrix_data": {},
                    "corrected_coa_data": final_coa_obj.model_dump(),
                }

        # ── 构建最终输出 ──
        final_coa_data = final_coa_obj.model_dump(mode="json")

        outputs: Dict[str, Any] = {}
        if parse_success:
            md_text = self.transformer.coa_to_markdown(
                final_coa_obj,
                mission_input=mission_input,
                score=best_score,
                iterations=len(history),
                validation=validation_data,
                raw_coa_table=final_text,
                best_iteration=best_iteration,
                stop_reason=stop_reason,
            )
            outputs = {
                "markdown_format": md_text,
                "json_format": self.transformer.coa_to_json(final_coa_obj),
                "yaml_format": self.transformer.coa_to_yaml(final_coa_obj),
                "flat_matrix": self.transformer.coa_to_flat_matrix(final_coa_obj),
                "condensed_format": self.transformer.coa_to_condensed_format(final_coa_obj),
            }
        else:
            outputs = {
                "json_format": "{}",
                "yaml_format": "",
                "flat_matrix": {},
                "condensed_format": {},
            }

        # ── token 用量统计 ──
        token_summary = self.token_callback.summary()
        total_tokens = self.token_callback.total_tokens()
        timing_summary = self.timing_callback.summary()

        result: COAResult = {
            "coa_table": final_text,
            "final_coa": final_coa_data,
            "parse_success": parse_success,
            "iterations": iteration,
            "final_score": best_score,
            "best_score": best_score,
            "best_iteration": best_iteration,
            "stop_reason": stop_reason,
            "history": history,
            "validation": validation_data,
            "expected_outcomes": outcomes,
            "input_expected_outcomes": input_outcomes,
            "config": {
                "planner": self.planner.model_id,
                "judge": self.judge.model_id,
                "reflector": self.reflector.model_id,
                "validator": self.validator.model_id,
                "rag_enabled": self.planner.use_rag,
                "max_iterations": self.max_iter,
                "quality_threshold": self.threshold,
                "early_stop": self.early_stop,
                "profile": self.planner.profile,
            },
            "outputs": outputs,
            "markdown": outputs.get("markdown_format", ""),
            "token_usage": {
                "by_model": token_summary,
                "total_tokens": total_tokens,
            },
            "timing": timing_summary,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

        logger.info(
            f"COA generation completed: {iteration} iterations, best={best_score}@iter{best_iteration}, "
            f"stop={stop_reason}, parse_success={parse_success}, is_valid={is_valid}, "
            f"total_tokens={total_tokens}"
        )
        return result
