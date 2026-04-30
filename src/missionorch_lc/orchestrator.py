"""COA Orchestrator —— 规划-评估-反思-验证循环。

架构说明：
- 循环中流转的是自然语言 COA 矩阵（str）
- Planner 输出自然语言矩阵 → Judge 评估 → Reflector 反思 → Planner 重新生成
- 最终阶段：COATableParser 将矩阵解析为 COA 对象；
  ValidatorAgent 验证并提取仿真矩阵数据；
  COATransformer 输出 JSON / YAML / 扁平矩阵 / 压缩格式。

可观测性：
- `TokenUsageCallback`：累计各模型 token 消耗
- `StageTimingCallback`：记录各阶段耗时
- 通过 `RunnableConfig` 传递给每个 Agent
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.runnables import RunnableConfig

from .agents import JudgeAgent, PlannerAgent, ReflectorAgent, ValidatorAgent
from .core.callbacks import StageTimingCallback, TokenUsageCallback
from .core.coa_parser import COATableParser
from .core.coa_transformer import COATransformer
from .schemas.coa import COA

logger = logging.getLogger(__name__)


class COAOrchestrator:
    """规划-评估-反思-验证的多智能体编排器。"""

    def __init__(self, config_path: str = "config/agents.yaml") -> None:
        self.planner = PlannerAgent()
        self.judge = JudgeAgent()
        self.reflector = ReflectorAgent()
        self.validator = ValidatorAgent()
        self.parser = COATableParser()
        self.transformer = COATransformer()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                workflow_cfg = (yaml.safe_load(f) or {}).get("workflow", {})
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

    async def generate(
        self,
        mission_input: str,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """主流程：规划 → 迭代（评估 → 反思 → 规划）→ 解析 → 验证 → 输出。

        Args:
            mission_input: 任务描述文本
            config: 外部传入的 RunnableConfig（如 LangSmith tracing / 额外 callbacks）
        """
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

        iteration = 0
        best_score = 0.0
        history: List[Dict[str, Any]] = []

        # ── 初始生成 ──
        logger.info("[Iter 0] Initial COA generation")
        current_coa_text = await self.planner.generate_coa(
            mission_input, config=run_config
        )

        # ── 迭代循环 ──
        while iteration < self.max_iter:
            iteration += 1

            logger.info(f"[Iter {iteration}] Judging COA ...")
            try:
                score, feedback, details = await self.judge.evaluate(
                    current_coa_text, mission_input, config=run_config
                )
            except Exception as eval_error:
                logger.error(f"[Iter {iteration}] Evaluation failed: {eval_error}")
                score, feedback, details = 5.0, "Evaluation failed", {}

            verdict = details.get("verdict", "UNKNOWN")
            logger.info(f"[Iter {iteration}] Score: {score}/10 | Verdict: {verdict}")

            history.append({
                "iteration": iteration,
                "score": score,
                "feedback": feedback,
                "verdict": verdict,
            })
            best_score = max(best_score, score)

            if self.early_stop and score >= self.threshold:
                logger.info(f"Quality threshold met ({score} >= {self.threshold})")
                break

            if iteration >= self.max_iter:
                logger.info("Max iterations reached")
                break

            logger.info(f"[Iter {iteration}] Reflecting ...")
            try:
                reflection = await self.reflector.reflect(
                    current_coa_text, feedback, iteration, config=run_config
                )
            except Exception as reflect_error:
                logger.error(f"[Iter {iteration}] Reflection failed: {reflect_error}")
                reflection = f"Previous evaluation feedback: {feedback}"

            logger.info(f"[Iter {iteration}] Regenerating COA ...")
            current_coa_text = await self.planner.generate_coa(
                mission_input,
                reflection=reflection,
                previous_coa_text=current_coa_text,
                config=run_config,
            )

        # ── 解析为结构化 COA ──
        logger.info("[Final] Parsing COA matrix into structured format")
        try:
            final_coa_obj = self.parser.parse(current_coa_text)
            parse_success = True
        except Exception as parse_error:
            logger.error(f"COA table parsing failed: {parse_error}")
            final_coa_obj = COA(
                description="Parse error",
                metadata={"parse_error": str(parse_error)},
            )
            parse_success = False

        # ── 验证并提取仿真矩阵 ──
        validation_data: Dict[str, Any] = {}
        is_valid = False
        if parse_success:
            try:
                is_valid, v_feedback, v_extras = (
                    await self.validator.validate_and_extract_matrix(
                        final_coa_obj, mission_input, config=run_config
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
        final_score = history[-1]["score"] if history else 0.0

        outputs = {
            "json_format": (
                self.transformer.coa_to_json(final_coa_obj) if parse_success else "{}"
            ),
            "yaml_format": (
                self.transformer.coa_to_yaml(final_coa_obj) if parse_success else ""
            ),
            "flat_matrix": (
                self.transformer.coa_to_flat_matrix(final_coa_obj) if parse_success else {}
            ),
            "condensed_format": (
                self.transformer.coa_to_condensed_format(final_coa_obj)
                if parse_success
                else {}
            ),
        }

        # ── token 用量统计 ──
        token_summary = self.token_callback.summary()
        total_tokens = self.token_callback.total_tokens()
        timing_summary = self.timing_callback.summary()

        result: Dict[str, Any] = {
            "coa_table": current_coa_text,
            "final_coa": final_coa_data,
            "parse_success": parse_success,
            "iterations": iteration,
            "final_score": final_score,
            "best_score": best_score,
            "history": history,
            "validation": validation_data,
            "config": {
                "planner": self.planner.model_id,
                "judge": self.judge.model_id,
                "reflector": self.reflector.model_id,
                "validator": self.validator.model_id,
                "rag_enabled": self.planner.use_rag,
                "max_iterations": self.max_iter,
                "quality_threshold": self.threshold,
                "early_stop": self.early_stop,
            },
            "outputs": outputs,
            "token_usage": {
                "by_model": token_summary,
                "total_tokens": total_tokens,
            },
            "timing": timing_summary,
        }

        logger.info(
            f"COA generation completed: {iteration} iterations, final_score={final_score}, "
            f"parse_success={parse_success}, is_valid={is_valid}, "
            f"total_tokens={total_tokens}"
        )
        return result