import asyncio
import logging
from typing import Dict, Any
import yaml
from ..core.agent_impl import PlannerAgent, JudgeAgent, ReflectorAgent
from ..core.coa_parser import COATableParser
from ..core.coa_transformer import COATransformer
from ..schemas.coa import COA

logger = logging.getLogger(__name__)


class COAOrchestrator:
    """
    多智能体编排器 - 规划-评估-反思循环

    架构说明：
    - 整个循环中，Agent 之间传递的都是**自然语言 COA 表格**（str）
    - Planner 输出自然语言 COA 表格
    - Judge 直接评估自然语言表格，输出评分
    - Reflector 基于自然语言表格和反馈输出改进建议
    - 只有在最终输出阶段，才由 COATableParser 将表格解析为 COA 对象，
      再由 COATransformer 转为 JSON/YAML 等结构化格式
    """

    def __init__(self, config_path: str = "config/agents.yaml"):
        self.planner = PlannerAgent()
        self.judge = JudgeAgent()
        self.reflector = ReflectorAgent()
        self.parser = COATableParser()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                workflow_config = cfg.get('workflow', {})

            self.max_iter = workflow_config.get('max_iterations', 3)
            self.threshold = workflow_config.get('quality_threshold', 8.0)
            self.early_stop = workflow_config.get('early_stop', True)
        except Exception as config_error:
            logger.error(f"Failed to load orchestrator configuration: {config_error}")
            self.max_iter = 3
            self.threshold = 8.0
            self.early_stop = True

        logger.info(
            f"COAOrchestrator initialized: max_iter={self.max_iter}, "
            f"threshold={self.threshold}, early_stop={self.early_stop}"
        )
        logger.info(
            f"Agents: Planner={self.planner.model_id} (RAG={self.planner.use_rag}), "
            f"Judge={self.judge.model_id}, Reflector={self.reflector.model_id}"
        )

    async def generate(self, mission_input: str) -> Dict[str, Any]:
        """
        主流程：规划-评估-反思循环。

        内部全程使用自然语言 COA 表格（str），
        只在最终输出时才转换为结构化格式。
        """
        logger.info(f"Starting COA generation for mission (length: {len(mission_input)})")

        iteration = 0
        best_score = 0.0
        history: list[Dict[str, Any]] = []

        # ── 初始生成（返回自然语言 COA 表格） ──
        logger.info("[Iter 0] Initial COA generation (natural language table)...")
        current_coa_text = await self.planner.generate_coa(mission_input)

        # ── 迭代循环：评估 → 反思 → 重新规划 ──
        while iteration < self.max_iter:
            iteration += 1

            # 评估（直接评估自然语言表格）
            logger.info(f"[Iter {iteration}] Evaluating COA table...")
            try:
                score, feedback, details = await self.judge.evaluate(current_coa_text, mission_input)
            except Exception as evaluation_error:
                logger.error(f"[Iter {iteration}] Evaluation failed: {evaluation_error}")
                score, feedback, details = 5.0, "Evaluation failed due to error", {}

            verdict = details.get('verdict', 'UNKNOWN')
            logger.info(f"[Iter {iteration}] Score: {score}/10 | Verdict: {verdict}")

            history.append({
                "iteration": iteration,
                "score": score,
                "feedback": feedback,
                "verdict": verdict,
            })
            best_score = max(best_score, score)

            # 检查终止条件
            if score >= self.threshold:
                logger.info(f"Quality threshold met ({score} >= {self.threshold})")
                break

            if iteration >= self.max_iter:
                logger.info("Max iterations reached")
                break

            # 反思（基于自然语言表格和反馈）
            logger.info(f"[Iter {iteration}] Generating reflection...")
            try:
                reflection = await self.reflector.reflect(current_coa_text, feedback, iteration)
            except Exception as reflection_error:
                logger.error(f"[Iter {iteration}] Reflection failed: {reflection_error}")
                reflection = f"Previous evaluation feedback: {feedback}"

            # 重新规划（传入上一版自然语言表格）
            logger.info(f"[Iter {iteration}] Regenerating COA table based on reflection...")
            current_coa_text = await self.planner.generate_coa(
                mission_input,
                reflection=reflection,
                previous_coa_text=current_coa_text,
            )

        # ── 最终输出：将自然语言表格转换为结构化格式 ──
        logger.info("[Final] Parsing COA matrix into structured format...")
        try:
            final_coa_obj = self.parser.parse(current_coa_text)
            parse_success = True
            logger.info(
                f"Parsed COA: {len(final_coa_obj.phases)} phases, "
                f"{len(final_coa_obj.units)} units, "
                f"{len(final_coa_obj.matrix)} matrix cells, "
                f"{len(final_coa_obj.effects_chain)} effects"
            )
        except Exception as parse_error:
            logger.error(f"COA table parsing failed: {parse_error}")
            final_coa_obj = COA(
                description="Parse error - could not convert natural language matrix to structured COA",
                metadata={"parse_error": str(parse_error)},
            )
            parse_success = False

        # 构建多格式输出
        transformer = COATransformer()
        final_coa_data = final_coa_obj.model_dump(mode='json')
        final_score = history[-1]["score"] if history else 0.0

        result = {
            # 自然语言 COA 表格（系统的主要输出）
            "coa_table": current_coa_text,
            # 结构化数据（由解析器从表格转换而来）
            "final_coa": final_coa_data,
            "parse_success": parse_success,
            # 迭代信息
            "iterations": iteration,
            "final_score": final_score,
            "best_score": best_score,
            "history": history,
            # 配置信息
            "config": {
                "planner": self.planner.model_id,
                "judge": self.judge.model_id,
                "reflector": self.reflector.model_id,
                "rag_enabled": self.planner.use_rag,
                "max_iterations": self.max_iter,
                "quality_threshold": self.threshold,
                "early_stop": self.early_stop,
            },
            # 多格式输出（仅在解析成功时有效）
            "outputs": {
                "json_format": transformer.coa_to_json(final_coa_obj) if parse_success else "{}",
                "yaml_format": transformer.coa_to_yaml(final_coa_obj) if parse_success else "",
                "flat_matrix": transformer.coa_to_flat_matrix(final_coa_obj) if parse_success else {},
                "condensed_format": transformer.coa_to_condensed_format(final_coa_obj) if parse_success else {},
            },
        }

        logger.info(
            f"COA generation completed: {iteration} iterations, "
            f"final_score={final_score}, parse_success={parse_success}"
        )
        return result