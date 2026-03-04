import asyncio
import logging
from typing import Dict, Any
import yaml
from ..core.agent_impl import PlannerAgent, JudgeAgent, ReflectorAgent
from ..core.agent_validator import ValidatorAgent
from ..schemas.coa import COA
from ..core.coa_transformer import COATransformer

logger = logging.getLogger(__name__)


class COAOrchestrator:
    """多智能体编排器 - 规划-评估-反思-验证循环（基于事件和条件）"""
    
    def __init__(self, config_path: str = "config/agents.yaml"):
        self.planner = PlannerAgent()
        self.judge = JudgeAgent()
        self.reflector = ReflectorAgent()
        self.validator = ValidatorAgent()
        
        # 加载工作流配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                wf = cfg.get('workflow', {})
            
            self.max_iter = wf.get('max_iterations', 3)
            self.threshold = wf.get('quality_threshold', 8.0)
            self.early_stop = wf.get('early_stop', True)
            
            logger.info(
                f"COAOrchestrator initialized with max_iter={self.max_iter}, "
                f"threshold={self.threshold}, early_stop={self.early_stop}"
            )
            logger.info(
                f"Agents loaded - Planner: {self.planner.model_id} (RAG: {self.planner.use_rag}), "
                f"Judge: {self.judge.model_id}, Reflector: {self.reflector.model_id}, "
                f"Validator: {self.validator.model_id}"
            )
        except Exception as e:
            logger.error(f"Failed to load orchestrator configuration: {e}")
            # 设置默认值
            self.max_iter = 3
            self.threshold = 8.0
            self.early_stop = True
    
    async def generate(self, mission_input: str) -> Dict[str, Any]:
        """主流程：规划-评估-反思-验证循环（基于事件和条件）"""
        try:
            logger.info(f"Starting COA generation for mission (length: {len(mission_input)})")
            
            iteration = 0
            current_coa: COA = None
            history = []
            
            # 初始生成
            logger.info(f"[Iter {iteration}] Initial COA generation...")
            current_coa = await self.planner.generate_coa(mission_input)
            
            # 开始迭代循环
            while iteration < self.max_iter:
                iteration += 1
                
                # 评估
                logger.info(f"[Iter {iteration}] Evaluating COA...")
                score, feedback, details = await self.judge.evaluate(current_coa, mission_input)
                verdict = details.get('verdict', 'UNKNOWN')
                logger.info(f"[Iter {iteration}] Evaluation completed - Score: {score}/10 | Verdict: {verdict}")
                
                history.append({
                    "iteration": iteration,
                    "score": score,
                    "feedback": feedback,
                    "verdict": verdict
                })
                
                # 检查终止条件
                if score >= self.threshold:
                    logger.info(f"Quality threshold met ({score} >= {self.threshold})")
                    break
                
                if iteration >= self.max_iter:
                    logger.info("Max iterations reached")
                    break
                
                # 反思
                logger.info(f"[Iter {iteration}] Generating reflection...")
                reflection = await self.reflector.reflect(current_coa, feedback, iteration)
                
                # 重新规划
                logger.info(f"[Iter {iteration}] Regenerating COA based on reflection...")
                current_coa = await self.planner.generate_coa(
                    mission_input,
                    reflection=reflection,
                    previous_coa=current_coa
                )
            
            # 验证最终COA
            logger.info("[Final] Validating COA format and extracting matrix...")
            is_valid, validation_feedback, validation_result = await self.validator.validate_and_extract_matrix(
                current_coa, mission_input
            )
            
            # 使用COA转换器生成多种格式
            transformer = COATransformer()
            
            # 根据验证结果选择使用哪个COA数据
            final_coa_data = validation_result["corrected_coa_data"] if is_valid else current_coa.model_dump()
            final_coa_obj = COA.model_validate(final_coa_data)
            
            result = {
                "final_coa": final_coa_data,
                "iterations": iteration,
                "final_score": score if 'score' in locals() else 0,
                "history": history,
                "validation": {
                    "is_valid": is_valid,
                    "feedback": validation_feedback,
                    "issues_found": validation_result.get("issues_found", []),
                    "pure_matrix_data": validation_result.get("pure_matrix_data", {}),
                    "simulation_ready_format": transformer.convert_to_simulation_friendly_format(
                        validation_result.get("pure_matrix_data", {})
                    )
                },
                "config": {
                    "planner": self.planner.model_id,
                    "judge": self.judge.model_id,
                    "reflector": self.reflector.model_id,
                    "validator": self.validator.model_id,
                    "rag_enabled": self.planner.use_rag,
                    "max_iterations": self.max_iter,
                    "quality_threshold": self.threshold,
                    "early_stop": self.early_stop
                },
                # 新增：多种输出格式
                "outputs": {
                    "json_format": transformer.coa_to_json(final_coa_obj),
                    "yaml_format": transformer.coa_to_yaml(final_coa_obj),
                    "simulation_matrix": transformer.coa_to_simulation_matrix(final_coa_obj),
                    "matrix_table": transformer.validate_and_format_for_simulation(
                        validation_result.get("pure_matrix_data", {})
                    ),  # 使用验证智能体提取的纯矩阵数据并验证格式
                    "condensed_format": transformer.coa_to_condensed_format(final_coa_obj)
                }
            }
            
            logger.info(f"COA generation completed in {iteration} iterations with final score {result['final_score']}")
            logger.info(f"COA validation result: {'Passed' if is_valid else 'Failed'}")
            return result
        except Exception as e:
            logger.error(f"COA generation failed: {e}")
            raise