import asyncio
import json
import logging
from typing import Tuple, Dict, Optional
from .agent_base import BaseAgent
from ..schemas.coa import COA, Task, Effect, DecisionPoint

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """规划Agent - 生成COA"""
    
    def __init__(self):
        super().__init__('planner')
        logger.info("PlannerAgent initialized")
    
    async def generate_coa(self, mission_desc: str, 
                          knowledge: str = "", 
                          reflection: str = "",
                          previous_coa: Optional[COA] = None) -> COA:
        """生成COA方案"""
        try:
            logger.info(f"Starting COA generation for mission (length: {len(mission_desc)})")
            
            # RAG检索（如果启用）
            rag_context = knowledge
            if self.use_rag and not knowledge:
                logger.debug("Retrieving RAG context for planning")
                rag_context = await self.retrieve_context(mission_desc)
            
            context = {
                "rag_context": rag_context,
                "reflection": reflection,
                "previous_coa": json.dumps(previous_coa.model_dump(mode='json'), indent=2) if previous_coa else "",
                "domain": "air_combat"
            }
            
            user_prompt = f"""基于以下任务描述生成COA：
            
【任务描述】
{mission_desc}

【要求】
1. 战役级抽象：只定义What/Who/When，不涉及How（具体高度、速度、机动）
2. 必须有明确的效果链（Effects Chain）
3. 任务依赖关系使用DAG（无循环）
4. 包含决策点（Decision Points）
5. 符合空战任务规划规范（SEAD/OCA/DCA等）
6. 确保任务包含时间参数和资源分配信息

输出严格JSON格式，符合COA Schema。"""

            logger.debug("Sending generation request to model")
            response = await self.generate(user_prompt, context)
            logger.debug(f"Received response with length: {len(response) if response else 0}")
            
            # 解析JSON
            json_str = self._extract_json(response)
            logger.debug(f"Parsing COA from JSON (length: {len(json_str) if json_str else 0})")
            
            try:
                coa_data = json.loads(json_str)
                coa = COA.model_validate(coa_data)
                logger.info("Successfully parsed COA from response")
                
                # 验证COA的有效性
                if not coa.tasks:
                    logger.warning("Generated COA has no tasks")
                
                if not coa.effects_chain:
                    logger.warning("Generated COA has no effects chain")
                
                return coa
            except Exception as e:
                logger.error(f"Failed to parse COA JSON: {e}")
                logger.debug(f"JSON string that failed to parse: {json_str[:200]}...")
                # 返回最小COA
                return COA(
                    description="Parse error",
                    effects_chain=[],
                    tasks=[],
                    critical_path=[],
                    synchronization_matrix={},
                    resource_allocation={},
                    metadata={}
                )
        except Exception as e:
            logger.error(f"COA generation failed: {e}")
            raise


class JudgeAgent(BaseAgent):
    """评估Agent - LLM as Judge"""
    
    def __init__(self):
        super().__init__('judge')
        self.criteria = self.agent_cfg.get('criteria', [])
        logger.info(f"JudgeAgent initialized with criteria: {self.criteria}")
    
    async def evaluate(self, coa: COA, mission_desc: str) -> Tuple[float, str, Dict]:
        """评估COA，返回(分数, 反馈, 详情)"""
        try:
            logger.info(f"Evaluating COA with {len(coa.tasks)} tasks and {len(coa.effects_chain)} effects")
            
            context = {
                "coa_json": json.dumps(coa.model_dump(mode='json'), indent=2),
                "mission": mission_desc,
                "criteria": ', '.join(self.criteria)
            }
            
            user_prompt = f"""请评估上述COA方案。

按以下维度评分（0-10分）：
""" + '\n'.join([f"- {c}" for c in self.criteria]) + """

特别关注：
1. 任务时间安排的合理性
2. 资源分配的均衡性
3. 任务依赖关系的正确性
4. 决策点设置的恰当性

输出JSON格式：
{{\n  "overall_score": 8.5,\n  "dimension_scores": {{"feasibility": 9, "completeness": 8, ...}},\n  "verdict": "ACCEPT" | "REVISE" | "REJECT",\n  "feedback": "详细评价...",\n  "critical_issues": ["问题1", "问题2"],\n  "improvement_suggestions": ["建议1", "建议2"]\n}}"""

            logger.debug("Sending evaluation request to model")
            response = await self.generate(user_prompt, context)
            logger.debug(f"Received evaluation response with length: {len(response) if response else 0}")
            
            try:
                result = json.loads(self._extract_json(response))
                score = result.get('overall_score', 5.0)
                feedback = result.get('feedback', '')
                logger.info(f"Completed evaluation with score: {score}")
                return score, feedback, result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode evaluation JSON: {e}")
                logger.debug(f"Response that failed to decode: {response[:200]}...")
                return 5.0, "Evaluation parse error", {}
            except Exception as e:
                logger.error(f"Unexpected error during evaluation parsing: {e}")
                return 5.0, "Evaluation parse error", {}
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


class ReflectorAgent(BaseAgent):
    """反思Agent - 缺陷分析"""
    
    def __init__(self):
        super().__init__('reflector')
        self.use_reasoning = self.agent_cfg.get('use_reasoning_chain', False)
        logger.info(f"ReflectorAgent initialized with reasoning chain: {self.use_reasoning}")
    
    async def reflect(self, coa: COA, feedback: str, iteration: int) -> str:
        """生成反思报告"""
        try:
            logger.info(f"Starting reflection for iteration {iteration}, COA with {len(coa.tasks)} tasks")
            
            context = {
                "coa": json.dumps(coa.model_dump(mode='json'), indent=2),
                "judge_feedback": feedback,
                "iteration": iteration
            }
            
            user_prompt = f"""请对当前COA进行深度反思分析。

重点关注：
1. 关键逻辑缺陷（循环依赖、资源冲突）
2. 时序可行性问题
3. 风险覆盖盲区
4. 与历史战例的差距
5. 如何改进任务时间安排和资源分配

输出结构化的修改建议（可直接用于重新生成）。"""

            logger.debug("Sending reflection request to model")
            temp = 0.6 if self.use_reasoning else None
            reflection_result = await self.generate(user_prompt, context, temp_override=temp)
            
            logger.info(f"Completed reflection with result length: {len(reflection_result) if reflection_result else 0}")
            return reflection_result
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            raise