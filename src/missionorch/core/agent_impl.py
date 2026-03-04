import asyncio
import json
import logging
from typing import Tuple, Dict, Optional
from .agent_base import BaseAgent

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """规划Agent - 生成自然语言COA表格"""

    def __init__(self):
        super().__init__('planner')
        logger.info("PlannerAgent initialized")

    async def generate_coa(self, mission_desc: str,
                           knowledge: str = "",
                           reflection: str = "",
                           previous_coa_text: Optional[str] = None) -> str:
        """
        生成COA方案，返回自然语言COA表格字符串。

        整个多智能体循环中传递的都是自然语言表格，
        只有在最终输出阶段才由解析器转换为结构化格式。
        """
        try:
            logger.info(f"Starting COA generation for mission (length: {len(mission_desc)})")

            rag_context = knowledge
            if self.use_rag and not knowledge:
                logger.debug("Retrieving RAG context for planning")
                rag_context = await self.retrieve_context(mission_desc)

            context = {
                "rag_context": rag_context,
            }

            reflection_section = ""
            if reflection:
                reflection_section = f"\n【改进建议（请根据以下反馈改进）】\n{reflection}\n"

            previous_coa_section = ""
            if previous_coa_text:
                previous_coa_section = f"\n【上一版COA表格（请在此基础上改进）】\n{previous_coa_text}\n"

            user_prompt = f"""基于以下任务描述生成COA（行动方案）矩阵：

【任务描述】
{mission_desc}
{reflection_section}{previous_coa_section}
【要求】
1. 战役级抽象：定义Who/What/When/Where，不涉及How（具体高度、速度、机动）
2. 以"作战单元 × 阶段"矩阵为核心输出
3. 至少3个作战单元（纵轴）、至少3个阶段（横轴）
4. 每个单元格至少2个行动描述
5. 阶段转换必须有明确的触发条件（条件/事件驱动，禁止使用固定时间窗口）
6. 效果链至少2个效果
7. 包含决策点和关键风险

请严格按照系统提示词中的COA矩阵表格模板格式输出，使用Markdown表格，不要输出JSON。"""

            logger.debug("Sending generation request to model")
            response = await self.generate(user_prompt, context)
            logger.info(f"Received COA table response: {len(response) if response else 0} chars")

            if not response or len(response.strip()) < 50:
                logger.error("LLM returned empty or too short response for COA generation")
                raise ValueError("COA generation returned empty response")

            return response

        except Exception as e:
            logger.error(f"COA generation failed: {e}")
            raise


class JudgeAgent(BaseAgent):
    """评估Agent - LLM as Judge，直接评估自然语言COA表格"""

    def __init__(self):
        super().__init__('judge')
        self.criteria = self.agent_cfg.get('criteria', [])
        logger.info(f"JudgeAgent initialized with criteria: {self.criteria}")

    async def evaluate(self, coa_text: str, mission_desc: str) -> Tuple[float, str, Dict]:
        """
        评估自然语言COA表格，返回(分数, 反馈, 详情)。

        Args:
            coa_text: 自然语言COA表格字符串
            mission_desc: 原始任务描述
        """
        try:
            logger.info(f"Evaluating COA table ({len(coa_text)} chars)")

            context = {
                "coa_table": coa_text,
                "mission": mission_desc,
                "criteria": ', '.join(self.criteria)
            }

            user_prompt = f"""请评估以下COA矩阵方案。

按以下维度评分（0-10分）：
""" + '\n'.join([f"- {c}" for c in self.criteria]) + """

特别关注：
1. 矩阵完整性：每个单元在每个阶段是否都有合理行动
2. 协同性：各单元在同一阶段的行动是否协调
3. 阶段递进：各阶段转换条件和目标是否合理递进
4. 效果达成：行动是否能有效达成战略效果
5. 决策点和风险覆盖

输出JSON格式：
{{
  "overall_score": 8.5,
  "dimension_scores": {{"feasibility": 9, "completeness": 8, "synchronization": 8, "phase_progression": 8, "effect_achievement": 8}},
  "verdict": "ACCEPT",
  "feedback": "详细评价...",
  "critical_issues": ["问题1", "问题2"],
  "improvement_suggestions": ["建议1", "建议2"]
}}"""

            logger.debug("Sending evaluation request to model")
            response = await self.generate(user_prompt, context)
            logger.debug(f"Received evaluation response: {len(response) if response else 0} chars")

            try:
                result = json.loads(self._extract_json(response))
                score = result.get('overall_score', 5.0)
                feedback = result.get('feedback', '')
                logger.info(f"Completed evaluation with score: {score}")
                return score, feedback, result
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode evaluation JSON: {json_err}")
                logger.debug(f"Response preview: {response[:300]}...")
                return 5.0, response, {"verdict": "REVISE", "feedback": response}
            except Exception as parse_err:
                logger.error(f"Unexpected error during evaluation parsing: {parse_err}")
                return 5.0, "Evaluation parse error", {}

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


class ReflectorAgent(BaseAgent):
    """反思Agent - 基于自然语言COA表格进行缺陷分析"""

    def __init__(self):
        super().__init__('reflector')
        self.use_reasoning = self.agent_cfg.get('use_reasoning_chain', False)
        logger.info(f"ReflectorAgent initialized with reasoning chain: {self.use_reasoning}")

    async def reflect(self, coa_text: str, feedback: str, iteration: int) -> str:
        """
        基于自然语言COA表格生成反思报告。

        Args:
            coa_text: 自然语言COA表格字符串
            feedback: Judge的评估反馈
            iteration: 当前迭代次数
        """
        try:
            logger.info(f"Starting reflection for iteration {iteration}")

            context = {
                "coa_table": coa_text,
                "judge_feedback": feedback,
                "iteration": str(iteration)
            }

            user_prompt = f"""请对当前COA矩阵进行深度反思分析。

重点关注：
1. 各单元在各阶段的行动是否存在冲突或遗漏
2. 阶段间的衔接和转换条件是否合理
3. 风险覆盖盲区
4. 效果链是否完整覆盖作战目标
5. 如何改进各单元的行动安排

输出结构化的修改建议，说明具体哪个单元在哪个阶段的行动需要修改，以及如何修改。
建议应该足够具体，可以直接用于重新生成改进后的COA矩阵。"""

            logger.debug("Sending reflection request to model")
            temp = 0.6 if self.use_reasoning else None
            reflection_result = await self.generate(user_prompt, context, temp_override=temp)

            logger.info(f"Completed reflection: {len(reflection_result) if reflection_result else 0} chars")
            return reflection_result

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            raise