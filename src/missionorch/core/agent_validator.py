import json
import logging
from typing import Tuple, Dict, Any, Optional
from .agent_base import BaseAgent
from ..schemas.coa import COA

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseAgent):
    """验证智能体 - 验证COA格式合理性并提取纯COA矩阵"""
    
    def __init__(self):
        super().__init__('validator')
        self.validation_rules = self.get_config_value('validation_rules', [
            "format_compliance",
            "logical_consistency", 
            "resource_feasibility",
            "timeline_feasibility"
        ])
        logger.info(f"ValidatorAgent initialized with rules: {self.validation_rules}")
    
    async def validate_and_extract_matrix(self, coa: COA, mission_desc: str = "") -> Tuple[bool, str, Dict[str, Any]]:
        """
        验证COA格式合理性并提取纯COA矩阵
        返回: (是否有效, 验证反馈, 提取的矩阵数据)
        """
        try:
            logger.info(f"Validating COA with {len(coa.tasks)} tasks and {len(coa.effects_chain)} effects")
            
            # 构建验证上下文
            context = {
                "coa_json": json.dumps(coa.model_dump(mode='json'), indent=2),
                "mission": mission_desc,
                "validation_rules": self._translate_rules(self.validation_rules)
            }
            
            user_prompt = f"""请验证以下COA的格式合理性，并提取纯COA矩阵数据用于仿真系统。

COA数据:
{context['coa_json']}

验证规则:
""" + '\n'.join([f"- {rule}" for rule in context['validation_rules']]) + """

请输出验证结果和提取的数据，格式如下:
{{\n  "is_valid": true,\n  "validation_feedback": "验证反馈信息，包括发现的问题和改进建议",\n  "issues_found": ["问题1", "问题2"],\n  "corrected_coa": {{...}},  // 修正后的完整COA数据\n  "pure_matrix_data": {{\n    "entities": ["F-35_001", "AWACS_001"],  // 所有参与实体\n    "events": ["ARRIVE_TARGET", "THREAT_DETECTED", "MISSION_COMPLETE"],  // 事件和条件\n    "matrix": {{\n      "F-35_001": {{\n        "ARRIVE_TARGET": ["PATROL_A", "ESCORT_B"],\n        "THREAT_DETECTED": ["INTERCEPT_C"],\n        "MISSION_COMPLETE": ["RTB_D"]\n      }}\n    }},\n    "task_details": {{\n      "PATROL_A": {{\n        "type": "DCA",\n        "description": "防御性制空巡逻",\n        "resources": "2xF-35",\n        "location": "COORDINATES_A",\n        "start_condition": "ARRIVE_TARGET",\n        "end_condition": "THREAT_DETECTED"\n      }}\n    }},\n    "simulation_params": {{\n      "F-35_001": {{\n        "initial_state": "ACTIVE",\n        "initial_position": "BASE_COORDS",\n        "capabilities": ["AIR_TO_AIR", "AIR_TO_GROUND", "STEALTH"]\n      }}\n    }}\n  }}\n}}

特别注意：
1. 确保任务ID唯一且有意义
2. 条件参数格式正确（基于事件和条件，而非固定时间）
3. 资源分配不冲突
4. 依赖关系合理
5. 提取的数据应适合直接用于仿真系统
"""
            
            logger.debug("Sending validation request to model")
            response = await self.generate(user_prompt, context)
            logger.debug(f"Received validation response with length: {len(response) if response else 0}")
            
            try:
                result = json.loads(self._extract_json(response))
                
                is_valid = result.get('is_valid', False)
                feedback = result.get('validation_feedback', '')
                corrected_coa_data = result.get('corrected_coa', coa.model_dump())
                pure_matrix_data = result.get('pure_matrix_data', {})
                
                logger.info(f"Validation completed - Valid: {is_valid}")
                
                # 如果验证失败，尝试修复COA
                if not is_valid:
                    logger.warning("COA validation failed, attempting to fix common issues")
                    corrected_coa = await self._attempt_fix_coa(coa, result.get('issues_found', []))
                    corrected_coa_data = corrected_coa.model_dump()
                
                return is_valid, feedback, {
                    "corrected_coa_data": corrected_coa_data,
                    "pure_matrix_data": pure_matrix_data,
                    "issues_found": result.get('issues_found', []),
                    "validation_feedback": feedback
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode validation JSON: {e}")
                logger.debug(f"Response that failed to decode: {response[:200]}...")
                
                # 如果JSON解析失败，进行基本验证和矩阵提取
                basic_validation = await self._basic_validate(coa)
                basic_matrix = self._extract_basic_matrix(coa)
                
                return basic_validation, "JSON解析失败，使用基本验证结果", {
                    "corrected_coa_data": coa.model_dump(),
                    "pure_matrix_data": basic_matrix,
                    "issues_found": ["JSON解析错误"],
                    "validation_feedback": "由于JSON解析错误，使用基本验证和矩阵提取"
                }
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def _translate_rules(self, rules: list) -> list:
        """将配置中的英文规则翻译为中文提示"""
        translation_map = {
            "format_compliance": "任务依赖关系不能形成循环",
            "logical_consistency": "任务时间安排要合理",
            "resource_feasibility": "资源分配不能冲突",
            "timeline_feasibility": "效果链要逻辑连贯",
            "decision_point_clarity": "决策点要明确",
            "task_description_clarity": "任务描述要清晰"
        }
        
        translated_rules = []
        for rule in rules:
            translated_rules.append(translation_map.get(rule, rule))
        
        # 如果没有匹配的翻译，使用默认规则
        if not translated_rules:
            translated_rules = [
                "任务依赖关系不能形成循环",
                "任务时间安排要合理",
                "资源分配不能冲突",
                "效果链要逻辑连贯",
                "决策点要明确",
                "任务描述要清晰"
            ]
        
        return translated_rules
    
    async def _attempt_fix_coa(self, coa: COA, issues: list) -> COA:
        """尝试修复COA中的常见问题"""
        try:
            logger.debug(f"Attempting to fix COA issues: {issues}")
            
            # 将COA转换为字典进行修改
            coa_dict = coa.model_dump()
            
            # 检查并修复任务ID重复问题
            task_ids = [task.task_id for task in coa.tasks]
            unique_task_ids = set()
            duplicate_found = False
            
            for i, task_id in enumerate(task_ids):
                original_task_id = task_id
                counter = 1
                while task_id in unique_task_ids:
                    task_id = f"{original_task_id}_{counter}"
                    counter += 1
                    duplicate_found = True
                unique_task_ids.add(task_id)
                if duplicate_found and counter > 1:
                    coa_dict['tasks'][i]['task_id'] = task_id
            
            # 确保基本字段存在
            if not coa_dict.get('description'):
                coa_dict['description'] = "Generated COA"
            
            if not coa_dict.get('synchronization_matrix'):
                coa_dict['synchronization_matrix'] = {}
            
            if not coa_dict.get('resource_allocation'):
                coa_dict['resource_allocation'] = {}
            
            # 从修改后的字典重建COA对象
            fixed_coa = COA.model_validate(coa_dict)
            
            logger.info(f"COA fixing attempt completed")
            return fixed_coa
            
        except Exception as e:
            logger.error(f"Attempt to fix COA failed: {e}")
            return coa  # 返回原始COA，让后续处理决定如何处理
    
    async def _basic_validate(self, coa: COA) -> bool:
        """基本验证COA结构"""
        try:
            # 检查基本字段是否存在
            checks = [
                hasattr(coa, 'tasks') and len(coa.tasks) > 0,
                hasattr(coa, 'effects_chain') and len(coa.effects_chain) > 0,
                hasattr(coa, 'description') and len(coa.description) > 0,
                hasattr(coa, 'synchronization_matrix'),
                hasattr(coa, 'resource_allocation')
            ]
            
            valid = all(checks)
            logger.debug(f"Basic validation result: {valid}, checks: {checks}")
            
            return valid
            
        except Exception as e:
            logger.error(f"Basic validation failed: {e}")
            return False
    
    def _extract_basic_matrix(self, coa: COA) -> Dict[str, Any]:
        """提取基本矩阵数据（基于事件和条件）"""
        entities = set(task.assigned_unit for task in coa.tasks)
        
        # 收集所有事件和条件
        events = set()
        for task in coa.tasks:
            if task.start_condition:
                events.add(task.start_condition)
            if task.end_condition:
                events.add(task.end_condition)
            if task.location:
                events.add(f"at_{task.location}")
        
        matrix_data = {
            "entities": list(entities),
            "events": sorted(list(events)) if events else ["START", "END"],
            "matrix": {},
            "task_details": {},
            "simulation_params": {}
        }
        
        # 填充矩阵
        for entity in entities:
            matrix_data["matrix"][entity] = {}
            entity_tasks = [task for task in coa.tasks if task.assigned_unit == entity]
            
            for task in entity_tasks:
                # 使用事件作为矩阵键
                event_keys = []
                if task.start_condition:
                    event_keys.append(task.start_condition)
                if task.end_condition:
                    event_keys.append(task.end_condition)
                if task.location:
                    event_keys.append(f"at_{task.location}")
                
                if not event_keys:
                    event_keys = ["GENERAL"]
                
                for event_key in event_keys:
                    if event_key not in matrix_data["matrix"][entity]:
                        matrix_data["matrix"][entity][event_key] = []
                    matrix_data["matrix"][entity][event_key].append({
                        "task_id": task.task_id,
                        "task_type": task.task_type.value
                    })
                
                # 添加任务详情
                matrix_data["task_details"][task.task_id] = {
                    "type": task.task_type.value,
                    "description": task.description,
                    "resources": f"{task.platform_count} units",
                    "location": task.location or 'UNSPECIFIED',
                    "start_condition": task.start_condition,
                    "end_condition": task.end_condition,
                    "duration_min": task.duration_min,
                    "duration_max": task.duration_max
                }
            
            # 为每个实体添加基础仿真参数
            matrix_data["simulation_params"][entity] = {
                "initial_state": "ACTIVE",
                "initial_position": "BASE_COORDS",
                "capabilities": ["DEFAULT"]
            }
        
        return matrix_data