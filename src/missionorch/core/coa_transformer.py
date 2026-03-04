"""
COA转换模块 - 将COA转换为不同格式
"""

from typing import Dict, Any, List
from ..schemas.coa import COA, SimulationMatrixEntry
import json
import yaml


class COATransformer:
    """COA转换器"""
    
    @staticmethod
    def coa_to_json(coa: COA) -> str:
        """将COA转换为JSON格式"""
        return json.dumps(coa.model_dump(mode='json'), indent=2)
    
    @staticmethod
    def coa_to_yaml(coa: COA) -> str:
        """将COA转换为YAML格式"""
        coa_dict = coa.model_dump()
        return yaml.dump(coa_dict, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def coa_to_simulation_matrix(coa: COA) -> List[Dict[str, Any]]:
        """将COA转换为仿真矩阵格式"""
        matrix_entries = coa.to_simulation_matrix()
        return [entry.model_dump() for entry in matrix_entries]
    
    @staticmethod
    def coa_to_matrix_table(coa: COA) -> Dict[str, Any]:
        """将COA转换为矩阵表格格式（基于事件和条件）"""
        # 创建实体-事件矩阵
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
        
        # 如果没有条件信息，则按任务顺序创建事件切片
        if not events:
            events = [f"task_start_{task.task_id}" for task in coa.tasks]
        
        matrix_data = {
            "entities": list(entities),
            "events": sorted(list(events)),
            "matrix": {},
            "legend": {
                "task_types": {task_type.value: task_type.name for task_type in coa.__fields__["tasks"].type_.__args__[0].__fields__["task_type"].type_}
            }
        }
        
        # 填充矩阵
        for entity in entities:
            matrix_data["matrix"][entity] = {}
            for event in events:
                # 找到该实体在该事件条件下涉及的任务
                entity_tasks = [task for task in coa.tasks 
                              if task.assigned_unit == entity and 
                                 (event == task.start_condition or 
                                  event == task.end_condition or 
                                  (task.location and f"at_{task.location}" == event))]
                
                if entity_tasks:
                    matrix_data["matrix"][entity][event] = [
                        {"task_id": task.task_id, "task_type": task.task_type.value, "location": task.location}
                        for task in entity_tasks
                    ]
                else:
                    matrix_data["matrix"][entity][event] = []
        
        return matrix_data
    
    @staticmethod
    def coa_to_condensed_format(coa: COA) -> Dict[str, Any]:
        """将COA转换为压缩格式，便于传输"""
        condensed = {
            "coa_id": coa.coa_id,
            "summary": coa.description,
            "key_metrics": {
                "total_tasks": len(coa.tasks),
                "total_effects": len(coa.effects_chain),
                "total_resources": len(coa.resource_allocation),
                "critical_path_length": len(coa.critical_path)
            },
            "timeline": {
                "phases": COATransformer._extract_timeline_phases(coa)
            },
            "resource_summary": COATransformer._summarize_resources(coa),
            "risk_summary": COATransformer._summarize_risks(coa)
        }
        
        return condensed
    
    @staticmethod
    def _extract_timeline_phases(coa: COA) -> List[Dict[str, Any]]:
        """提取时间线阶段"""
        phases = []
        
        # 按时间对任务进行分组
        time_groups = {}
        for task in coa.tasks:
            time_key = task.time_on_target or "UNDEFINED_TIME"
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(task)
        
        # 为每个时间组创建阶段
        for time_key, tasks in time_groups.items():
            phase = {
                "time_period": time_key,
                "tasks": [
                    {
                        "id": task.task_id,
                        "type": task.task_type.value,
                        "description": task.description,
                        "unit": task.assigned_unit,
                        "platforms": task.platform_count
                    } for task in tasks
                ]
            }
            phases.append(phase)
        
        return sorted(phases, key=lambda x: x["time_period"])
    
    @staticmethod
    def _summarize_resources(coa: COA) -> Dict[str, Any]:
        """汇总资源分配"""
        resource_summary = {
            "allocation": coa.resource_allocation,
            "utilization": {},
            "types": {}
        }
        
        # 计算资源利用率
        total_tasks = len(coa.tasks)
        for unit, assigned_tasks in coa.resource_allocation.items():
            resource_summary["utilization"][unit] = len(assigned_tasks) / max(total_tasks, 1) if total_tasks > 0 else 0
        
        # 按类型统计
        for task in coa.tasks:
            task_type = task.task_type.value
            if task_type not in resource_summary["types"]:
                resource_summary["types"][task_type] = {"count": 0, "units": []}
            resource_summary["types"][task_type]["count"] += 1
            if task.assigned_unit not in resource_summary["types"][task_type]["units"]:
                resource_summary["types"][task_type]["units"].append(task.assigned_unit)
        
        return resource_summary
    
    @staticmethod
    def _summarize_risks(coa: COA) -> Dict[str, Any]:
        """汇总风险信息"""
        return {
            "total_risks": len(coa.critical_risks),
            "risk_categories": COATransformer._categorize_risks(coa.critical_risks)
        }
    
    @staticmethod
    def _categorize_risks(risks: List[Dict]) -> Dict[str, int]:
        """对风险进行分类统计"""
        categories = {}
        for risk in risks:
            category = risk.get("category", "Uncategorized")
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    @staticmethod
    def coa_from_dict(coa_dict: Dict[str, Any]) -> COA:
        """从字典创建COA对象"""
        return COA.model_validate(coa_dict)
    
    @staticmethod
    def validate_and_format_for_simulation(pure_matrix_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证并格式化用于仿真的数据（基于事件和条件）"""
        # 确保必需字段存在
        required_fields = ["entities", "events", "matrix", "task_details"]
        for field in required_fields:
            if field not in pure_matrix_data:
                if field in ["entities", "events"]:
                    pure_matrix_data[field] = []
                else:
                    pure_matrix_data[field] = {}
        
        # 验证实体和事件
        entities = pure_matrix_data.get("entities", [])
        events = pure_matrix_data.get("events", [])
        
        # 确保矩阵结构完整
        matrix = pure_matrix_data.get("matrix", {})
        for entity in entities:
            if entity not in matrix:
                matrix[entity] = {}
            for event in events:
                if event not in matrix[entity]:
                    matrix[entity][event] = []
        
        # 更新纯矩阵数据
        pure_matrix_data["matrix"] = matrix
        
        return pure_matrix_data
    
    @staticmethod
    def convert_to_simulation_friendly_format(pure_matrix_data: Dict[str, Any]) -> Dict[str, Any]:
        """将纯矩阵数据转换为仿真系统友好格式（基于事件和条件）"""
        # 创建仿真系统可直接使用的格式
        sim_ready_data = {
            "entities": [],
            "timeline": [],  # 现在是事件驱动的，而非时间驱动
            "actions": [],
            "conditions": [],
            "event_driven_schedule": []
        }
        
        entities = pure_matrix_data.get("entities", [])
        events = pure_matrix_data.get("events", [])  # 使用事件而不是时间周期
        matrix = pure_matrix_data.get("matrix", {})
        task_details = pure_matrix_data.get("task_details", {})
        
        # 转换实体信息
        for entity in entities:
            entity_info = {
                "id": entity,
                "initial_state": "STANDBY",  # 默认初始状态
                "tasks_assigned": [],
                "event_handlers": []  # 事件处理器
            }
            
            # 获取该实体在各种事件下的任务
            entity_tasks = matrix.get(entity, {})
            for event, task_list in entity_tasks.items():
                for task_item in task_list:  # task_list 现在是字典列表
                    task_id = task_item.get("task_id", "")
                    if task_id in task_details:
                        task_detail = task_details[task_id]
                        task_assignment = {
                            "task_id": task_id,
                            "trigger_event": event,  # 使用事件而非时间周期
                            "task_type": task_detail.get("type", "GENERAL"),
                            "description": task_detail.get("description", ""),
                            "resources": task_detail.get("resources", ""),
                            "location": task_detail.get("location", "UNSPECIFIED"),
                            "start_condition": task_detail.get("start_condition"),
                            "end_condition": task_detail.get("end_condition")
                        }
                        
                        entity_info["tasks_assigned"].append(task_assignment)
                        
                        # 添加事件处理器
                        entity_info["event_handlers"].append({
                            "event": event,
                            "action": f"execute_task_{task_id}",
                            "condition": task_detail.get("start_condition", event)
                        })
            
            sim_ready_data["entities"].append(entity_info)
        
        # 转换事件驱动的时间线信息
        for event in events:
            event_entry = {
                "event": event,
                "triggers": []
            }
            
            # 为每个事件收集所有触发器
            for entity in entities:
                entity_tasks = matrix.get(entity, {})
                if event in entity_tasks:
                    for task_item in entity_tasks[event]:
                        event_entry["triggers"].append({
                            "entity": entity,
                            "task_id": task_item.get("task_id", ""),
                            "task_type": task_item.get("task_type", "GENERAL"),
                            "location": task_item.get("location", "")
                        })
            
            sim_ready_data["event_driven_schedule"].append(event_entry)
        
        # 收集所有条件
        all_conditions = set()
        for task_detail in task_details.values():
            if task_detail.get("start_condition"):
                all_conditions.add(task_detail["start_condition"])
            if task_detail.get("end_condition"):
                all_conditions.add(task_detail["end_condition"])
        
        sim_ready_data["conditions"] = list(all_conditions)
        
        return sim_ready_data