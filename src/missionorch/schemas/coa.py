from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    SEAD = "SEAD"                    # 压制防空
    OCA = "OFFENSIVE_COUNTER_AIR"    # 攻势制空
    DCA = "DEFENSIVE_COUNTER_AIR"    # 防御制空
    STRIKE = "STRIKE"                # 打击
    ESCORT = "ESCORT"                # 护航
    AAR = "AAR"                      # 空中加油
    AEW = "AEW"                      # 预警指挥
    RECON = "RECON"                  # 侦察

class Task(BaseModel):
    """战役级任务（不含战术机动细节）"""
    task_id: str = Field(default="task_default", description="任务标识")
    task_type: TaskType = Field(default=TaskType.RECON, description="任务类型")
    description: str = Field(default="Default task description", description="任务描述（做什么，非怎么做）")
    assigned_unit: str = Field(default="default_unit", description="分配单位")
    platform_count: int = Field(default=1, ge=1)
    
    # 时空参数（基于事件和条件，而非固定时间）
    start_condition: Optional[str] = Field(None, description="开始条件（如：'接到命令'、'到达某位置'、'检测到威胁'）")
    end_condition: Optional[str] = Field(None, description="结束条件（如：'完成任务'、'燃料不足'、'威胁解除'）")
    location: Optional[str] = Field(None, description="执行位置或区域")
    duration_min: Optional[int] = None
    duration_max: Optional[int] = None
    
    # 依赖网络
    dependencies: List[str] = Field(default=[], description="前置任务ID")
    supported_tasks: List[str] = Field(default=[], description="支持的任务ID")
    hard_dependencies: bool = True  # 是否必须完成前置才能开始
    
    # 效果关联
    contributes_to: List[str] = Field(default=[], description="贡献的效果ID")
    
    # 评估标准
    success_criteria: str = Field(default="Task completed successfully", description="成功标准")
    abort_triggers: List[str] = Field(default=[])

class Effect(BaseModel):
    """战略效果"""
    effect_id: str = Field(default="effect_default", description="效果ID")
    description: str = Field(default="Default effect description", description="效果描述")
    measures: List[str] = Field(default=["Default measure"], description="衡量指标")
    achieved_by: List[str] = Field(default=[], description="达成该效果的任务ID")
    duration_min: Optional[int] = None

class DecisionPoint(BaseModel):
    """决策点"""
    dp_id: str = Field(default="dp_default", description="决策点ID")
    trigger_time: str = Field(default="T+0", description="触发时间（T+XX）")
    condition: str = Field(default="Default condition", description="触发条件")
    options: List[Dict[str, str]] = Field(default=[], description="选项：if/then")

class SimulationMatrixEntry(BaseModel):
    """仿真矩阵条目"""
    entity_id: str = Field(default="entity_default", description="实体ID")
    phase: str = Field(default="phase_default", description="阶段名称")
    task_type: str = Field(default="task_default", description="任务类型")
    spatial_params: Dict[str, Any] = Field(default_factory=dict, description="空间参数")
    temporal_params: Dict[str, Any] = Field(default_factory=dict, description="时间参数")
    behavior_params: Dict[str, Any] = Field(default_factory=dict, description="行为参数")
    condition_params: Dict[str, Any] = Field(default_factory=dict, description="条件参数")
    transition_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="转换条件")
    simulation_commands: List[Dict[str, Any]] = Field(default_factory=list, description="仿真命令")

class COA(BaseModel):
    """行动方案（Course of Action）"""
    coa_id: str = Field(default_factory=lambda: f"COA-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    description: str = Field(default="Generated COA", description="COA描述")
    
    # 核心要素
    effects_chain: List[Effect] = Field(default_factory=list, description="效果链")
    tasks: List[Task] = Field(default_factory=list, description="任务网络（DAG）")
    critical_path: List[str] = Field(default=[], description="关键路径任务ID")
    
    # 同步矩阵
    synchronization_matrix: Dict[str, List[str]] = Field(default_factory=dict, description="时间->任务列表")
    
    # 资源分配
    resource_allocation: Dict[str, List[str]] = Field(default_factory=dict, description="单位->任务列表")
    
    # 决策点
    decision_points: List[DecisionPoint] = Field(default=[], description="决策点")
    
    # 风险
    critical_risks: List[Dict] = Field(default=[], description="关键风险")
    contingency_coas: List[str] = Field(default=[], description="应急COA")
    
    # 元数据
    metadata: Dict = Field(default_factory=dict, description="元数据")
    
    def to_simulation_matrix(self) -> List[SimulationMatrixEntry]:
        """将COA转换为仿真矩阵"""
        matrix_entries = []
        
        # 为每个任务创建对应的仿真矩阵条目
        for task in self.tasks:
            # 确定任务阶段
            phase = self._determine_task_phase(task)
            
            # 构建空间参数
            spatial_params = self._build_spatial_params(task)
            
            # 构建时间参数
            temporal_params = self._build_temporal_params(task)
            
            # 构建行为参数
            behavior_params = self._build_behavior_params(task)
            
            # 构建条件参数
            condition_params = self._build_condition_params(task)
            
            # 构建转换条件
            transition_conditions = self._build_transition_conditions(task)
            
            # 构建仿真命令
            simulation_commands = self._build_simulation_commands(task)
            
            entry = SimulationMatrixEntry(
                entity_id=task.assigned_unit,
                phase=phase,
                task_type=task.task_type.value,
                spatial_params=spatial_params,
                temporal_params=temporal_params,
                behavior_params=behavior_params,
                condition_params=condition_params,
                transition_conditions=transition_conditions,
                simulation_commands=simulation_commands
            )
            
            matrix_entries.append(entry)
        
        return matrix_entries
    
    def _determine_task_phase(self, task: Task) -> str:
        """确定任务所属阶段"""
        if task.start_condition:
            return f"PHASE_{task.start_condition.replace(' ', '_').upper()}"
        elif task.location:
            return f"PHASE_AT_{task.location.replace(' ', '_').upper()}"
        else:
            # 根据任务类型确定阶段
            phase_map = {
                TaskType.SEAD: "PHASE_SUPPRESSION",
                TaskType.OCA: "PHASE_AIR_SUPERIORITY",
                TaskType.DCA: "PHASE_DEFENSE",
                TaskType.STRIKE: "PHASE_STRIKE",
                TaskType.ESCORT: "PHASE_ESCORT",
                TaskType.AAR: "PHASE_REFUELING",
                TaskType.AEW: "PHASE_SURVEILLANCE",
                TaskType.RECON: "PHASE_RECONNAISSANCE"
            }
            return phase_map.get(task.task_type, "PHASE_GENERAL")
    
    def _build_spatial_params(self, task: Task) -> Dict[str, Any]:
        """构建空间参数"""
        params = {}
        
        # 添加位置信息（如果有的话）
        if task.location:
            params["location"] = task.location
        
        # 添加单位标识
        params["unit"] = task.assigned_unit
        params["platform_count"] = task.platform_count
        
        return params
    
    def _build_temporal_params(self, task: Task) -> Dict[str, Any]:
        """构建时间参数（基于事件和条件）"""
        params = {}
        
        if task.start_condition:
            params["start_condition"] = task.start_condition
        if task.end_condition:
            params["end_condition"] = task.end_condition
        if task.duration_min:
            params["duration_min"] = task.duration_min
        if task.duration_max:
            params["duration_max"] = task.duration_max
            
        return params
    
    def _build_behavior_params(self, task: Task) -> Dict[str, Any]:
        """构建行为参数"""
        params = {
            "task_type": task.task_type.value,
            "description": task.description,
            "success_criteria": task.success_criteria
        }
        
        # 根据任务类型添加特定参数
        if task.task_type in [TaskType.SEAD, TaskType.STRIKE]:
            params["engagement_rules"] = "ENGAGE_SELECTED_TARGETS"
        elif task.task_type in [TaskType.OCA, TaskType.DCA]:
            params["engagement_rules"] = "ENGAGE_HOSTILE_AIRCRAFT"
        elif task.task_type == TaskType.AEW:
            params["engagement_rules"] = "MONITOR_AND_REPORT_ONLY"
        elif task.task_type == TaskType.RECON:
            params["engagement_rules"] = "AVOID_ENGAGEMENT_UNLESS_NECESSARY"
        
        return params
    
    def _build_condition_params(self, task: Task) -> Dict[str, Any]:
        """构建条件参数"""
        params = {
            "dependencies": task.dependencies,
            "supported_tasks": task.supported_tasks,
            "hard_dependencies": task.hard_dependencies,
            "contributing_effects": task.contributes_to,
            "abort_triggers": task.abort_triggers
        }
        
        return params
    
    def _build_transition_conditions(self, task: Task) -> List[Dict[str, Any]]:
        """构建转换条件"""
        conditions = []
        
        # 添加完成条件
        conditions.append({
            "type": "TASK_COMPLETED",
            "trigger": "success_criteria_met",
            "description": task.success_criteria
        })
        
        # 添加中止条件
        for trigger in task.abort_triggers:
            conditions.append({
                "type": "TASK_ABORTED",
                "trigger": trigger,
                "description": f"Abort on {trigger}"
            })
        
        # 添加依赖条件
        if task.dependencies:
            conditions.append({
                "type": "DEPENDENCY_CHECK",
                "trigger": "all_dependencies_met",
                "dependencies": task.dependencies
            })
        
        return conditions
    
    def _build_simulation_commands(self, task: Task) -> List[Dict[str, Any]]:
        """构建仿真命令"""
        commands = []
        
        # 根据任务类型生成基本命令
        if task.task_type in [TaskType.SEAD, TaskType.STRIKE]:
            commands.extend([
                {"command": "NAVIGATE_TO", "target": "TARGET_AREA"},
                {"command": "ACQUIRE_TARGETS", "range": "50nm"},
                {"command": "ENGAGE_TARGETS", "rules": "WEAPON_PRIORITY"}
            ])
        elif task.task_type in [TaskType.OCA, TaskType.DCA]:
            commands.extend([
                {"command": "PATROL_AREA", "pattern": "ORBIT"},
                {"command": "MONITOR_AIRSPACE", "sector": "AUTHORIZED"},
                {"command": "INTERCEPT_THREATS", "priority": "HOSTILE_FIRST"}
            ])
        elif task.task_type == TaskType.AEW:
            commands.extend([
                {"command": "MAINTAIN_ORBIT", "altitude": "30000ft"},
                {"command": "MONITOR_RADAR", "range": "200nm"},
                {"command": "RELAY_INFORMATION", "network": "TACTICAL_DATA_LINK"}
            ])
        elif task.task_type == TaskType.RECON:
            commands.extend([
                {"command": "FLY_RECON_ROUTE", "pattern": "PRE_PLANNED"},
                {"command": "GATHER_INTELLIGENCE", "focus": "MOVEMENTS"},
                {"command": "REPORT_FINDINGS", "frequency": "EVERY_15_MINUTES"}
            ])
        elif task.task_type == TaskType.ESCORT:
            commands.extend([
                {"command": "POSITION_NEAR", "escortee": "PROTECTED_UNIT"},
                {"command": "PROVIDE_PROTECTION", "threats": "ALL_AIRBORNE"},
                {"command": "MAINTAIN_FORMATION", "distance": "CUSTOMARY_INTERVAL"}
            ])
        elif task.task_type == TaskType.AAR:
            commands.extend([
                {"command": "NAVIGATE_TO", "tanker_position": "COORDINATED_POINT"},
                {"command": "EXECUTE_REFUELING", "sequence": "PRIORITY_ORDER"},
                {"command": "MAINTAIN_SAFETY", "distance": "MINIMUM_SEPARATION"}
            ])
        
        return commands