#!/usr/bin/env python3
"""
MissionOrch 使用示例
展示如何使用完整的COA生成和验证系统
"""

import asyncio
import json
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from missionorch.core.orchestrator import COAOrchestrator


async def main():
    """主函数 - 演示完整使用流程"""
    print("MissionOrch - COA生成与验证系统演示")
    print("=" * 50)
    
    # 创建编排器
    orchestrator = COAOrchestrator()
    print(f"✓ 编排器已初始化")
    print(f"  - 规划智能体: {orchestrator.planner.model_id}")
    print(f"  - 评估智能体: {orchestrator.judge.model_id}")
    print(f"  - 反思智能体: {orchestrator.reflector.model_id}")
    print(f"  - 验证智能体: {orchestrator.validator.model_id}")
    
    # 示例任务描述
    mission_description = """
    任务：保护我方重要设施免受敌方空中威胁
    背景：敌方拥有先进战斗机和地对空导弹系统，我方需要建立防空体系
    目标：确保关键基础设施安全，拦截来袭敌机
    资源：F-35战机4架，爱国者导弹系统2套，预警机1架
    限制：避免平民区域战斗，燃料限制飞行时间不超过2小时
    """
    
    print(f"\n📋 任务描述:")
    print(f"  {mission_description.strip()[:100]}...")
    
    print(f"\n🚀 开始COA生成流程...")
    print("   步骤1: 规划 -> 评估 -> 反思 -> 验证")
    
    # 生成COA
    result = await orchestrator.generate(mission_description)
    
    print(f"\n✅ COA生成完成!")
    print(f"   迭代次数: {result['iterations']}")
    print(f"   最终得分: {result['final_score']}")
    print(f"   验证结果: {'通过' if result['validation']['is_valid'] else '未通过'}")
    
    if result['validation']['issues_found']:
        print(f"   发现问题: {len(result['validation']['issues_found'])} 个")
        for issue in result['validation']['issues_found'][:3]:  # 只显示前3个
            print(f"     - {issue}")
    
    print(f"\n📊 COA统计信息:")
    final_coa = result['final_coa']
    print(f"   任务总数: {len(final_coa['tasks'])}")
    print(f"   效果总数: {len(final_coa['effects_chain'])}")
    print(f"   资源分配: {len(final_coa['resource_allocation'])} 个单位")
    print(f"   决策点: {len(final_coa['decision_points'])} 个")
    
    print(f"\n💾 输出格式:")
    outputs = result['outputs']
    print(f"   JSON格式: {len(outputs['json_format'])} 字符")
    print(f"   YAML格式: {len(outputs['yaml_format'])} 字符")
    print(f"   仿真矩阵: {len(outputs['simulation_matrix'])} 个条目")
    print(f"   矩阵表格: {len(outputs['matrix_table']['entities'])} 个实体")
    
    print(f"\n🎯 验证智能体提取的纯矩阵数据:")
    pure_matrix = result['validation']['pure_matrix_data']
    print(f"   实体数量: {len(pure_matrix['entities'])}")
    print(f"   事件数量: {len(pure_matrix['events'])}")
    print(f"   任务详情: {len(pure_matrix['task_details'])} 个")
    
    print(f"\n🎮 仿真就绪格式:")
    sim_ready = result['validation']['simulation_ready_format']
    print(f"   仿真实体: {len(sim_ready['entities'])}")
    print(f"   事件驱动时间线: {len(sim_ready['event_driven_schedule'])} 个事件")
    
    # 显示一些具体的矩阵数据
    print(f"\n📋 示例矩阵数据 (前3个实体):")
    matrix = outputs['matrix_table']
    for i, entity in enumerate(list(matrix['matrix'].keys())[:3]):
        entity_events = matrix['matrix'][entity]
        print(f"   {entity}:")
        for event, tasks in list(entity_events.items())[:2]:  # 只显示前2个事件
            if tasks:
                task_list = ", ".join([t['task_id'] for t in tasks])
                print(f"     {event}: [{task_list}]")
            else:
                print(f"     {event}: 无任务")
    
    print(f"\n✨ 演示完成!")
    print(f"系统已成功生成人类可读的COA和仿真系统可用的矩阵数据。")


if __name__ == "__main__":
    asyncio.run(main())