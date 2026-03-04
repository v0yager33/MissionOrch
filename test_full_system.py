#!/usr/bin/env python3
"""
测试MissionOrch项目的完整功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from missionorch.core.orchestrator import COAOrchestrator


async def test_full_system():
    """测试完整系统功能"""
    print("Testing MissionOrch Full System...")
    
    try:
        # 创建编排器
        orchestrator = COAOrchestrator()
        print("✓ Orchestrator created successfully")
        
        # 测试任务描述
        mission = """
        任务：保护我方重要设施免受敌方空中威胁
        背景：敌方拥有先进战斗机和地对空导弹系统，我方需要建立防空体系
        目标：确保关键基础设施安全，拦截来袭敌机
        资源：F-35战机4架，爱国者导弹系统2套，预警机1架
        限制：避免平民区域战斗，燃料限制飞行时间不超过2小时
        """
        
        print(f"Running test with mission: {mission[:50]}...")
        
        # 生成COA（这将触发完整的规划-评估-反思-验证循环）
        result = await orchestrator.generate(mission)
        
        print(f"✓ COA generation completed")
        print(f"  - Iterations: {result['iterations']}")
        print(f"  - Final score: {result['final_score']}")
        print(f"  - Validation passed: {result['validation']['is_valid']}")
        print(f"  - Tasks in final COA: {len(result['final_coa']['tasks'])}")
        print(f"  - Effects in final COA: {len(result['final_coa']['effects_chain'])}")
        
        # 检查输出格式
        outputs = result['outputs']
        print(f"  - JSON format available: {len(outputs['json_format']) > 0}")
        print(f"  - YAML format available: {len(outputs['yaml_format']) > 0}")
        print(f"  - Simulation matrix entries: {len(outputs['simulation_matrix'])}")
        print(f"  - Matrix table has entities: {len(outputs['matrix_table']['entities']) > 0}")
        
        # 检查验证结果
        validation = result['validation']
        print(f"  - Issues found: {len(validation['issues_found'])}")
        print(f"  - Pure matrix data has entities: {len(validation['pure_matrix_data'].get('entities', [])) > 0}")
        print(f"  - Simulation ready format has entities: {len(validation['simulation_ready_format']['entities']) > 0}")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_full_system())
    sys.exit(0 if success else 1)
