#!/usr/bin/env python3
"""
测试MissionOrch项目的完整功能
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from missionorch.log_config import setup_logging

# 每次运行生成独立的带时间戳日志文件
log_timestamp = setup_logging()
print(f"日志文件: logs/debug_{log_timestamp}.log, logs/interactions_{log_timestamp}.log")

from missionorch.core.orchestrator import COAOrchestrator


async def test_full_system():
    """测试完整系统功能"""
    print("Testing MissionOrch Full System (v3.0 - COA Matrix Architecture)")
    print("=" * 70)
    
    try:
        # 创建编排器
        orchestrator = COAOrchestrator()
        print("✓ Orchestrator created successfully")
        
        # 测试任务描述
        mission = """
        任务：保护我方重要设施免受敌方空中威胁
        背景：敌方拥有先进4*F35战斗机编队*3，我方需要建立防空体系
        目标：确保关键基础设施安全，拦截来袭敌机
        场景：海上空域作战
        资源：4*J20战机编队*3，电子干扰机*1，预警机*1
        限制：燃料限制飞行时间不超过2小时
        """
        
        print(f"Running test with mission: {mission.strip()[:60]}...")
        print("-" * 70)
        
        # 生成COA（规划-评估-反思循环，全程自然语言矩阵）
        result = await orchestrator.generate(mission)
        
        # 显示自然语言 COA 矩阵（系统的主要输出）
        print("\n" + "=" * 70)
        print("【最终 COA 矩阵（自然语言）】")
        print("=" * 70)
        print(result['coa_table'])
        print("=" * 70)
        
        # 显示迭代信息
        print(f"\n【迭代信息】")
        print(f"  - 迭代次数: {result['iterations']}")
        print(f"  - 最终得分: {result['final_score']}")
        print(f"  - 最佳得分: {result['best_score']}")
        for entry in result['history']:
            print(f"  - 迭代 {entry['iteration']}: 得分 {entry['score']} | 判定 {entry['verdict']}")
        
        # 显示结构化转换结果
        print(f"\n【结构化转换】")
        print(f"  - 解析成功: {result['parse_success']}")
        if result['parse_success']:
            final_coa = result['final_coa']
            print(f"  - 阶段数量: {len(final_coa.get('phases', []))}")
            print(f"  - 作战单元数量: {len(final_coa.get('units', []))}")
            print(f"  - 矩阵单元格数量: {len(final_coa.get('matrix', []))}")
            print(f"  - 效果链数量: {len(final_coa.get('effects_chain', []))}")
            print(f"  - 决策点数量: {len(final_coa.get('decision_points', []))}")
            
            outputs = result['outputs']
            print(f"  - JSON 格式: {len(outputs['json_format'])} chars")
            print(f"  - YAML 格式: {len(outputs['yaml_format'])} chars")
        
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
