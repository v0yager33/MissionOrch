#!/usr/bin/env python3
"""MissionOrch-LC 使用示例 —— 展示完整的 COA 生成与验证流程。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from missionorch_lc.core.log_config import setup_logging  # noqa: E402

setup_logging()

from missionorch_lc.orchestrator import COAOrchestrator  # noqa: E402


async def main() -> None:
    print("MissionOrch-LC — COA 生成与验证系统演示")
    print("=" * 60)

    orchestrator = COAOrchestrator()
    print("✓ 编排器已初始化")
    print(f"  - 规划:   {orchestrator.planner.model_id}")
    print(f"  - 评估:   {orchestrator.judge.model_id}")
    print(f"  - 反思:   {orchestrator.reflector.model_id}")
    print(f"  - 验证:   {orchestrator.validator.model_id}")

    mission_description = """
    任务：保护我方重要设施免受敌方空中威胁
    背景：敌方拥有先进战斗机和地对空导弹系统，我方需要建立防空体系
    目标：确保关键基础设施安全，拦截来袭敌机
    资源：F-35战机4架，爱国者导弹系统2套，预警机1架
    限制：避免平民区域战斗，燃料限制飞行时间不超过2小时
    """

    print("\n📋 任务描述:")
    print(f"  {mission_description.strip()[:100]}...")

    print("\n🚀 开始 COA 生成流程...")
    result = await orchestrator.generate(mission_description)

    print("\n✅ COA 生成完成!")
    print(f"   迭代次数: {result['iterations']}")
    print(f"   最终得分: {result['final_score']}")
    print(f"   最佳得分: {result['best_score']}")
    print(f"   矩阵解析: {'成功' if result['parse_success'] else '失败'}")

    validation = result.get("validation", {})
    if validation:
        print(f"   验证结果: {'通过' if validation.get('is_valid') else '未通过'}")
        issues = validation.get("issues_found", [])
        if issues:
            print(f"   发现问题 ({len(issues)} 个):")
            for issue in issues[:3]:
                print(f"     - {issue}")

    if result["parse_success"]:
        final_coa = result["final_coa"]
        print("\n📊 COA 统计信息:")
        print(f"   阶段数量:     {len(final_coa['phases'])}")
        print(f"   作战单元数量: {len(final_coa['units'])}")
        print(f"   矩阵单元格:   {len(final_coa['matrix'])}")
        print(f"   效果链:       {len(final_coa['effects_chain'])}")
        print(f"   决策点:       {len(final_coa['decision_points'])}")
        print(f"   关键风险:     {len(final_coa['critical_risks'])}")

        outputs = result["outputs"]
        print("\n💾 输出格式长度:")
        print(f"   JSON:         {len(outputs['json_format'])} chars")
        print(f"   YAML:         {len(outputs['yaml_format'])} chars")
        print(f"   扁平矩阵:     {len(outputs['flat_matrix'].get('matrix', {}))} 个单元")

    print("\n✨ 演示完成！")


if __name__ == "__main__":
    asyncio.run(main())
