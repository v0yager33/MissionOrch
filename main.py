#!/usr/bin/env python3
"""
MissionOrch - 任务编排系统主入口
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from missionorch.log_config import setup_logging

# 每次运行生成独立的带时间戳日志文件
log_timestamp = setup_logging()

logger = logging.getLogger(__name__)

from missionorch.core.orchestrator import COAOrchestrator


async def async_input(prompt: str) -> str:
    """非阻塞的异步输入，避免阻塞事件循环"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def main():
    """主函数"""
    print("MissionOrch - 任务编排系统")
    print("=" * 50)
    
    try:
        orchestrator = COAOrchestrator()
        print(f"编排器初始化成功")
        print(f"配置: 最大迭代次数={orchestrator.max_iter}, 质量阈值={orchestrator.threshold}")
        
        print("\n请输入任务描述（输入 'quit' 退出）:")
        while True:
            mission_input = (await async_input("\n任务描述: ")).strip()
            
            if mission_input.lower() in ['quit', 'exit', 'q']:
                print("退出系统")
                break
            
            if not mission_input:
                print("任务描述不能为空，请重新输入")
                continue
            
            print(f"\n正在生成COA方案...")
            try:
                result = await orchestrator.generate(mission_input)
                
                # 显示自然语言 COA 矩阵（系统的主要输出）
                print("\n" + "=" * 70)
                print("【COA 矩阵方案】")
                print("=" * 70)
                print(result['coa_table'])
                print("=" * 70)
                
                print(f"\n【生成信息】")
                print(f"迭代次数: {result['iterations']}")
                print(f"最终得分: {result['final_score']}")
                print(f"最佳得分: {result['best_score']}")
                print(f"矩阵解析: {'成功' if result['parse_success'] else '失败'}")
                if result['parse_success']:
                    final_coa = result['final_coa']
                    print(f"阶段数量: {len(final_coa.get('phases', []))}")
                    print(f"作战单元数量: {len(final_coa.get('units', []))}")
                    print(f"矩阵单元格: {len(final_coa.get('matrix', []))}")
                    print(f"效果链数量: {len(final_coa.get('effects_chain', []))}")
                
                print(f"\n迭代历史:")
                for entry in result['history']:
                    print(f"  迭代 {entry['iteration']}: 得分 {entry['score']} | 判定 {entry['verdict']}")
                
                if result['parse_success']:
                    print(f"\n结构化格式已生成（JSON/YAML），详见日志文件: logs/")
                
            except Exception as e:
                logger.error(f"COA生成失败: {e}")
                print(f"生成失败: {e}")
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"系统错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)