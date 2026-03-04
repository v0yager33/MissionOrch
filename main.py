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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from missionorch.core.orchestrator import COAOrchestrator


async def main():
    """主函数"""
    print("MissionOrch - 任务编排系统")
    print("=" * 50)
    
    try:
        # 创建编排器实例
        orchestrator = COAOrchestrator()
        print(f"编排器初始化成功")
        print(f"配置: 最大迭代次数={orchestrator.max_iter}, 质量阈值={orchestrator.threshold}")
        
        # 获取用户输入
        print("\n请输入任务描述（输入 'quit' 退出）:")
        while True:
            mission_input = input("\n任务描述: ").strip()
            
            if mission_input.lower() in ['quit', 'exit', 'q']:
                print("退出系统")
                break
            
            if not mission_input:
                print("任务描述不能为空，请重新输入")
                continue
            
            print(f"\n正在生成COA方案...")
            try:
                result = await orchestrator.generate(mission_input)
                
                print(f"\nCOA生成完成!")
                print(f"迭代次数: {result['iterations']}")
                print(f"最终得分: {result['final_score']}")
                print(f"任务数量: {len(result['final_coa'].get('tasks', []))}")
                print(f"效果链数量: {len(result['final_coa'].get('effects_chain', []))}")
                
                # 显示历史记录
                print(f"\n迭代历史:")
                for entry in result['history']:
                    print(f"  迭代 {entry['iteration']}: 得分 {entry['score']}")
                
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
