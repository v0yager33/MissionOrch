#!/usr/bin/env python3
"""
验证新项目结构是否正确
"""

import sys
from pathlib import Path

# 添加项目源码路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """测试所有必要的导入"""
    try:
        from missionorch.core.orchestrator import COAOrchestrator
        print("✓ COAOrchestrator import successful")
        
        from missionorch.core.agent_impl import PlannerAgent, JudgeAgent, ReflectorAgent
        print("✓ Agent implementations import successful")
        
        from missionorch.core.agent_validator import ValidatorAgent
        print("✓ ValidatorAgent import successful")
        
        from missionorch.schemas.coa import COA, Task, TaskType
        print("✓ COA schema import successful")
        
        from missionorch.core.coa_transformer import COATransformer
        print("✓ COATransformer import successful")
        
        print("\n✓ All imports successful! Project structure is correct.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
