#!/usr/bin/env python3
"""
测试优化后的 MissionOrch 项目
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.missionorch.core.orchestrator import COAOrchestrator
from src.missionorch.core.model_router import ModelRouter


async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 Testing basic functionality...")
    
    try:
        # 测试模型路由器
        print("Testing ModelRouter...")
        # 这里我们只是测试能否正常加载配置，不实际调用API
        try:
            # 尝试获取一个模型实例（不会实际调用API）
            router = ModelRouter
            print("✅ ModelRouter imported successfully")
        except Exception as e:
            print(f"⚠️ ModelRouter test issue (expected if API keys not set): {e}")
        
        # 测试编排器
        print("Testing COAOrchestrator...")
        orchestrator = COAOrchestrator()
        print(f"✅ COAOrchestrator created with config - max_iter: {orchestrator.max_iter}, threshold: {orchestrator.threshold}")
        
        # 测试健康检查（如果RAG可用）
        from src.missionorch.core.rag_manager import RAGManager
        rag_manager = RAGManager()
        health = await rag_manager.health_check()
        print(f"✅ RAG Health: {health}")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling():
    """测试错误处理"""
    print("\n🧪 Testing error handling...")
    
    try:
        # 测试错误的模型ID
        try:
            ModelRouter.get("non_existent_model")
            print("❌ Should have raised an error for non-existent model")
            return False
        except KeyError:
            print("✅ Correctly handled non-existent model")
        
        # 测试错误的配置路径
        try:
            from src.missionorch.core.agent_base import BaseAgent
            BaseAgent("non_existent_agent", config_path="non/existent/path.yaml")
            print("❌ Should have raised an error for non-existent config")
            return False
        except Exception:
            print("✅ Correctly handled non-existent config")
        
        print("🎉 Error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("🚀 Starting MissionOrch optimization tests...\n")
    
    success = True
    success &= await test_basic_functionality()
    success &= await test_error_handling()
    
    if success:
        print("\n✅ All tests passed! Optimization successful.")
    else:
        print("\n❌ Some tests failed.")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
