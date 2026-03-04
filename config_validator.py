#!/usr/bin/env python3
"""
配置验证工具
用于验证 MissionOrch 项目的配置文件
"""

import sys
from pathlib import Path
import yaml
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def validate_models_config(config_path="config/models.yaml"):
    """验证模型配置"""
    print("验证模型配置...")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'models' not in config:
            print("  错误: 配置中缺少 'models' 部分")
            return False
        
        models = config['models']
        required_fields = ['provider', 'model']
        
        valid_providers = ['openai', 'gemini', 'anthropic', 'openai_compatible', 'doubao']
        
        for model_id, model_config in models.items():
            print(f"  检查模型: {model_id}")
            
            # 检查必需字段
            for field in required_fields:
                if field not in model_config:
                    print(f"    错误: 模型 {model_id} 缺少 '{field}' 字段")
                    return False
            
            # 检查提供者是否有效
            provider = model_config['provider']
            if provider not in valid_providers:
                print(f"    错误: 模型 {model_id} 的提供者 '{provider}' 无效")
                print(f"    有效提供者: {valid_providers}")
                return False
            
            # 检查API密钥（如果是环境变量格式）
            if 'api_key' in model_config:
                api_key = model_config['api_key']
                if isinstance(api_key, str) and api_key.startswith('${') and api_key.endswith('}'):
                    print(f"    注意: 模型 {model_id} 使用环境变量 {api_key}")
        
        print("  模型配置验证通过")
        return True
        
    except FileNotFoundError:
        print(f"  错误: 配置文件 {config_path} 不存在")
        return False
    except yaml.YAMLError as e:
        print(f"  错误: YAML 解析错误 - {e}")
        return False
    except Exception as e:
        print(f"  错误: 验证过程中发生错误 - {e}")
        return False


def validate_agents_config(config_path="config/agents.yaml"):
    """验证智能体配置"""
    print("验证智能体配置...")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'agents' not in config:
            print("  错误: 配置中缺少 'agents' 部分")
            return False
        
        agents = config['agents']
        required_fields = ['model_id']
        
        for agent_id, agent_config in agents.items():
            print(f"  检查智能体: {agent_id}")
            
            # 检查必需字段
            for field in required_fields:
                if field not in agent_config:
                    print(f"    错误: 智能体 {agent_id} 缺少 '{field}' 字段")
                    return False
            
            # 检查提示文件是否存在
            if 'prompt_file' in agent_config:
                prompt_file = agent_config['prompt_file']
                if not Path(prompt_file).exists():
                    print(f"    警告: 提示文件 {prompt_file} 不存在")
        
        # 检查工作流配置
        if 'workflow' in config:
            workflow = config['workflow']
            print("  检查工作流配置")
            
            if 'max_iterations' in workflow:
                max_iter = workflow['max_iterations']
                if not isinstance(max_iter, int) or max_iter <= 0:
                    print(f"    错误: max_iterations 应为正整数，当前值: {max_iter}")
                    return False
            
            if 'quality_threshold' in workflow:
                threshold = workflow['quality_threshold']
                if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 10:
                    print(f"    错误: quality_threshold 应为 0-10 之间的数值，当前值: {threshold}")
                    return False
        
        print("  智能体配置验证通过")
        return True
        
    except FileNotFoundError:
        print(f"  错误: 配置文件 {config_path} 不存在")
        return False
    except yaml.YAMLError as e:
        print(f"  错误: YAML 解析错误 - {e}")
        return False
    except Exception as e:
        print(f"  错误: 验证过程中发生错误 - {e}")
        return False


def validate_rag_config(config_path="config/rag.yaml"):
    """验证RAG配置"""
    print("验证RAG配置...")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        rag_config = config.get('rag', {})
        
        print(f"  RAG状态: {'启用' if rag_config.get('enabled', False) else '禁用'}")
        
        # 检查工作目录
        if 'working_dir' in rag_config:
            working_dir = rag_config['working_dir']
            print(f"  工作目录: {working_dir}")
        
        # 检查知识源
        knowledge_sources = rag_config.get('knowledge_sources', {})
        for source_name, source_config in knowledge_sources.items():
            path = source_config.get('path')
            if path:
                exists = Path(path).exists()
                print(f"  知识源 '{source_name}': {path} ({'存在' if exists else '不存在'})")
        
        print("  RAG配置验证通过")
        return True
        
    except FileNotFoundError:
        print(f"  错误: 配置文件 {config_path} 不存在")
        return False
    except yaml.YAMLError as e:
        print(f"  错误: YAML 解析错误 - {e}")
        return False
    except Exception as e:
        print(f"  错误: 验证过程中发生错误 - {e}")
        return False


def main():
    """主函数"""
    print("MissionOrch 配置验证工具")
    print("=" * 40)
    
    all_valid = True
    
    # 验证所有配置文件
    all_valid &= validate_models_config()
    print()
    all_valid &= validate_agents_config()
    print()
    all_valid &= validate_rag_config()
    
    print()
    if all_valid:
        print("所有配置验证通过!")
        return 0
    else:
        print("部分配置存在问题，请检查以上错误信息")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)