#!/usr/bin/env python3
"""
API端点连通性测试脚本
发送'你好'测试请求并打印返回结果
"""

import os
import yaml
import subprocess
import json
import sys
from pathlib import Path

def load_models_config():
    """加载模型配置"""
    with open("config/models.yaml", 'r', encoding='utf-8') as f:
        models_config = yaml.safe_load(f)
    
    return models_config

def check_api_key(key_name):
    """检查API密钥是否已设置"""
    # 处理${VAR_NAME:default_value}格式
    if key_name.startswith('${') and key_name.endswith('}'):
        var_expr = key_name[2:-1]
        if ':' in var_expr:
            var_name, default_value = var_expr.split(':', 1)
            env_value = os.getenv(var_name.strip())
            if env_value is not None:
                return env_value, True
            else:
                return default_value, False
        else:
            var_name = var_expr
            env_value = os.getenv(var_name.strip())
            return env_value, env_value is not None
    else:
        return key_name, True

def test_curl_with_response(url, headers=None, data=None):
    """使用curl测试连通性并返回完整响应"""
    try:
        cmd = ["curl", "-s", "-w", "\n%{http_code}", "--max-time", "30"]
        
        if headers:
            for header in headers:
                cmd.extend(["-H", header])
        
        if data:
            cmd.extend(["-d", json.dumps(data, separators=(',', ':'))])
            if not any("Content-Type" in h for h in headers or []):
                cmd.extend(["-H", "Content-Type: application/json"])
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            # 分离响应体和状态码
            parts = output.rsplit('\n', 1)
            if len(parts) == 2 and parts[1].isdigit():
                response_body = parts[0]
                status_code = parts[1]
            else:
                response_body = output
                status_code = "UNKNOWN"
            return status_code, response_body
        else:
            return "ERROR", result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "请求超时"
    except Exception as e:
        return f"EXCEPTION", str(e)

def test_api_endpoints(models_config):
    """测试API端点连通性"""
    print("\n🔍 测试API端点 - 发送'你好'请求...")
    
    models = models_config.get('models', {})
    
    for model_name, config in models.items():
        provider = config.get('provider', 'unknown')
        base_url = config.get('base_url')
        api_key_placeholder = config.get('api_key')
        model_id = config.get('model')
        
        print(f"\n--- {model_name} ---")
        print(f"  提供者: {provider}")
        print(f"  模型: {model_id}")
        print(f"  端点: {base_url}")
        
        # 检查API密钥
        api_key, is_set = check_api_key(api_key_placeholder)
        if not is_set or not api_key or api_key.startswith("${"):
            print(f"  API密钥: ⚠️  未设置或为占位符")
            print(f"    跳过连通性测试")
            continue
        
        print(f"  API密钥: ✅ 已设置")
        
        # 根据提供者类型构造测试请求
        if provider in ['openai', 'openai_compatible', 'doubao']:
            # 测试/chat/completions端点
            test_url = f"{base_url.rstrip('/')}/chat/completions"
            headers = [
                f"Authorization: Bearer {api_key}",
                "Content-Type: application/json"
            ]
            
            # 发送"你好"测试请求
            test_data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "你好"}
                ],
                "max_tokens": 20,
                "temperature": 0.1
            }
            
            print(f"  📡 发送请求: POST {test_url}")
            print(f"  💬 请求内容: 你好")
            status_code, response = test_curl_with_response(test_url, headers, test_data)
            
            print(f"  📊 HTTP状态码: {status_code}")
            if status_code.isdigit() and int(status_code) < 400:
                print(f"  ✅ 请求成功")
                try:
                    response_json = json.loads(response)
                    if 'choices' in response_json and len(response_json['choices']) > 0:
                        content = response_json['choices'][0].get('message', {}).get('content', '无内容')
                        print(f"  💬 返回内容: {content}")
                    elif 'error' in response_json:
                        print(f"  ❌ API错误: {response_json['error']}")
                    else:
                        print(f"  📝 响应: {response[:200]}...")
                except json.JSONDecodeError:
                    print(f"  📝 非JSON响应: {response[:200]}...")
            elif status_code == "401":
                print(f"  🚫 API密钥无效 (HTTP 401)")
            elif status_code == "404":
                print(f"  🚫 端点不存在 (HTTP 404)")
            elif status_code == "TIMEOUT":
                print(f"  ⏰ 请求超时")
            elif status_code == "ERROR":
                print(f"  ❌ 连接错误或端点不可达")
                print(f"     错误详情: {response}")
            else:
                print(f"  ❌ 请求失败 (HTTP {status_code})")
                print(f"     响应: {response[:200]}...")
        elif provider == 'gemini':
            # Gemini API使用不同的格式
            test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            headers = ["Content-Type: application/json"]
            test_data = {
                "contents": [{
                    "parts": [{
                        "text": "你好"
                    }]
                }]
            }
            print(f"  📡 发送请求: POST {test_url}")
            print(f"  💬 请求内容: 你好")
            status_code, response = test_curl_with_response(test_url, headers, test_data)
            
            print(f"  📊 Gemini API状态码: {status_code}")
            if status_code.isdigit() and int(status_code) < 400:
                print(f"  ✅ 请求成功")
                try:
                    response_json = json.loads(response)
                    if 'candidates' in response_json and len(response_json['candidates']) > 0:
                        content = response_json['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', '无内容')
                        print(f"  💬 返回内容: {content}")
                    elif 'error' in response_json:
                        print(f"  ❌ API错误: {response_json['error']}")
                    else:
                        print(f"  📝 响应: {response[:200]}...")
                except json.JSONDecodeError:
                    print(f"  📝 非JSON响应: {response[:200]}...")
            else:
                print(f"  ❌ 请求失败 (HTTP {status_code})")
                print(f"     响应: {response[:200]}...")
        elif provider == 'anthropic':
            # Anthropic API测试
            test_url = f"{base_url.rstrip('/')}/messages" if base_url else "https://api.anthropic.com/v1/messages"
            headers = [
                f"x-api-key: {api_key}",
                "Content-Type: application/json",
                "anthropic-version: 2023-06-01"
            ]
            test_data = {
                "model": model_id,
                "max_tokens": 20,
                "messages": [
                    {"role": "user", "content": "你好"}
                ]
            }
            print(f"  📡 发送请求: POST {test_url}")
            print(f"  💬 请求内容: 你好")
            status_code, response = test_curl_with_response(test_url, headers, test_data)
            
            print(f"  📊 Anthropic API状态码: {status_code}")
            if status_code.isdigit() and int(status_code) < 400:
                print(f"  ✅ 请求成功")
                try:
                    response_json = json.loads(response)
                    if 'content' in response_json and len(response_json['content']) > 0:
                        content = response_json['content'][0].get('text', '无内容')
                        print(f"  💬 返回内容: {content}")
                    elif 'error' in response_json:
                        print(f"  ❌ API错误: {response_json['error']}")
                    else:
                        print(f"  📝 响应: {response[:200]}...")
                except json.JSONDecodeError:
                    print(f"  📝 非JSON响应: {response[:200]}...")
            else:
                print(f"  ❌ 请求失败 (HTTP {status_code})")
                print(f"     响应: {response[:200]}...")
        else:
            print(f"  ℹ️  未知提供者类型，跳过测试")

def main():
    print("🌐 API端点连通性测试 - '你好'请求")
    print("=" * 60)
    
    try:
        models_config = load_models_config()
        
        print(f"✅ 成功加载模型配置")
        print(f"  模型数量: {len(models_config.get('models', {}))}")
        
        test_api_endpoints(models_config)
        
        print(f"\n✅ API端点测试完成!")
        print(f"💡 说明：")
        print(f"   - 发送'你好'请求到每个API端点")
        print(f"   - 打印HTTP状态码和返回内容")
        print(f"   - 需要设置环境变量中的API密钥才能测试")
        
    except FileNotFoundError as e:
        print(f"❌ 配置文件未找到: {e}")
        print(f"请确保在项目根目录下有 config/models.yaml 文件")
    except yaml.YAMLError as e:
        print(f"❌ YAML解析错误: {e}")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()