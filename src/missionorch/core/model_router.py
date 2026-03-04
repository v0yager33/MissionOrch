import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import time
import yaml
import openai
import google.generativeai as genai
from anthropic import AsyncAnthropic


logger = logging.getLogger(__name__)


class ModelRouterError(Exception):
    """模型路由器自定义异常基类"""
    pass


class ConfigLoadError(ModelRouterError):
    """配置加载错误"""
    pass


class UnknownProviderError(ModelRouterError):
    """未知提供者错误"""
    pass


class ModelRouter:
    """统一模型路由 - 支持OpenAI/Gemini/Anthropic/OpenAI兼容API"""
    
    _instances: Dict[str, Any] = {}  # 模型实例缓存
    _config_cache: Dict[str, tuple] = {}  # 配置缓存 (配置内容, 修改时间, 过期时间)
    _config_cache_ttl = 300  # 配置缓存过期时间（秒），默认5分钟
    
    @classmethod
    def get(cls, model_id: str, config_path: str = "config/models.yaml"):
        """
        获取模型实例（单例）
        
        Args:
            model_id: 模型标识符
            config_path: 配置文件路径
            
        Returns:
            模型适配器实例
            
        Raises:
            ConfigLoadError: 当配置加载失败时
            KeyError: 当模型ID不存在时
        """
        # 检查是否已有该模型实例且配置未过期
        if model_id in cls._instances:
            return cls._instances[model_id]
        
        config = cls._load_config_cached(config_path)
        if 'models' not in config:
            raise ConfigLoadError(f"配置中缺少 'models' 键: {config_path}")
        
        if model_id not in config['models']:
            available_models = list(config['models'].keys())
            raise KeyError(f"模型 '{model_id}' 不存在于配置中。可用模型: {available_models}")
            
        model_config = config['models'][model_id]
        cls._instances[model_id] = cls._create_adapter(model_config)
        return cls._instances[model_id]
    
    @classmethod
    def _load_config_cached(cls, path: str) -> Dict[str, Any]:
        """
        加载配置文件并使用缓存
        
        Args:
            path: 配置文件路径
            
        Returns:
            处理后的配置字典
        """
        config_path = Path(path).resolve()
        current_time = time.time()
        
        # 检查缓存是否存在且未过期
        if path in cls._config_cache:
            cached_config, file_mtime, cache_timestamp = cls._config_cache[path]
            
            # 检查文件是否被修改以及缓存是否过期
            if config_path.exists() and config_path.stat().st_mtime <= file_mtime:
                if current_time - cache_timestamp < cls._config_cache_ttl:
                    return cached_config
        
        # 重新加载配置
        config = cls._load_config(path)
        
        # 更新缓存
        if config_path.exists():
            file_mtime = config_path.stat().st_mtime
            cls._config_cache[path] = (config, file_mtime, current_time)
        
        return config
    
    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """
        加载配置文件并处理环境变量
        
        Args:
            path: 配置文件路径
            
        Returns:
            处理后的配置字典
            
        Raises:
            ConfigLoadError: 当配置加载或解析失败时
        """
        try:
            config_path = Path(path).resolve()
            if not config_path.exists():
                raise ConfigLoadError(f"配置文件不存在: {path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ConfigLoadError(f"配置文件格式无效: {path}，应为字典格式")
                
            return ModelRouter._replace_env_vars(config)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML 解析错误: {e}")
        except Exception as e:
            raise ConfigLoadError(f"配置加载失败: {e}")
    
    @staticmethod
    def _replace_env_vars(obj: Any) -> Any:
        """
        递归替换对象中的环境变量占位符
        
        Args:
            obj: 要处理的对象（可以是字典、列表或字符串）
            
        Returns:
            替换后的对象
        """
        if isinstance(obj, dict):
            return {k: ModelRouter._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelRouter._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            # 提取环境变量名，支持默认值语法 ${VAR_NAME:default_value}
            var_expr = obj[2:-1]  # 移除 ${ 和 }
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name.strip(), default_value)
            else:
                var_name = var_expr.strip()
                value = os.getenv(var_name)
                if value is None:
                    logger.warning(f"环境变量未设置: {var_name}")
                return value
        else:
            return obj
    
    @staticmethod
    def _create_adapter(config: Dict[str, Any]):
        """
        根据配置创建相应的适配器实例
        
        Args:
            config: 模型配置字典
            
        Returns:
            对应的适配器实例
            
        Raises:
            UnknownProviderError: 当提供者不支持时
        """
        provider = config.get('provider')
        if not provider:
            raise ValueError("配置中缺少 'provider' 字段")
            
        if provider == 'openai':
            return OpenAIAdapter(config)
        elif provider == 'gemini':
            return GeminiAdapter(config)
        elif provider == 'anthropic':
            return AnthropicAdapter(config)
        elif provider == 'openai_compatible':
            return OpenAICompatibleAdapter(config)
        elif provider == 'doubao':
            return DoubaoAdapter(config)
        else:
            raise UnknownProviderError(f"不支持的模型提供者: {provider}")
    
    @classmethod
    def clear_cache(cls):
        """清除所有缓存的模型实例"""
        cls._instances.clear()
        cls._config_cache.clear()
    
    @classmethod
    def remove_instance(cls, model_id: str):
        """移除特定模型实例的缓存"""
        if model_id in cls._instances:
            del cls._instances[model_id]
    
    @classmethod
    def get_cached_instances(cls) -> Dict[str, Any]:
        """获取当前缓存的所有实例"""
        return cls._instances.copy()


class BaseAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config['model']
        self.default_temp = config.get('default_temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        self.timeout = config.get('timeout', 60)
        logger.debug(f"基础适配器已初始化，模型: {self.model}")
    
    async def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, **kwargs) -> str:
        """
        执行聊天请求的抽象方法
        
        Args:
            messages: 消息列表，每个消息包含 role 和 content
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            模型生成的文本响应
        """
        raise NotImplementedError


class OpenAIAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI 适配器需要 'api_key' 配置")
            
        # 使用配置中的超时值，默认为120秒以处理较慢的API
        timeout_val = config.get('timeout', 120)
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=config.get('base_url'),
            timeout=timeout_val
        )
        self.timeout = timeout_val
        logger.info(f"OpenAI 适配器已初始化，模型: {self.model}")
    
    async def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, **kwargs) -> str:
        """
        执行 OpenAI 聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            **kwargs: 其他传递给 API 的参数
            
        Returns:
            生成的文本响应
            
        Raises:
            Exception: 当 API 请求失败时
        """
        try:
            logger.debug(f"向 OpenAI 模型 {self.model} 发送请求，消息数量: {len(messages)}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.default_temp,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise Exception("API 返回空内容")
                
            logger.debug(f"OpenAI 模型 {self.model} 响应成功，内容长度: {len(content)}")
            return content
        except openai.APIError as e:
            logger.error(f"OpenAI API 错误 (模型 {self.model}): {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI 请求失败 (模型 {self.model}): {e}")
            raise


class GeminiAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("Gemini 适配器需要 'api_key' 配置")
            
        genai.configure(api_key=api_key)
        
        safety_settings = config.get('safety_settings', {
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        })
        
        # 初始化基础模型，系统指令将在聊天时处理
        self.gen_model = genai.GenerativeModel(
            model_name=config['model'],
            safety_settings=safety_settings
        )
        logger.info(f"Gemini 适配器已初始化，模型: {self.model}")
    
    async def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, **kwargs) -> str:
        """
        执行 Gemini 聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            **kwargs: 其他传递给 API 的参数
            
        Returns:
            生成的文本响应
            
        Raises:
            Exception: 当 API 请求失败时
        """
        try:
            logger.debug(f"向 Gemini 模型 {self.model} 发送请求，消息数量: {len(messages)}")
            
            # 分离系统消息和其他消息
            system_instruction = None
            chat_history = []
            
            for msg in messages:
                role = msg['role']
                content = msg['content']
                
                if role == 'system':
                    # 使用第一个系统消息作为指令
                    if system_instruction is None:
                        system_instruction = content
                elif role in ['user', 'assistant']:
                    # 将标准角色映射到 Gemini 角色
                    gemini_role = 'user' if role == 'user' else 'model'
                    chat_history.append({
                        'role': gemini_role,
                        'parts': [content]
                    })
            
            # 创建带系统指令的模型（如果存在）
            if system_instruction:
                model_with_system = genai.GenerativeModel(
                    model_name=self.config['model'],
                    safety_settings=self.gen_model.safety_settings,
                    system_instruction=system_instruction
                )
                logger.debug(f"使用系统指令初始化 Gemini 模型")
            else:
                model_with_system = self.gen_model
            
            generation_config = {
                'temperature': temperature if temperature is not None else self.default_temp,
                'max_output_tokens': self.max_tokens
            }
            
            # 添加其他可能的配置参数
            for key in ['top_p', 'top_k']:
                if key in kwargs:
                    generation_config[key] = kwargs[key]
            
            # 如果有历史消息，创建带历史的聊天会话
            if chat_history:
                # 移除最后一条消息用于发送，保留前面的消息作为历史
                last_message = chat_history.pop() if chat_history else {'role': 'user', 'parts': ['Hello']}
                
                chat_session = model_with_system.start_chat(history=chat_history)
                logger.debug(f"创建带历史记录的 Gemini 聊天会话，历史消息数: {len(chat_history)}")
                
                response = await chat_session.send_message_async(
                    last_message['parts'][0],
                    generation_config=generation_config
                )
            else:
                # 如果没有历史消息，直接生成内容
                logger.debug(f"直接生成 Gemini 内容，无历史消息")
                response = await model_with_system.generate_content_async(
                    "Hello",
                    generation_config=generation_config
                )
            
            content = response.text
            if not content:
                raise Exception("Gemini API 返回空内容")
                
            logger.debug(f"Gemini 模型 {self.model} 响应成功，内容长度: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"Gemini API 错误 (模型 {self.model}): {e}")
            raise


class AnthropicAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("Anthropic 适配器需要 'api_key' 配置")
            
        self.client = AsyncAnthropic(api_key=api_key)
        logger.info(f"Anthropic 适配器已初始化，模型: {self.model}")
    
    async def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, **kwargs) -> str:
        """
        执行 Anthropic 聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            **kwargs: 其他传递给 API 的参数
            
        Returns:
            生成的文本响应
            
        Raises:
            Exception: 当 API 请求失败时
        """
        try:
            logger.debug(f"向 Anthropic 模型 {self.model} 发送请求，消息数量: {len(messages)}")
            
            # 分离系统消息和其他消息
            system_msg_parts = [msg['content'] for msg in messages if msg['role'] == 'system']
            system_msg = '\n'.join(system_msg_parts) if system_msg_parts else None
            
            # 过滤掉系统消息，只保留 user 和 assistant 消息
            chat_msgs = [
                {"role": msg['role'], "content": msg['content']} 
                for msg in messages 
                if msg['role'] in ['user', 'assistant']
            ]
            
            # 确保消息交替出现 (user, assistant, user, ...)
            # Anthropic 要求必须以 user 消息结尾
            if chat_msgs and chat_msgs[-1]['role'] != 'user':
                # 如果最后一条不是用户消息，我们添加一个空的用户消息
                chat_msgs.append({"role": "user", "content": "请继续"})
            
            api_params = {
                'model': self.model,
                'max_tokens': self.max_tokens,
                'temperature': temperature if temperature is not None else self.default_temp,
                'messages': chat_msgs
            }
            
            if system_msg:
                api_params['system'] = system_msg
                logger.debug(f"使用系统消息初始化 Anthropic 请求")
                
            # 添加其他可选参数
            for key in ['top_p', 'top_k', 'stop_sequences']:
                if key in kwargs:
                    api_params[key] = kwargs[key]
            
            response = await self.client.messages.create(**api_params)
            
            if not response.content:
                raise Exception("Anthropic API 返回空内容")
                
            # Anthropic 返回的是内容块列表，提取文本
            content_text = ""
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    content_text += content_block.text
                elif hasattr(content_block, 'type') and content_block.type == 'text':
                    content_text += content_block.text
                    
            if not content_text:
                raise Exception("Anthropic API 返回空内容")
                
            logger.debug(f"Anthropic 模型 {self.model} 响应成功，内容长度: {len(content_text)}")
            return content_text
        except Exception as e:
            logger.error(f"Anthropic API 错误 (模型 {self.model}): {e}")
            raise


class OpenAICompatibleAdapter(OpenAIAdapter):
    """兼容OpenAI接口的所有模型（Seed/Qwen/DeepSeek/Local）"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logger.info(f"OpenAI兼容适配器已初始化，模型: {self.model}")


class DoubaoAdapter(BaseAdapter):
    """豆包（Doubao）适配器 - 专门支持字节跳动豆包API"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("豆包适配器需要 'api_key' 配置")
            
        # 使用 OpenAI 客户端，但配置为豆包 API
        self.client = openai.AsyncOpenAI(  # 使用异步客户端
            base_url=config.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3'),
            api_key=api_key,
            timeout=config.get('timeout', 120)  # 增加默认超时时间到120秒
        )
        self.timeout = config.get('timeout', 120)  # 增加默认超时时间
        logger.info(f"豆包适配器已初始化，模型: {self.model}")
    
    async def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, **kwargs) -> str:
        """
        执行豆包聊天请求
        """
        try:
            logger.debug(f"向豆包模型 {self.model} 发送请求，消息数量: {len(messages)}")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.default_temp,
                max_tokens=self.max_tokens,
                timeout=self.timeout,  # 显式设置超时
                **kwargs
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise Exception("豆包API 返回空内容")
                
            logger.debug(f"豆包模型 {self.model} 响应成功，内容长度: {len(content)}")
            return content
        except Exception as e:
            logger.error(f"豆包API 错误 (模型 {self.model}): {e}")
            raise