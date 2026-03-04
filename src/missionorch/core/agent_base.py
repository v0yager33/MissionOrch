import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from .model_router import ModelRouter
from .rag_manager import RAGManager

logger = logging.getLogger(__name__)


class AgentConfigError(Exception):
    """智能体配置错误"""
    pass


class BaseAgent:
    """Agent基类 - 支持RAG（可开关）和提示词热加载"""
    
    def __init__(self, agent_type: str, config_path: str = "config/agents.yaml"):
        self.agent_type = agent_type
        self.config_path = config_path
        
        # 加载配置
        self.config = self._load_config(config_path)
        self.agent_cfg = self._validate_agent_config(self.config, agent_type)
        self.workflow_cfg = self.config.get('workflow', {})
        
        # 初始化模型
        model_id = self.agent_cfg['model_id']
        self.model = ModelRouter.get(model_id)
        self.model_id = model_id
        
        # RAG（可开关）
        self.use_rag = self.agent_cfg.get('use_rag', False)
        self.rag_manager = RAGManager() if self.use_rag else None
        
        # 提示词热加载
        self.prompt_path = self.agent_cfg.get('prompt_file')
        self._prompt_cache = None
        self._prompt_mtime = 0
        
        logger.info(f"BaseAgent '{agent_type}' initialized with model '{model_id}'")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件并处理错误"""
        try:
            config_path_obj = Path(config_path).resolve()
            if not config_path_obj.exists():
                raise AgentConfigError(f"配置文件不存在: {config_path}")
            
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise AgentConfigError(f"配置文件格式无效: {config_path}")
            
            if 'agents' not in config:
                raise AgentConfigError(f"配置文件中缺少 'agents' 部分: {config_path}")
                
            return config
        except yaml.YAMLError as e:
            raise AgentConfigError(f"YAML 解析错误: {e}")
        except Exception as e:
            raise AgentConfigError(f"配置加载失败: {e}")
    
    def _validate_agent_config(self, config: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        """验证智能体配置"""
        if agent_type not in config.get('agents', {}):
            available_agents = list(config['agents'].keys()) if 'agents' in config else []
            raise AgentConfigError(f"智能体类型 '{agent_type}' 不存在于配置中。可用类型: {available_agents}")
        
        agent_config = config['agents'][agent_type]
        
        if 'model_id' not in agent_config:
            raise AgentConfigError(f"智能体 '{agent_type}' 配置中缺少 'model_id'")
        
        return agent_config
    
    @property
    def system_prompt(self) -> str:
        """动态加载提示词（支持热更新）"""
        if not self.prompt_path:
            logger.debug(f"No prompt file configured for agent '{self.agent_type}'")
            return ""
        
        try:
            prompt_path_obj = Path(self.prompt_path)
            if not prompt_path_obj.exists():
                logger.warning(f"Prompt file does not exist: {self.prompt_path}")
                return ""
            
            mtime = prompt_path_obj.stat().st_mtime
            if mtime > self._prompt_mtime:
                with open(prompt_path_obj, 'r', encoding='utf-8') as f:
                    self._prompt_cache = f.read()
                self._prompt_mtime = mtime
                logger.info(f"Reloaded prompt: {self.prompt_path}")
        except Exception as e:
            logger.error(f"Failed to load prompt file {self.prompt_path}: {e}")
            return ""
        
        return self._prompt_cache or ""
    
    async def retrieve_context(self, query: str) -> str:
        """RAG检索（如果启用）"""
        if not self.use_rag or not self.rag_manager:
            logger.debug(f"RAG not enabled for agent '{self.agent_type}'")
            return ""
        
        try:
            logger.debug(f"Retrieving context for agent '{self.agent_type}' with query length: {len(query)}")
            context = await self.rag_manager.retrieve(
                query, 
                top_k=self.agent_cfg.get('rag_retrieval_depth', 5)
            )
            logger.debug(f"Retrieved context with length: {len(context) if context else 0}")
            return context
        except Exception as e:
            logger.error(f"RAG retrieval failed for agent '{self.agent_type}': {e}")
            return ""
    
    async def generate(self, user_content: str, context: Optional[Dict[str, Any]] = None, 
                      temp_override: Optional[float] = None) -> str:
        """通用生成方法"""
        try:
            logger.debug(f"Generating response for agent '{self.agent_type}' with content length: {len(user_content)}")
            
            # 构建消息，安全地处理格式化以避免JSON中的大括号被误解析
            import re
            prompt = self.system_prompt
            for key, value in (context or {}).items():
                # 使用正则表达式进行安全的字符串替换，避免特殊字符的影响
                # 匹配 {key} 模式，其中key是变量名，不包含空格或其他特殊字符
                pattern = r'\{' + re.escape(key) + r'\}'
                
                # 将值转为字符串，如果是复杂对象则转换为JSON字符串
                if isinstance(value, (dict, list)):
                    import json
                    replacement = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                else:
                    replacement = str(value)
                
                # 使用正则表达式替换，确保只替换完整的变量占位符
                prompt = re.sub(pattern, replacement, prompt)
            
            temp = temp_override
            if temp is None and 'temperature_override' in self.agent_cfg:
                temp = self.agent_cfg['temperature_override']
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content}
            ]
            
            response = await self.model.chat(messages, temperature=temp)
            logger.debug(f"Generated response with length: {len(response) if response else 0}")
            return response
        except Exception as e:
            logger.error(f"Generation failed for agent '{self.agent_type}': {e}")
            raise
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON"""
        try:
            if '```json' in text:
                return text.split('```json')[1].split('```')[0].strip()
            if '```' in text:
                return text.split('```')[1].split('```')[0].strip()
            # 尝试提取花括号内的内容
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return match.group(0).strip()
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract JSON from text: {e}")
            return text.strip()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套键访问"""
        keys = key.split('.')
        value = self.agent_cfg
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value