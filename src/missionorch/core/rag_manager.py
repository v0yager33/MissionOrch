import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)

# RAG-Anything导入
try:
    from raganything import RAGAnything, RAGAnythingConfig
    RAG_AVAILABLE = True
    logger.info("RAG-Anything is available")
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG-Anything not installed, RAG functionality will be disabled")


class RAGConfigError(Exception):
    """RAG配置错误"""
    pass


class RAGManager:
    """RAG管理器 - 可开关，封装RAG-Anything"""
    
    def __init__(self, config_path: str = "config/rag.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.enabled = self.config.get('rag', {}).get('enabled', False) and RAG_AVAILABLE
        
        self.rag_engine = None
        self._engine_initialized = False
        
        if self.enabled:
            logger.info("RAG is enabled, initializing engine...")
            try:
                self._init_engine()
            except Exception as e:
                logger.error(f"Failed to initialize RAG engine: {e}")
                self.enabled = False
        else:
            if not RAG_AVAILABLE:
                logger.warning("RAG-Anything not available, RAG disabled")
            else:
                logger.info("RAG is disabled by configuration")
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_path = Path(path).resolve()
            if not config_path.exists():
                logger.warning(f"RAG config file does not exist: {path}, using defaults")
                return {'rag': {'enabled': False}}
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise RAGConfigError(f"RAG配置文件格式无效: {path}")
                
            return config
        except yaml.YAMLError as e:
            logger.error(f"YAML 解析错误 in RAG config: {e}")
            return {'rag': {'enabled': False}}
        except Exception as e:
            logger.error(f"RAG配置加载失败: {e}")
            return {'rag': {'enabled': False}}
    
    def _init_engine(self):
        """初始化RAG-Anything引擎"""
        if not RAG_AVAILABLE:
            logger.error("Cannot initialize RAG engine: RAG-Anything not available")
            return
            
        rag_cfg = self.config.get('rag', {})
        
        try:
            # 配置
            working_dir = Path(rag_cfg.get('working_dir', './knowledge_base/rag_storage'))
            working_dir.mkdir(parents=True, exist_ok=True)
            
            config = RAGAnythingConfig(
                working_dir=str(working_dir),
                parser=rag_cfg.get('parser', 'mineru'),
                enable_image_processing=rag_cfg.get('enable_image_processing', True),
                enable_table_processing=rag_cfg.get('enable_table_processing', True),
                enable_equation_processing=rag_cfg.get('enable_equation_processing', False)
            )
            
            # 初始化RAG引擎
            logger.info(f"Initializing RAG engine with working dir: {working_dir}")
            self.rag_engine = RAGAnything(config=config)
            self._engine_initialized = True
            logger.info("RAG engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            self._engine_initialized = False
            raise
    
    async def retrieve(self, query: str, top_k: int = 5) -> str:
        """检索知识（如果启用）"""
        if not self.enabled or not self._engine_initialized or not self.rag_engine:
            logger.debug("RAG not available for retrieval")
            return ""
        
        try:
            logger.debug(f"Performing RAG retrieval for query (length: {len(query)}, top_k: {top_k})")
            
            search_mode = self.config.get('retrieval', {}).get('search_mode', 'hybrid')
            result = await self.rag_engine.aquery(
                query,
                mode=search_mode,
                top_k=top_k,
                vlm_enhanced=self.config.get('rag', {}).get('enable_image_processing', True)
            )
            
            result_str = str(result) if result is not None else ""
            logger.debug(f"RAG retrieval completed, result length: {len(result_str)}")
            return result_str
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return ""
    
    async def index_knowledge_base(self) -> bool:
        """索引知识库（初始化时调用）"""
        if not self.enabled or not self._engine_initialized or not self.rag_engine:
            logger.warning("Cannot index knowledge base: RAG not available")
            return False
        
        try:
            logger.info("Starting knowledge base indexing...")
            
            sources = self.config.get('rag', {}).get('knowledge_sources', {})
            indexed_count = 0
            
            for kb_name, kb_config in sources.items():
                path = kb_config.get('path')
                if path and Path(path).exists():
                    logger.info(f"Indexing {kb_name} from {path}...")
                    
                    # 使用RAG-Anything进行索引
                    await self.rag_engine.ainsert(
                        path,
                        embedding_batch_num=10,
                        insert_threads=4
                    )
                    
                    indexed_count += 1
                    logger.info(f"Successfully indexed {kb_name}")
                else:
                    logger.warning(f"Knowledge source path does not exist: {path}")
            
            logger.info(f"Knowledge base indexing completed. Indexed {indexed_count} sources.")
            return indexed_count > 0
        except Exception as e:
            logger.error(f"Knowledge base indexing failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查RAG是否可用"""
        return self.enabled and self._engine_initialized and self.rag_engine is not None
    
    def is_enabled(self) -> bool:
        """检查RAG是否已启用"""
        return self.enabled
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "enabled": self.is_enabled(),
            "available": self.is_available(),
            "engine_initialized": self._engine_initialized,
            "rag_available": RAG_AVAILABLE
        }