"""RAG 管理器 —— 正确装配 RAG-Anything（LightRAG 后端）。

修复了原项目的几个关键问题：
1. 原项目只传了 `config`，没传 `llm_model_func` / `embedding_func` —— 必然崩溃
2. 原项目调 `aquery(top_k=..., vlm_enhanced=...)` —— 这两个参数 aquery 不接受
3. 原项目 LLM 与 embedding 都依赖 OpenAI API，本地无法跑

现在的方案：
- LLM 走 `ModelRouter`（已支持 doubao / qwen / deepseek / 本地 ollama）
- Embedding 走本地 ``Qwen3-Embedding-0.6B``
- Reranker 走本地 ``Qwen3-Reranker-0.6B``（可选）
- 多源知识库：每个源（doctrines / maps / historical / glossary）有独立的
  workspace，互不污染，可以分别检索

调用方应优先通过 `tools/rag_tools.py` 的 LangChain Tool 暴露给 Agent。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .rag.lightrag_factory import build_rag_anything, ensure_initialized

logger = logging.getLogger(__name__)


# ── 检查 raganything 是否可用 ──
try:
    import raganything  # type: ignore  # noqa: F401

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning(
        "RAG-Anything 未安装；RAG 功能将禁用。安装方法见 docs/RAG_GUIDE.md"
    )


class RAGManager:
    """RAG 管理器（多知识源 + 本地模型）。

    每个知识源对应一个独立的 LightRAG workspace（子目录），
    这样可以按知识类型（条令 / 地图 / 战例 / 术语）分别检索。
    """

    def __init__(self, config_path: str = "config/rag.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.rag_cfg: Dict[str, Any] = self.config.get("rag", {})
        self.retrieval_cfg: Dict[str, Any] = self.config.get("retrieval", {})

        self.enabled = bool(self.rag_cfg.get("enabled", False)) and RAG_AVAILABLE
        self._engines: Dict[str, Any] = {}
        self._initialized: Dict[str, bool] = {}

        if not self.enabled:
            logger.info("RAGManager: 禁用（rag.enabled=False 或 RAG-Anything 未安装）")
            return

        # 立即构建所有 source 的引擎对象（懒初始化 LightRAG storage）
        sources = self.rag_cfg.get("knowledge_sources") or {}
        if not sources:
            logger.warning("RAGManager: rag.knowledge_sources 为空，启用但无可检索源")
        for source_name in sources.keys():
            try:
                self._engines[source_name] = self._build_engine_for_source(source_name)
                logger.info(f"RAGManager: 引擎 '{source_name}' 已构建")
            except Exception as build_error:
                logger.error(
                    f"RAGManager: 构建引擎 '{source_name}' 失败: {build_error}",
                    exc_info=True,
                )

    # ── 配置 ──
    def _load_config(self, path: str) -> Dict[str, Any]:
        config_path = Path(path).resolve()
        if not config_path.exists():
            logger.warning(f"RAG config not found: {path}, using defaults")
            return {"rag": {"enabled": False}}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {"rag": {"enabled": False}}
        except Exception as load_error:
            logger.error(f"RAG 配置加载失败: {load_error}")
            return {"rag": {"enabled": False}}

    # ── 引擎构建 ──
    def _build_engine_for_source(self, source_name: str) -> Any:
        """按 source 创建独立 working_dir 的 RAGAnything 实例。"""
        sources_cfg = self.rag_cfg.get("knowledge_sources") or {}
        source_cfg = sources_cfg.get(source_name) or {}

        base_working_dir = Path(self.rag_cfg.get("working_dir", "./knowledge_base/rag_storage"))
        source_working_dir = base_working_dir / source_name

        models_cfg = self.rag_cfg.get("models") or {}
        embedding_model_path = models_cfg.get("embedding_path")
        if not embedding_model_path:
            raise ValueError("rag.models.embedding_path 未配置（必须）")
        rerank_model_path = models_cfg.get("rerank_path")  # 可选
        llm_model_id = models_cfg.get("llm_model_id", "seed_doubao")
        vision_model_id = models_cfg.get("vision_model_id")  # 可选

        # source 级别的覆盖
        enable_image = source_cfg.get(
            "enable_image_processing",
            self.rag_cfg.get("enable_image_processing", True),
        )

        # LightRAG 调参
        lightrag_kwargs: Dict[str, Any] = {}
        top_k = self.retrieval_cfg.get("top_k")
        if top_k is not None:
            lightrag_kwargs["top_k"] = int(top_k)
        chunk_top_k = self.retrieval_cfg.get("chunk_top_k")
        if chunk_top_k is not None:
            lightrag_kwargs["chunk_top_k"] = int(chunk_top_k)

        return build_rag_anything(
            working_dir=str(source_working_dir),
            embedding_model_path=embedding_model_path,
            llm_model_id=llm_model_id,
            rerank_model_path=rerank_model_path,
            vision_model_id=vision_model_id,
            parser=self.rag_cfg.get("parser", "mineru"),
            enable_image_processing=enable_image,
            enable_table_processing=self.rag_cfg.get("enable_table_processing", True),
            enable_equation_processing=self.rag_cfg.get(
                "enable_equation_processing", False
            ),
            embedding_device=models_cfg.get("embedding_device", "auto"),
            rerank_device=models_cfg.get("rerank_device", "auto"),
            embedding_max_length=int(models_cfg.get("embedding_max_length", 8192)),
            embedding_batch_size=int(models_cfg.get("embedding_batch_size", 8)),
            lightrag_kwargs=lightrag_kwargs,
        )

    async def _ensure_engine_initialized(self, source_name: str) -> Optional[Any]:
        engine = self._engines.get(source_name)
        if engine is None:
            return None
        if not self._initialized.get(source_name, False):
            try:
                await ensure_initialized(engine)
                self._initialized[source_name] = True
            except Exception as init_error:
                logger.error(
                    f"LightRAG 初始化失败 (source={source_name}): {init_error}",
                    exc_info=True,
                )
                return None
        return engine

    # ── 公共 API ──
    def is_enabled(self) -> bool:
        return self.enabled

    def list_sources(self) -> List[str]:
        return list(self._engines.keys())

    async def retrieve(
        self,
        query: str,
        source: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """检索单一 source（或默认 source），返回拼好的文本。"""
        if not self.enabled:
            return ""
        if not query:
            return ""

        target = source or self._default_source()
        if target is None:
            logger.debug("RAGManager.retrieve: no source available")
            return ""

        engine = await self._ensure_engine_initialized(target)
        if engine is None:
            return ""

        search_mode = mode or self.retrieval_cfg.get("search_mode", "hybrid")
        try:
            # 注意：aquery 只接受 (query, mode, ...)；top_k 由 LightRAG 的初始化参数决定
            result = await engine.aquery(query, mode=search_mode)
            return str(result) if result else ""
        except Exception as retrieval_error:
            logger.error(
                f"RAG retrieval failed (source={target}, mode={search_mode}): "
                f"{retrieval_error}",
                exc_info=True,
            )
            return ""

    async def retrieve_all(
        self,
        query: str,
        mode: Optional[str] = None,
    ) -> Dict[str, str]:
        """并发检索所有 source，返回 {source_name: text}。"""
        import asyncio

        if not self.enabled or not self._engines:
            return {}
        names = list(self._engines.keys())
        results = await asyncio.gather(
            *[self.retrieve(query, source=name, mode=mode) for name in names],
            return_exceptions=False,
        )
        return {name: text for name, text in zip(names, results) if text}

    async def insert(self, source: str, file_path: str) -> bool:
        """把单个文件索引进指定 source 的知识库。"""
        if not self.enabled:
            return False
        engine = await self._ensure_engine_initialized(source)
        if engine is None:
            return False
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"insert: 文件不存在 {file_path}")
            return False
        try:
            await engine.process_document_complete(
                file_path=str(path_obj),
                output_dir=str(path_obj.parent / "_parsed"),
            )
            logger.info(f"已索引 {file_path} → source={source}")
            return True
        except Exception as insert_error:
            logger.error(f"索引失败 {file_path}: {insert_error}", exc_info=True)
            return False

    async def insert_directory(self, source: str, directory: str) -> int:
        """递归索引目录下的所有支持文件，返回成功数量。"""
        if not self.enabled:
            return 0
        sources_cfg = self.rag_cfg.get("knowledge_sources") or {}
        file_types = (sources_cfg.get(source) or {}).get(
            "file_types", [".pdf", ".docx", ".txt", ".md"]
        )
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"insert_directory: 目录不存在 {directory}")
            return 0

        success = 0
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in file_types:
                if await self.insert(source, str(file_path)):
                    success += 1
        logger.info(f"目录索引完成: {success} 个文件 → source={source}")
        return success

    def _default_source(self) -> Optional[str]:
        if not self._engines:
            return None
        # 优先 doctrines（条令一般是兜底主源）
        if "doctrines" in self._engines:
            return "doctrines"
        return next(iter(self._engines.keys()))