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

    def __init__(self, config_path: str | None = None) -> None:
        if config_path is None:
            from .settings import get_settings
            config_path = str(get_settings().rag_config)

        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.rag_cfg: Dict[str, Any] = self.config.get("rag", {})
        self.retrieval_cfg: Dict[str, Any] = self.config.get("retrieval", {})
        self.chunking_cfg: Dict[str, Any] = self.config.get("chunking", {})

        self.enabled = bool(self.rag_cfg.get("enabled", False)) and RAG_AVAILABLE
        self._engines: Dict[str, Any] = {}
        self._initialized: Dict[str, bool] = {}
        self._query_rewriter: Optional[Any] = None  # MultiQueryRewriter，懒加载

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

        # 构造 Query Rewriter（配置驱动，可关）
        self._query_rewriter = self._build_query_rewriter()

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
        """按 source 创建独立 working_dir 的 RAGAnything 实例。

        从 rag.yaml 读取的字段（已与 docs/RAG_GUIDE.md 对齐）：
          rag.embedding.{model_path, device, max_length, batch_size, instruction}
          rag.reranker.{enabled, model_path, device, max_length}
          rag.llm_model_id      (默认 seed_doubao)
          rag.vision_model_id   (可选)
          rag.parser / enable_*_processing
        """
        sources_cfg = self.rag_cfg.get("knowledge_sources") or {}
        source_cfg = sources_cfg.get(source_name) or {}

        base_working_dir = Path(
            self.rag_cfg.get("working_dir", "./knowledge_base/rag_storage")
        )
        source_working_dir = base_working_dir / source_name

        # ── Embedding 配置（必填）──
        embedding_cfg: Dict[str, Any] = self.rag_cfg.get("embedding") or {}
        embedding_model_path = embedding_cfg.get("model_path")
        if not embedding_model_path:
            raise ValueError(
                "rag.embedding.model_path 未配置（必须）。"
                "请在 config/rag.yaml 设置本地 Qwen3-Embedding 模型路径。"
            )

        # ── Reranker 配置（可选）──
        reranker_cfg: Dict[str, Any] = self.rag_cfg.get("reranker") or {}
        rerank_model_path: Optional[str] = None
        if reranker_cfg.get("enabled", True):
            rerank_model_path = reranker_cfg.get("model_path")

        # ── LLM / Vision ──
        llm_model_id = self.rag_cfg.get("llm_model_id", "seed_doubao")
        vision_model_id = self.rag_cfg.get("vision_model_id")  # 可选

        # source 级别的覆盖
        enable_image = source_cfg.get(
            "enable_image_processing",
            self.rag_cfg.get("enable_image_processing", True),
        )

        # ── LightRAG 调参 ──
        # 检索阶段（影响每次查询的 recall / context 大小）
        # 这些字段都在初始化 LightRAG 时一次性写入实例属性，运行时不变。
        lightrag_kwargs: Dict[str, Any] = {}
        retrieval_kw_map = {
            "top_k": int,                        # 实体/关系层 top_k
            "chunk_top_k": int,                  # naive 模式核心参数
            "cosine_threshold": float,           # 余弦阈值
            "related_chunk_number": int,         # 实体被命中时附带的 chunk 数
            "max_total_tokens": int,             # 拼进 prompt 的总 token
            "max_entity_tokens": int,            # 实体段 token 上限
            "max_relation_tokens": int,          # 关系段 token 上限
        }
        for key, caster in retrieval_kw_map.items():
            value = self.retrieval_cfg.get(key)
            if value is not None:
                try:
                    lightrag_kwargs[key] = caster(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "RAGManager: retrieval.%s 转 %s 失败，跳过 (raw=%r)",
                        key, caster.__name__, value,
                    )

        # Chunking 阶段（影响下一次 ainsert 的切分；已索引产物不会重切）
        chunking_kw_map = {
            "chunk_token_size": int,
            "chunk_overlap_token_size": int,
            "tiktoken_model_name": str,
        }
        for key, caster in chunking_kw_map.items():
            value = self.chunking_cfg.get(key)
            if value is not None:
                try:
                    lightrag_kwargs[key] = caster(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "RAGManager: chunking.%s 转 %s 失败，跳过 (raw=%r)",
                        key, caster.__name__, value,
                    )

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
            embedding_device=embedding_cfg.get("device", "auto"),
            rerank_device=reranker_cfg.get("device", "auto"),
            embedding_max_length=int(embedding_cfg.get("max_length", 8192)),
            embedding_batch_size=int(embedding_cfg.get("batch_size", 8)),
            embedding_instruction=embedding_cfg.get("instruction"),
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
        return {name: text for name, text in zip(names, results, strict=True) if text}

    # 已经是纯文本、不需要 MinerU 解析的扩展名（小写）
    # MD/TXT 原本就是文本；CSV 术语表也按纯文本处理（朴素 RAG 不需要列结构）
    _PLAINTEXT_EXTS = {".md", ".txt", ".csv"}

    async def insert(self, source: str, file_path: str) -> bool:
        """把单个文件索引进指定 source 的知识库。

        分 3 条路径：
          1. 纯文本 (.md / .txt / .csv)：直接读文件内容 → LightRAG.ainsert(text)，
             **绕过 MinerU**。MinerU 在 CPU 上会加载 LayoutLM/OCR 模型 40s+，
             对纯文本是完全没必要的开销。
          2. HTML (.html / .htm / .xhtml)：用 bs4 转成 md → 按路径 1 处理。
             （MinerU pipeline backend 本来就没有 HTML 原生分支）
          3. 其它格式 (.pdf / .docx / .doc / 图片)：走 MinerU 的
             process_document_complete，这些文件真的需要版面分析/OCR。
        """
        if not self.enabled:
            return False
        engine = await self._ensure_engine_initialized(source)
        if engine is None:
            return False
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error(f"insert: 文件不存在 {file_path}")
            return False

        # 对 HTML 做预处理：转 md 到同目录下的 _html_as_md/ 缓存目录，
        # 并记录"实际要喂给下游的路径"
        effective_path = path_obj
        try:
            from .rag.html_to_md import convert_html_file_to_md, is_html_file

            if is_html_file(path_obj):
                md_dir = path_obj.parent / "_html_as_md"
                effective_path = convert_html_file_to_md(path_obj, md_dir)
                logger.info(
                    "insert: HTML 预处理 %s → %s", path_obj.name, effective_path.name
                )
        except Exception as html_convert_error:
            logger.warning(
                "insert: HTML→MD 预处理失败 %s: %s（跳过文件）",
                path_obj.name,
                html_convert_error,
            )
            return False

        # 纯文本快速路径：直接喂给 LightRAG 底层的 ainsert
        if effective_path.suffix.lower() in self._PLAINTEXT_EXTS:
            return await self._insert_plaintext(
                engine=engine,
                effective_path=effective_path,
                original_path=path_obj,
                source=source,
            )

        # 其它格式：MinerU 解析路径
        try:
            # 强制走 pipeline 后端：我们只下了 PDF-Extract-Kit（pipeline 模型），
            # 没下 VLM 模型；MinerU 3.x 默认 backend=hybrid-auto-engine 会要 VLM
            await engine.process_document_complete(
                file_path=str(effective_path),
                output_dir=str(effective_path.parent / "_parsed"),
                backend="pipeline",
                lang=self.rag_cfg.get("parser_lang", "en"),
            )
            logger.info(f"已索引 {file_path} → source={source} (mineru)")
            return True
        except Exception as insert_error:
            logger.error(f"索引失败 {file_path}: {insert_error}", exc_info=True)
            return False

    async def _insert_plaintext(
        self,
        *,
        engine: Any,
        effective_path: Path,
        original_path: Path,
        source: str,
    ) -> bool:
        """纯文本快速入库：读文件 → LightRAG.ainsert(text)。

        绕过 RAGAnything 的 process_document_complete，因为后者即使对 md/txt
        也会先走 ReportLab 转成 PDF、再启一个 MinerU FastAPI 子进程做 LayoutLM+OCR。
        在当前 GPU driver 太旧 fallback 到 CPU 的机器上，这个链路慢得不可用，
        且对纯文本文件毫无信息增益（文本已经是 markdown 了）。

        朴素 RAG 模式下，LightRAG 的实体抽取 / 合并已被 monkey-patch 为 no-op，
        所以 ainsert 实际只会做 enqueue → chunk → embedding → vdb_chunks 落盘。
        """
        # 拿到底层 LightRAG 实例。RAGAnything 包装层叫 self.lightrag
        lightrag_instance = getattr(engine, "lightrag", None) or engine
        if not hasattr(lightrag_instance, "ainsert"):
            logger.error(
                "_insert_plaintext: 底层对象没有 ainsert 方法，降级到 MinerU 路径"
            )
            try:
                await engine.process_document_complete(
                    file_path=str(effective_path),
                    output_dir=str(effective_path.parent / "_parsed"),
                    backend="pipeline",
                    lang=self.rag_cfg.get("parser_lang", "en"),
                )
                return True
            except Exception as fallback_error:
                logger.error(f"_insert_plaintext 降级失败: {fallback_error}")
                return False

        try:
            text = effective_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as read_error:
            logger.error(
                "_insert_plaintext: 读取 %s 失败: %s", effective_path, read_error
            )
            return False

        if not text.strip():
            logger.warning(
                "_insert_plaintext: %s 内容为空，跳过", effective_path.name
            )
            return False

        try:
            await lightrag_instance.ainsert(
                input=text,
                file_paths=str(original_path),
            )
            logger.info(
                "已索引 %s → source=%s (plaintext, %d bytes)",
                original_path,
                source,
                len(text),
            )
            return True
        except Exception as insert_error:
            logger.error(
                "_insert_plaintext 失败 %s: %s",
                original_path,
                insert_error,
                exc_info=True,
            )
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

    # ── Query Rewriter ──
    def _build_query_rewriter(self) -> Optional[Any]:
        """按 rag.yaml 的 retrieval.query_rewrite 配置构造 MultiQueryRewriter。

        enabled=false 或构造失败时返回 None，调用方用 ``get_query_rewriter()`` 时
        为 None 即表示"没有启用改写，直接用原 query"。
        """
        rewrite_cfg: Dict[str, Any] = self.retrieval_cfg.get("query_rewrite") or {}
        if not rewrite_cfg.get("enabled", False):
            logger.info("RAGManager: query_rewrite.enabled=false，跳过 Rewriter 构造")
            return None

        # 延迟导入，避免 rag 包未装时 import 期就炸
        try:
            from .rag.query_rewriter import MultiQueryRewriter
        except ImportError as import_error:
            logger.warning(
                "RAGManager: 无法 import MultiQueryRewriter（%s），禁用改写",
                import_error,
            )
            return None

        try:
            rewriter = MultiQueryRewriter(
                llm_model_id=str(
                    rewrite_cfg.get("llm_model_id", "deepseek_v4_flash")
                ),
                num_queries=int(rewrite_cfg.get("num_queries", 3)),
                cache_size=int(rewrite_cfg.get("cache_size", 256)),
                enabled=True,
                temperature=float(rewrite_cfg.get("temperature", 0.2)),
                max_tokens=int(rewrite_cfg.get("max_tokens", 400)),
            )
            logger.info("RAGManager: MultiQueryRewriter 已就绪")
            return rewriter
        except Exception as build_error:
            logger.warning(
                "RAGManager: 构造 MultiQueryRewriter 失败，禁用改写: %s", build_error
            )
            return None

    def get_query_rewriter(self) -> Optional[Any]:
        """返回 MultiQueryRewriter（可能为 None，表示未启用）。"""
        return self._query_rewriter

    async def retrieve_with_rewrite(
        self,
        query: str,
        source: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """改写增强的检索：按 rag.yaml 配置决定是否走 rewriter。

        - 若 rewriter 可用：多路并发 retrieve 后合并文本
        - 若不可用（未启用 / 构造失败）：退化为 ``retrieve(query, source, mode)``
        """
        if not self.enabled:
            return ""
        if not query:
            return ""

        rewriter = self._query_rewriter
        if rewriter is None:
            # 未启用改写，直接单路检索
            return await self.retrieve(query, source=source, mode=mode)

        # 每一路共享同一 source + mode
        target = source or self._default_source()
        if target is None:
            logger.debug("retrieve_with_rewrite: no source available")
            return ""

        rewrite_cfg: Dict[str, Any] = self.retrieval_cfg.get("query_rewrite") or {}
        max_per_section = int(rewrite_cfg.get("max_per_section_chars", 2500))

        async def _one_path(sub_query: str) -> str:
            # 复用 ``retrieve`` 的异常处理 / mode 兜底逻辑
            return await self.retrieve(sub_query, source=target, mode=mode)

        return await rewriter.retrieve_with_rewrite(
            _one_path,
            query,
            domain=target,
            max_per_section_chars=max_per_section,
        )
