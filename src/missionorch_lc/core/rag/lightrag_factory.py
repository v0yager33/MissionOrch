"""RAGAnything 工厂：把本地 embedding/rerank + ModelRouter LLM 装配成一个完整实例。

这是修复原项目 RAG 用法不正确（缺 llm_model_func / embedding_func）的关键模块。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .local_models import build_local_embedding_func, build_local_rerank_func

logger = logging.getLogger(__name__)


def _make_llm_callable_from_router(model_id: str) -> Callable:
    """把 `ModelRouter` 暴露的 LangChain ChatModel 包装成
    LightRAG 期望的 `llm_model_func` 签名：

        async def llm(prompt, system_prompt=None, history_messages=[], **kwargs) -> str
    """
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )

    from ..model_router import ModelRouter

    raw_model = ModelRouter.get_raw(model_id)

    async def _llm(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        for hm in history_messages or []:
            role = hm.get("role", "user")
            content = hm.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
        messages.append(HumanMessage(content=prompt))

        # 把 LightRAG 透传过来的 hashing_kv 之类的东西丢掉，
        # 它们只对 LightRAG 自己有用，传到 ChatModel 会报错
        kwargs.pop("hashing_kv", None)
        kwargs.pop("history_messages", None)

        # ChatModel.ainvoke 不接受 max_tokens / temperature 等任意 kwarg；
        # 用 .bind 转换
        bound = raw_model
        bind_kwargs: Dict[str, Any] = {}
        for k in ("temperature", "max_tokens", "top_p"):
            if k in kwargs and kwargs[k] is not None:
                bind_kwargs[k] = kwargs.pop(k)
        if bind_kwargs:
            bound = raw_model.bind(**bind_kwargs)

        message = await bound.ainvoke(messages)
        return message.content if hasattr(message, "content") else str(message)

    return _llm


def _make_vision_callable_from_router(model_id: str) -> Callable:
    """与 `_make_llm_callable_from_router` 类似，但接收 image_data / messages。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    from ..model_router import ModelRouter

    raw_model = ModelRouter.get_raw(model_id)

    async def _vlm(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        image_data: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        # 优先使用调用方组装好的 OpenAI-style messages
        if messages:
            # 已经是 OpenAI 格式，直接走 raw model 的同步底层
            from langchain_core.messages import AIMessage

            converted = []
            for m in messages:
                if not m:
                    continue
                role = m.get("role")
                content = m.get("content")
                if role == "system":
                    converted.append(SystemMessage(content=content or ""))
                elif role == "user":
                    converted.append(HumanMessage(content=content))
                elif role == "assistant":
                    converted.append(AIMessage(content=content))
            response = await raw_model.ainvoke(converted)
            return response.content if hasattr(response, "content") else str(response)

        if image_data:
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
            converted = []
            if system_prompt:
                converted.append(SystemMessage(content=system_prompt))
            converted.append(HumanMessage(content=user_content))
            response = await raw_model.ainvoke(converted)
            return response.content if hasattr(response, "content") else str(response)

        # 纯文本 → 退化为普通 LLM
        return await _make_llm_callable_from_router(model_id)(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    return _vlm


_NAIVE_RAG_PATCHED = False


def _patch_lightrag_for_naive_rag() -> None:
    """把 LightRAG 索引流水线里所有 LLM 驱动的图构建步骤都关掉。

    朴素 RAG 场景下，我们只要 parse → chunk → embedding → 向量检索，
    完全不需要实体 / 关系 / 摘要 / 图合并。关掉之后：
      - 索引速度提升 10x 以上（不再逐 chunk 调 LLM）
      - 索引 token 花费降为 0
      - 检索只能用 mode="naive"（已在 rag.yaml 默认）
      - PDF/DOCX/HTML/MD/TXT 解析路径不受影响（还是 MinerU）

    为什么不重写 LightRAG？
      RAGAnything 的 aquery 和整个 ensure_initialized 都深度依赖 LightRAG 实例，
      重写成本高、风险大。保留 LightRAG 实例、把图构建步骤 no-op 化最小侵入。

    打哪几个点？（基于 lightrag 源码审查）：
      1. ``LightRAG._process_extract_entities``：实体/关系抽取总入口，返回 [] 即跳过合并
      2. ``lightrag.operate.extract_entities``：底层实现，兜底也换成 no-op
      3. ``lightrag.operate.merge_nodes_and_edges``：只处理 chunk_results；给空 list 是 no-op，
         但为了防止有别处绕过 _process_extract_entities 直接调它，也包一层 no-op
      4. ``entity_extract_max_gleaning=0``：在 build_rag_anything 里通过 lightrag_kwargs 传

    幂等：模块级标志位保证只 patch 一次，避免重复包装。
    """
    global _NAIVE_RAG_PATCHED
    if _NAIVE_RAG_PATCHED:
        return
    try:
        from lightrag import operate as lightrag_operate  # type: ignore
        from lightrag.lightrag import LightRAG  # type: ignore
    except ImportError:
        logger.warning("LightRAG 未安装，无法 patch 为朴素 RAG 模式")
        return

    # ── Patch #1：LightRAG._process_extract_entities（实例方法） ──
    async def _noop_process_extract_entities(
        self,
        chunk: Dict[str, Any],
        pipeline_status=None,
        pipeline_status_lock=None,
    ) -> list:
        n = len(chunk) if isinstance(chunk, dict) else 0
        logger.info("[naive_rag] skip extract_entities: %d chunks", n)
        return []

    LightRAG._process_extract_entities = _noop_process_extract_entities  # type: ignore[method-assign]

    # ── Patch #2：operate.extract_entities（模块级函数） ──
    # 万一未来有别处直接从 operate import 调用，兜底
    async def _noop_extract_entities(*args: Any, **kwargs: Any) -> list:
        return []

    if hasattr(lightrag_operate, "extract_entities"):
        lightrag_operate.extract_entities = _noop_extract_entities  # type: ignore[assignment]

    # ── Patch #3：operate.merge_nodes_and_edges（模块级函数） ──
    # 签名：(chunk_results, knowledge_graph_inst, entity_vdb, relationships_vdb,
    #        global_config, ...) -> None
    # 给空 chunk_results 时本身就是 no-op；但为了防止调用方传进来带内容的列表
    # （比如缓存命中），再显式包一层 no-op，保证朴素模式下图库一定是空的
    async def _noop_merge_nodes_and_edges(*args: Any, **kwargs: Any) -> None:
        return None

    if hasattr(lightrag_operate, "merge_nodes_and_edges"):
        lightrag_operate.merge_nodes_and_edges = _noop_merge_nodes_and_edges  # type: ignore[assignment]

    # ── Patch #4：同步 lightrag.lightrag 模块级 import 引用 ──
    # lightrag.py 顶部 ``from lightrag.operate import (..., merge_nodes_and_edges, ...)``
    # 把名字绑定到自己模块命名空间里了。单独改 operate 模块不够，要同步改 lightrag 模块。
    try:
        from lightrag import lightrag as lightrag_module  # type: ignore

        if hasattr(lightrag_module, "extract_entities"):
            lightrag_module.extract_entities = _noop_extract_entities  # type: ignore[attr-defined]
        if hasattr(lightrag_module, "merge_nodes_and_edges"):
            lightrag_module.merge_nodes_and_edges = _noop_merge_nodes_and_edges  # type: ignore[attr-defined]
    except ImportError:
        pass

    _NAIVE_RAG_PATCHED = True
    logger.info(
        "朴素 RAG 补丁已生效：LightRAG._process_extract_entities / "
        "operate.extract_entities / operate.merge_nodes_and_edges 全部 no-op"
    )


# 模块 import 时就立即 patch，确保在任何 LightRAG 实例化之前生效
# （之前放在 build_rag_anything 里会导致同进程第一次构建之前的代码已经拿到了原函数引用）
_patch_lightrag_for_naive_rag()


# 旧函数名 alias，兼容外部调用点
def _patch_lightrag_skip_entity_extraction() -> None:  # pragma: no cover
    _patch_lightrag_for_naive_rag()


def build_rag_anything(
    *,
    working_dir: str,
    embedding_model_path: str,
    llm_model_id: str,
    rerank_model_path: Optional[str] = None,
    vision_model_id: Optional[str] = None,
    parser: str = "mineru",
    enable_image_processing: bool = True,
    enable_table_processing: bool = True,
    enable_equation_processing: bool = False,
    embedding_device: str = "auto",
    rerank_device: str = "auto",
    embedding_max_length: int = 8192,
    embedding_batch_size: int = 8,
    embedding_instruction: Optional[str] = None,
    lightrag_kwargs: Optional[Dict[str, Any]] = None,
    skip_entity_extraction: bool = True,
) -> Any:
    """构建一个完全本地化的 RAGAnything 实例。

    Args:
        working_dir: LightRAG 持久化目录
        embedding_model_path: 本地 Qwen3-Embedding 路径
        llm_model_id: ModelRouter 中的 LLM ID（如 ``seed_doubao``）
        rerank_model_path: 本地 Qwen3-Reranker 路径，None 则不启用 rerank
        vision_model_id: ModelRouter 中的 VLM ID，None 则复用 llm
        parser / enable_*: 透传给 RAGAnythingConfig
        embedding_instruction: query 编码时的 instruction 前缀（None 走默认）
        lightrag_kwargs: 透传给 LightRAG（如 ``top_k`` / ``chunk_top_k`` /
            ``rerank_model_func`` 等）
    """
    try:
        from raganything import RAGAnything, RAGAnythingConfig  # type: ignore
    except ImportError as ie:
        raise ImportError(
            "RAG-Anything 未安装。请按 docs/RAG_GUIDE.md 安装 raganything 与 lightrag-hku"
        ) from ie

    # ── 朴素 RAG 模式：双保险再 patch 一次 ──
    # 注：模块 import 时已经 patch 过一次；这里再调一次是幂等的，只是为了防御
    # 『用户自己 reimport lightrag 或热重载』这种极端情况。
    if skip_entity_extraction:
        _patch_lightrag_for_naive_rag()

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        enable_image_processing=enable_image_processing,
        enable_table_processing=enable_table_processing,
        enable_equation_processing=enable_equation_processing,
    )

    # ── 朴素 RAG：把 gleaning 轮数降到 0（多做也没意义，实体抽取本身已 no-op） ──
    extra_lightrag_kwargs_early: Dict[str, Any] = dict(lightrag_kwargs or {})
    extra_lightrag_kwargs_early.setdefault("entity_extract_max_gleaning", 0)
    lightrag_kwargs = extra_lightrag_kwargs_early

    # ── 构建必填回调 ──
    embedding_func = build_local_embedding_func(
        embedding_model_path,
        device=embedding_device,
        max_length=embedding_max_length,
        batch_size=embedding_batch_size,
        instruction=embedding_instruction,
    )
    llm_model_func = _make_llm_callable_from_router(llm_model_id)
    vision_model_func = (
        _make_vision_callable_from_router(vision_model_id)
        if vision_model_id
        else None
    )

    # ── 可选 rerank ──
    extra_lightrag_kwargs: Dict[str, Any] = dict(lightrag_kwargs or {})
    if rerank_model_path:
        rerank_func = build_local_rerank_func(rerank_model_path, device=rerank_device)
        extra_lightrag_kwargs.setdefault("rerank_model_func", rerank_func)
        logger.info("Rerank function registered with LightRAG")

    # raganything 的 RAGAnything 构造函数接收 lightrag_kwargs 来透传给底层 LightRAG
    # 不同版本签名可能差异 → 先尝试带 lightrag_kwargs，失败则降级
    try:
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
            lightrag_kwargs=extra_lightrag_kwargs,
        )
    except TypeError as type_error:
        logger.warning(
            f"RAGAnything 不接受 lightrag_kwargs 参数，降级构造: {type_error}"
        )
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

    logger.info(
        "RAGAnything assembled: working_dir=%s, llm=%s, rerank=%s, vision=%s",
        working_dir,
        llm_model_id,
        bool(rerank_model_path),
        bool(vision_model_id),
    )
    return rag


async def ensure_initialized(rag: Any) -> None:
    """触发 RAGAnything 的 LightRAG 内部初始化（建表、加载向量库）。

    RAGAnything 在 `aquery` / `ainsert` 时会自动调一次 `_ensure_lightrag_initialized`，
    我们也提供一个显式入口，方便启动时预热。
    """
    if hasattr(rag, "_ensure_lightrag_initialized"):
        result = await rag._ensure_lightrag_initialized()
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(
                f"LightRAG initialization failed: {result.get('error')}"
            )
