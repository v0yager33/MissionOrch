"""RAGAnything 工厂：把本地 embedding/rerank + ModelRouter LLM 装配成一个完整实例。

这是修复原项目 RAG 用法不正确（缺 llm_model_func / embedding_func）的关键模块。
"""

from __future__ import annotations

import asyncio
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
    from ..model_router import ModelRouter
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )

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
    from ..model_router import ModelRouter
    from langchain_core.messages import HumanMessage, SystemMessage

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
    lightrag_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """构建一个完全本地化的 RAGAnything 实例。

    Args:
        working_dir: LightRAG 持久化目录
        embedding_model_path: 本地 Qwen3-Embedding 路径
        llm_model_id: ModelRouter 中的 LLM ID（如 ``seed_doubao``）
        rerank_model_path: 本地 Qwen3-Reranker 路径，None 则不启用 rerank
        vision_model_id: ModelRouter 中的 VLM ID，None 则复用 llm
        parser / enable_*: 透传给 RAGAnythingConfig
        lightrag_kwargs: 透传给 LightRAG（如 ``top_k`` / ``chunk_top_k`` 等）
    """
    try:
        from raganything import RAGAnything, RAGAnythingConfig  # type: ignore
    except ImportError as ie:
        raise ImportError(
            "RAG-Anything 未安装。请按 docs/RAG_GUIDE.md 安装 raganything 与 lightrag-hku"
        ) from ie

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        enable_image_processing=enable_image_processing,
        enable_table_processing=enable_table_processing,
        enable_equation_processing=enable_equation_processing,
    )

    # ── 构建必填回调 ──
    embedding_func = build_local_embedding_func(
        embedding_model_path,
        device=embedding_device,
        max_length=embedding_max_length,
        batch_size=embedding_batch_size,
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

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        lightrag_kwargs=extra_lightrag_kwargs,
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
