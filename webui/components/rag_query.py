"""RAG 直查封装 —— 给 Streamlit 用的同步入口。

特性：
- ``query_with_rewrite``: 同步调用 ``RAGManager.retrieve_with_rewrite``
- 同时返回原 query 与改写后的子 queries（如果 rewriter 启用）
- 把多路检索拆成结构化结果，方便前端按子 query 分组展示
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGQueryResult:
    """RAG 单次查询的完整结果。"""

    query: str
    source: str
    mode: str
    raw_text: str = ""                            # retrieve_with_rewrite 拼好的整段
    sub_queries: List[str] = field(default_factory=list)
    sub_sections: List[Dict[str, Any]] = field(default_factory=list)  # [{sub_query, body}]
    elapsed_seconds: float = 0.0
    rewrite_enabled: bool = False
    error: Optional[str] = None


def _split_by_subqueries(raw_text: str) -> List[Dict[str, str]]:
    """把 retrieve_with_rewrite 拼装的文本按 ``### Sub-query: xxx`` 切回若干段。

    输入示例（来自 query_rewriter._merge_sections）::

        > Retrieved via multi-query rewrite (3/3 sub-queries hit)

        ### Sub-query: three phases of SEAD
        ...正文...

        ### Sub-query: SEAD execution in Desert Storm
        ...正文...
    """
    if not raw_text:
        return []
    # 用 lookbehind 切分；保留每段标题
    pattern = re.compile(r"(?m)^### Sub-query:\s*(?P<title>.+?)\s*$")
    matches = list(pattern.finditer(raw_text))
    if not matches:
        # 没改写过，整段返回
        return [{"sub_query": "", "body": raw_text.strip()}]
    sections: List[Dict[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        sections.append({"sub_query": title, "body": body})
    return sections


def _run_async(coro):
    """跨线程跑 asyncio 协程：Streamlit 主线程没有 running loop，每次都开一个新 loop。

    注：Streamlit 是同步脚本模型，每次脚本 rerun 都是干净的栈，不会嵌在已有
    event loop 里，所以直接 ``new_event_loop`` 就行；无需先探测 ``get_event_loop``。
    """
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def query_with_rewrite(
    *,
    rag_manager: Any,
    query: str,
    source: Optional[str] = None,
    mode: Optional[str] = None,
) -> RAGQueryResult:
    """对外主入口：同步调用 RAG 检索（自动启用 Multi-Query 改写）。"""
    import time

    used_source = source or (rag_manager.list_sources() or [None])[0]
    if used_source is None:
        return RAGQueryResult(
            query=query,
            source="",
            mode=mode or "",
            error="RAG 没有可用 source（rag.knowledge_sources 为空或未启用）",
        )
    used_mode = mode or rag_manager.retrieval_cfg.get("search_mode", "naive")

    rewrite_enabled = rag_manager.get_query_rewriter() is not None

    started = time.time()
    try:
        text = _run_async(
            rag_manager.retrieve_with_rewrite(query, source=used_source, mode=used_mode)
        )
    except Exception as run_error:
        logger.exception("rag_query: 检索失败: %s", run_error)
        return RAGQueryResult(
            query=query,
            source=used_source,
            mode=used_mode,
            rewrite_enabled=rewrite_enabled,
            error=str(run_error),
            elapsed_seconds=time.time() - started,
        )
    elapsed = time.time() - started

    sections = _split_by_subqueries(text)
    sub_queries = [section["sub_query"] for section in sections if section["sub_query"]]

    return RAGQueryResult(
        query=query,
        source=used_source,
        mode=used_mode,
        raw_text=text or "",
        sub_queries=sub_queries,
        sub_sections=sections,
        elapsed_seconds=elapsed,
        rewrite_enabled=rewrite_enabled,
    )


def get_or_create_rag_manager(
    config_path: str = "config/rag.yaml",
    cached_singleton: Optional[Any] = None,
) -> Any:
    """复用进程内 RAGManager 单例。Streamlit 每次脚本重跑都会丢全局对象，
    所以我们把它绑在 ``session_state`` 里（由调用方传入 / 写回）。
    """
    if cached_singleton is not None:
        return cached_singleton
    from missionorch_lc.core.rag_manager import RAGManager

    return RAGManager(config_path)
