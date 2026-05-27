"""Query Rewrite 端到端冒烟测试。

目的：验证"中文 query 经过 Multi-Query 改写后，能命中英文 / 混合语料"。

前置：
- glossary 已有索引（knowledge_base/rag_storage/glossary/ 非空）
- config/rag.yaml 的 retrieval.query_rewrite.enabled=true（默认）
- DeepSeek API key 已配置

输出：
- 先打印 rewriter 生成的 3 条子 query（验证改写逻辑）
- 再打印 rag_glossary_search 工具的最终返回（验证端到端检索）
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from missionorch_lc.core.rag_manager import RAGManager  # noqa: E402
from missionorch_lc.tools.rag_tools import build_rag_tools  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_rewrite")
for noisy in ("lightrag", "nano-vectordb", "raganything", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


TEST_QUERIES = [
    # (测试说明, query)
    ("纯英文 baseline", "What is Course of Action (COA)?"),
    ("纯中文 → 需要跨语言改写", "行动方案 COA 的定义是什么"),
    ("中文近战术语", "近距空中支援的含义"),
]


def _short(text: str, limit: int = 400) -> str:
    """把长文本截成一行预览。"""
    flat = " ".join(text.split())
    return flat[:limit] + (" ..." if len(flat) > limit else "")


async def run() -> int:
    manager = RAGManager()
    if not manager.is_enabled():
        logger.error("RAGManager 未启用；跳过")
        return 2

    if "glossary" not in manager.list_sources():
        logger.error("glossary source 未配置")
        return 2

    rewriter = manager.get_query_rewriter()
    if rewriter is None:
        logger.error("query_rewriter 未启用（检查 config/rag.yaml retrieval.query_rewrite.enabled）")
        return 2

    logger.info("Query rewriter 就绪: %s", rewriter)

    # ── Part 1：单独验证 rewriter.rewrite() 的改写能力 ──
    print("\n" + "=" * 72)
    print("Part 1: Query Rewriter 改写输出")
    print("=" * 72)
    for label, query in TEST_QUERIES:
        print(f"\n[{label}]")
        print(f"  原始: {query}")
        rewritten = await rewriter.rewrite(query, domain="glossary")
        for i, sub in enumerate(rewritten, 1):
            print(f"  {i}. {sub}")

    # ── Part 2：走完整 tool._arun（含 retrieve_with_rewrite + source 去重）──
    print("\n" + "=" * 72)
    print("Part 2: rag_glossary_search 工具端到端检索")
    print("=" * 72)
    tools = build_rag_tools(rag_manager=manager, enabled_sources=["glossary"])
    if not tools:
        logger.error("build_rag_tools 返回空")
        return 2
    glossary_tool = tools[0]

    pass_count = 0
    for label, query in TEST_QUERIES:
        print(f"\n[{label}]  query={query!r}")
        try:
            result = await glossary_tool._arun(query)
        except Exception as tool_error:
            print(f"  ❌ 异常: {tool_error}")
            continue

        if not result or result.startswith("[未在") or result.startswith("[RAG "):
            print(f"  ❌ 未命中 / 错误: {_short(result, 200)}")
            continue

        # 检查是否有多路 sub-query header（说明 rewriter 真的在工作）
        has_rewrite_marker = "multi-query rewrite" in result.lower() or "Sub-query:" in result
        print(f"  ✅ 命中 {len(result)} 字符, 多路改写生效={has_rewrite_marker}")
        print(f"  预览: {_short(result, 500)}")
        pass_count += 1

    print("\n" + "=" * 72)
    print(f"Summary: {pass_count}/{len(TEST_QUERIES)} 测试命中")
    print("=" * 72)
    return 0 if pass_count == len(TEST_QUERIES) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
