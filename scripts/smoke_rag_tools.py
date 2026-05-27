"""RAG 工具冒烟测试：按 source 逐个调用 4 个 LangChain Tool 的 aquery。

用法：
    # 测所有已索引的 source（自动跳过没索引的）
    python scripts/smoke_rag_tools.py

    # 只测指定 source
    python scripts/smoke_rag_tools.py --sources glossary doctrines

    # 覆盖默认查询
    python scripts/smoke_rag_tools.py --query "What is OODA loop?"

输出：每个工具调用结果的前 400 字符 + 总体成功/失败统计。
退出码 0=全部通过，1=有工具返回空/异常。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List

# 把项目根加入 sys.path，兼容直接 python scripts/xxx.py 调用
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from missionorch_lc.core.rag_manager import RAGManager  # noqa: E402
from missionorch_lc.tools.rag_tools import build_rag_tools  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_rag")
# LightRAG/raganything 的 INFO 日志太吵，压到 WARNING
for noisy in ("lightrag", "nano-vectordb", "raganything"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# 每个 source 的默认测试 query（贴近真实 Agent 场景）
DEFAULT_QUERIES: Dict[str, str] = {
    "doctrines": "What are the key principles of air campaign planning?",
    "maps": "Describe the terrain features mentioned in the maps.",
    "historical": "What lessons can be drawn from past air campaigns?",
    "glossary": "What does OODA loop mean?",
}


def _has_indexed_data(source: str) -> bool:
    """判断 source 是否已有索引数据（避免对空库发起无意义查询）。"""
    storage_dir = PROJECT_ROOT / "knowledge_base" / "rag_storage" / source
    if not storage_dir.exists():
        return False
    # 有实体向量库就算已索引
    vdb = storage_dir / "vdb_entities.json"
    if not vdb.exists():
        return False
    try:
        return vdb.stat().st_size > 200  # 空 json 也就 ~100B
    except Exception:
        return False


async def test_one_tool(tool, query: str) -> Dict[str, object]:
    """调一个 tool 的 _arun，返回结构化结果。"""
    name = tool.name
    source = getattr(tool, "source_name", "?")
    logger.info(f"▶ 测试 {name} (source={source})  query={query!r}")
    try:
        text = await tool._arun(query)
    except Exception as tool_error:
        logger.error(f"✗ {name} 异常: {tool_error}", exc_info=True)
        return {"name": name, "ok": False, "reason": f"exception: {tool_error}", "text": ""}

    if not text or text.startswith("[未在") or text.startswith("[RAG ") or text.startswith("[知识源"):
        logger.warning(f"✗ {name} 返回空/错误: {text[:200]}")
        return {"name": name, "ok": False, "reason": text[:200], "text": text}

    preview = text[:400].replace("\n", " ⏎ ")
    logger.info(f"✓ {name} 命中 {len(text)} 字符，预览: {preview} ...")
    return {"name": name, "ok": True, "reason": "", "text": text}


async def main_async(sources_filter: List[str] | None, query_override: str | None) -> int:
    manager = RAGManager()
    if not manager.is_enabled():
        logger.error("RAGManager 未启用（rag.enabled=False 或 raganything 未安装），无法测试")
        return 2

    all_sources = manager.list_sources()
    logger.info(f"RAGManager 已配置 sources: {all_sources}")

    # 过滤：只测有索引数据的 source
    candidate = sources_filter or all_sources
    targets = [s for s in candidate if s in all_sources and _has_indexed_data(s)]
    skipped = [s for s in candidate if s in all_sources and not _has_indexed_data(s)]
    if skipped:
        logger.warning(f"跳过未索引的 source: {skipped}")
    if not targets:
        logger.error("没有可测试的 source（都没索引）")
        return 2

    logger.info(f"本次测试的 sources: {targets}")

    tools = build_rag_tools(rag_manager=manager, enabled_sources=targets)
    if not tools:
        logger.error("build_rag_tools 返回空列表")
        return 2

    results: List[Dict[str, object]] = []
    for tool in tools:
        source = getattr(tool, "source_name", "")
        query = query_override or DEFAULT_QUERIES.get(source, "Give an overview.")
        result = await test_one_tool(tool, query)
        results.append(result)

    # 汇总
    print("\n" + "=" * 72)
    print("RAG Tools Smoke Test Summary")
    print("=" * 72)
    passed = 0
    for result in results:
        status = "✅ PASS" if result["ok"] else "❌ FAIL"
        extra = "" if result["ok"] else f"  reason={result['reason'][:120]}"
        print(f"  {status}  {result['name']}{extra}")
        if result["ok"]:
            passed += 1
    print(f"\nTotal: {passed}/{len(results)} passed")
    print("=" * 72)

    return 0 if passed == len(results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 4 工具冒烟测试")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["doctrines", "maps", "historical", "glossary"],
        help="只测指定 source（默认：所有已索引的）",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="覆盖所有工具的默认查询",
    )
    args = parser.parse_args()
    exit_code = asyncio.run(main_async(args.sources, args.query))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
