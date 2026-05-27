"""朴素 RAG + Multi-Query 检索冒烟脚本。

用法：
    PYTHONPATH=src python scripts/smoke_naive_retrieve.py [--source tmp_plain]

成功判据：每条中文 query 都能从英文语料里召回（snippet 里至少命中一个关键词、
长度 > 200），同时日志里可以肉眼看到：
  - LightRAG 查询走 mode="naive"
  - MultiQueryRewriter 生成了 3 条互补 query
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from missionorch_lc.core.rag_manager import RAGManager  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,  # 基础设施噪声太大，只保留关键业务日志
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 业务关键 logger 升到 INFO
for name in (
    "missionorch_lc.core.rag.query_rewriter",
    "missionorch_lc.core.rag_manager",
    "missionorch_lc.core.rag.lightrag_factory",
):
    logging.getLogger(name).setLevel(logging.INFO)

QUERIES = [
    (
        "SEAD 的三个阶段是什么？",
        ["suppress", "deceive", "strike", "phase"],
    ),
    (
        "EA-18G Growler 用来干什么？",
        ["ea-18g", "growler", "jamm", "electronic"],
    ),
    (
        "沙漠风暴行动里 SEAD 是怎么做的？",
        ["desert storm", "wild weasel", "harm", "ef-111", "ea-6b"],
    ),
]


async def _amain(source: str) -> int:
    mgr = RAGManager()
    print(f"\n[mgr] enabled={mgr.is_enabled()}  sources={mgr.list_sources()}")
    if source not in mgr.list_sources():
        print(f"[错误] source {source!r} 未配置；可用: {mgr.list_sources()}")
        return 2

    passed = 0
    for q, keywords in QUERIES:
        text = await mgr.retrieve_with_rewrite(q, source=source, mode="naive")
        low = (text or "").lower()
        hits = [k for k in keywords if k in low]
        ok = bool(hits) and len(text) > 200
        mark = "✅" if ok else "❌"
        print(f"\n{mark} query={q!r}")
        print(f"   len={len(text)}  hit_keywords={hits}")
        if text:
            snippet = text[:260].replace("\n", " ")
            print(f"   snippet: {snippet}")
        passed += int(ok)

    print(f"\n==== 命中率: {passed}/{len(QUERIES)} ====")
    return 0 if passed == len(QUERIES) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="tmp_plain")
    args = parser.parse_args()
    sys.exit(asyncio.run(_amain(args.source)))


if __name__ == "__main__":
    main()
