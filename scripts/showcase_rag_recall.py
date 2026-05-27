"""展示当前 RAG 能召回什么的实测脚本。

针对当前 5 个 source 的实际语料，跑几条目标 query 看每个 source 真正能贡献的内容。
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from missionorch_lc.core.rag_manager import RAGManager  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
for name in (
    "missionorch_lc.core.rag.query_rewriter",
    "missionorch_lc.core.rag_manager",
):
    logging.getLogger(name).setLevel(logging.INFO)

# (query, source, 期望关键词)
SHOWCASES = [
    ("作战原则中的『集中』和『目标』分别强调什么？", "doctrines",
     ["mass", "objective", "concentrat", "principle"]),
    ("空中战役规划的关键步骤是什么？", "doctrines",
     ["air campaign", "planning", "step", "phase"]),
    ("EA-18G Growler 的核心任务", "tmp_plain",
     ["ea-18g", "growler", "jamm", "electronic"]),
    ("SEAD 的三个阶段 suppress / deceive / strike", "tmp_plain",
     ["suppress", "deceive", "strike"]),
    ("沙漠风暴行动里的 SEAD 战法", "tmp_plain",
     ["desert storm", "wild weasel", "harm", "ef-111", "ea-6b"]),
    ("SEAD 是什么意思？", "glossary",
     ["sead", "suppress", "air defense"]),
    ("HARM 反辐射导弹的定义", "glossary",
     ["harm", "anti-radiation", "missile", "agm-88"]),
]


async def _amain() -> None:
    mgr = RAGManager()
    print(f"\n[mgr] enabled={mgr.is_enabled()}  sources={mgr.list_sources()}\n")

    for idx, (q, src, keywords) in enumerate(SHOWCASES, 1):
        print(f"\n{'═' * 80}")
        print(f"[{idx}/{len(SHOWCASES)}] source={src!r}")
        print(f"  query: {q}")
        print("─" * 80)
        try:
            text = await mgr.retrieve_with_rewrite(q, source=src, mode="naive")
        except Exception as exc:
            print(f"  ❌ 检索失败: {exc}")
            continue

        low = (text or "").lower()
        hits = [k for k in keywords if k in low]
        ok = bool(hits) and len(text) > 200
        mark = "✅ HIT" if ok else "❌ MISS"
        print(f"  {mark}   len={len(text)}  hit_keywords={hits}")
        if text:
            snippet = text[:600].replace("\n", " ")
            print(f"  snippet (前 600 字):\n    {snippet}")


if __name__ == "__main__":
    asyncio.run(_amain())
