"""一次性索引知识库脚本。

用法：

    # 索引所有 source（rag.yaml 中配置的）
    python scripts/index_knowledge_base.py

    # 只索引一个 source
    python scripts/index_knowledge_base.py --source doctrines

    # 强制重建（忽略已索引文件名 hash）
    python scripts/index_knowledge_base.py --force

实现要点：
- 复用 RAGManager 中已经构建好的 RAGAnything 引擎，每个 source 独立 working_dir
- 用 source 子目录下的 `.indexed_files.txt` 维护已索引文件名 → 简单去重
- 大文件解析失败不会终止整个流程，只打 warning 继续
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from typing import List, Set

# 让脚本可以直接 `python scripts/xxx.py` 运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from missionorch_lc.core.rag_manager import RAGManager  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index_kb")


def _file_signature(path: Path) -> str:
    """文件名 + size + mtime 做一个轻量签名（避免重新跑 hash 整个大文件）。"""
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _load_indexed(record_file: Path) -> Set[str]:
    if not record_file.exists():
        return set()
    return {line.strip() for line in record_file.read_text().splitlines() if line.strip()}


def _append_indexed(record_file: Path, sig: str) -> None:
    record_file.parent.mkdir(parents=True, exist_ok=True)
    with record_file.open("a") as f:
        f.write(sig + "\n")


def _collect_files(source_dir: Path, file_types: List[str]) -> List[Path]:
    if not source_dir.exists():
        return []
    files: List[Path] = []
    for ext in file_types:
        files.extend(source_dir.rglob(f"*{ext}"))
    return sorted(files)


async def _index_one_source(
    manager: RAGManager,
    source_name: str,
    *,
    force: bool,
) -> None:
    cfg = manager.rag_cfg.get("knowledge_sources", {}).get(source_name)
    if not cfg:
        logger.warning(f"[{source_name}] 未在 rag.yaml 中配置，跳过")
        return

    src_dir = (ROOT / cfg["path"]).resolve()
    file_types = cfg.get("file_types", [])
    if not src_dir.exists():
        logger.warning(f"[{source_name}] 目录不存在: {src_dir}，跳过")
        return

    files = _collect_files(src_dir, file_types)
    if not files:
        logger.info(f"[{source_name}] 目录为空，跳过：{src_dir}")
        return

    # 拿到引擎（首次会触发 _build_engine_for_source）
    try:
        engine = manager._build_engine_for_source(source_name)
    except Exception as build_error:
        logger.error(f"[{source_name}] 构建引擎失败: {build_error}")
        return

    record_file = (
        Path(manager.rag_cfg.get("working_dir", "knowledge_base/rag_storage"))
        / source_name
        / ".indexed_files.txt"
    )
    indexed = set() if force else _load_indexed(record_file)
    logger.info(
        f"[{source_name}] 待索引 {len(files)} 个文件，已索引 {len(indexed)} 个 "
        f"({'force=True' if force else 'incremental'})"
    )

    inserted = 0
    skipped = 0
    failed = 0
    for path in files:
        sig = _file_signature(path)
        if sig in indexed:
            skipped += 1
            continue
        try:
            logger.info(f"[{source_name}] indexing: {path.name}")
            # RAGAnything: process_document_complete 走完整流水线（含解析）
            await engine.process_document_complete(file_path=str(path))
            _append_indexed(record_file, sig)
            inserted += 1
        except Exception as ingest_error:
            logger.warning(
                f"[{source_name}] 索引失败 {path.name}: {ingest_error}（跳过，继续下一个）"
            )
            failed += 1

    logger.info(
        f"[{source_name}] 完成：inserted={inserted}, skipped={skipped}, failed={failed}"
    )


async def _main_async(args: argparse.Namespace) -> int:
    manager = RAGManager(args.config)
    if not manager.is_enabled():
        logger.error("RAG 在 config/rag.yaml 中未启用 (rag.enabled=false)")
        return 1

    all_sources = list(manager.rag_cfg.get("knowledge_sources", {}).keys())
    targets = [args.source] if args.source else all_sources
    if args.source and args.source not in all_sources:
        logger.error(f"source '{args.source}' 不在配置中。可用：{all_sources}")
        return 1

    logger.info(f"开始索引 sources: {targets}")
    for src in targets:
        await _index_one_source(manager, src, force=args.force)
    logger.info("全部索引完成 ✅")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="索引 knowledge_base/ 到 LightRAG")
    parser.add_argument("--config", default="config/rag.yaml", help="rag.yaml 路径")
    parser.add_argument("--source", default=None, help="只索引指定 source（默认全部）")
    parser.add_argument(
        "--force", action="store_true", help="忽略已索引记录，强制重建"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
