"""冒烟索引脚本：绕过 MinerU，自己把 PDF/HTML 转成纯文本直接灌给 LightRAG。

为什么需要这个脚本：
- RAGAnything.process_document_complete 默认走 MinerU，需要下 ~GB 模型且启动很慢
- 在 MinerU 就绪前，我们要先把"embedding + LLM 实体抽取 + 4-tool 检索"的主链路打通
- 所以用 pypdf / bs4 在本地把文档转成纯文本，再走 LightRAG 的 ``ainsert`` 低层接口

用法：
    python scripts/smoke_index_plaintext.py --source doctrines
    python scripts/smoke_index_plaintext.py --source glossary --force

设计要点：
- PDF：pypdf 按页抽取，用页号作为 doc_id 后缀
- HTML：bs4 提取 body 内可见文本，单文件作为一条 doc 插入
- Markdown/Text：原样读入
- 所有插入都打到 LightRAG（RAGAnything 的 .lightrag 属性），复用 RAGManager 构建的引擎
- 简单去重：同一文件用 (filename, size, mtime) 做签名写进 .indexed_plain.txt
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from missionorch_lc.core.rag_manager import RAGManager  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_index")


# ── 文件 → 纯文本 ────────────────────────────────────────────────────


def _extract_pdf_chunks(path: Path) -> List[Tuple[str, str]]:
    """用 pypdf 逐页抽文本，返回 [(doc_id, text), ...]。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks: List[Tuple[str, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as extract_error:
            logger.warning(f"[{path.name}] page {idx} extract failed: {extract_error}")
            continue
        text = text.strip()
        if text:
            chunks.append((f"{path.name}#page={idx}", text))
    return chunks


def _extract_html_text(path: Path) -> str:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    # 去掉脚本/样式
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()
    body = soup.body or soup
    text = body.get_text(separator="\n", strip=True)
    return text


def _extract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _file_to_chunks(path: Path) -> List[Tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_chunks(path)
    if suffix in (".html", ".htm"):
        text = _extract_html_text(path)
        return [(path.name, text)] if text.strip() else []
    if suffix in (".md", ".markdown", ".txt", ".csv"):
        text = _extract_text(path)
        return [(path.name, text)] if text.strip() else []
    logger.warning(f"skip unsupported file: {path}")
    return []


# ── 已索引记录 ────────────────────────────────────────────────────────


def _file_signature(path: Path) -> str:
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


# ── 主流程 ────────────────────────────────────────────────────────────


def _collect_files(source_dir: Path, file_types: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for ext in file_types:
        files.extend(source_dir.rglob(f"*{ext}"))
    return sorted(files)


async def _get_lightrag(engine) -> object:
    """拿到 RAGAnything 内部的 LightRAG 实例（已初始化）。"""
    # 触发 RAGAnything._ensure_lightrag_initialized
    if hasattr(engine, "_ensure_lightrag_initialized"):
        await engine._ensure_lightrag_initialized()
    lightrag_obj = getattr(engine, "lightrag", None)
    if lightrag_obj is None:
        raise RuntimeError("RAGAnything 没有 .lightrag 属性，无法直接走底层 ainsert")
    return lightrag_obj


async def _index_one_source(
    manager: RAGManager,
    source_name: str,
    *,
    force: bool,
) -> None:
    sources_cfg = manager.rag_cfg.get("knowledge_sources", {})
    source_cfg = sources_cfg.get(source_name)
    if not source_cfg:
        logger.error(f"[{source_name}] 未在 rag.yaml 中配置")
        return

    src_dir = (ROOT / source_cfg["path"]).resolve()
    file_types = source_cfg.get("file_types", [])
    if not src_dir.exists():
        logger.warning(f"[{source_name}] 目录不存在: {src_dir}")
        return

    files = _collect_files(src_dir, file_types)
    if not files:
        logger.info(f"[{source_name}] 目录为空: {src_dir}")
        return

    engine = manager._engines.get(source_name)
    if engine is None:
        logger.error(f"[{source_name}] 引擎未就绪")
        return

    lightrag = await _get_lightrag(engine)

    record_file = (
        Path(manager.rag_cfg.get("working_dir", "knowledge_base/rag_storage"))
        / source_name
        / ".indexed_plain.txt"
    )
    indexed = set() if force else _load_indexed(record_file)
    logger.info(
        f"[{source_name}] 待处理 {len(files)} 个文件，已索引签名 {len(indexed)} 个 "
        f"(mode={'force' if force else 'incremental'})"
    )

    inserted, skipped, failed = 0, 0, 0
    for path in files:
        sig = _file_signature(path)
        if sig in indexed:
            skipped += 1
            logger.info(f"[{source_name}] skip (already indexed): {path.name}")
            continue

        chunks = _file_to_chunks(path)
        if not chunks:
            logger.warning(f"[{source_name}] no text extracted: {path.name}")
            failed += 1
            continue

        logger.info(
            f"[{source_name}] inserting {path.name} → {len(chunks)} chunks"
        )
        try:
            texts = [text for _, text in chunks]
            ids = [doc_id for doc_id, _ in chunks]
            # LightRAG 的 ainsert: 支持 (string | List[str], ids=List[str], file_paths=...)
            try:
                await lightrag.ainsert(
                    texts,
                    ids=ids,
                    file_paths=[path.name] * len(chunks),
                )
            except TypeError:
                # 老版本 lightrag 可能签名不同
                await lightrag.ainsert(texts)
            _append_indexed(record_file, sig)
            inserted += 1
        except Exception as insert_error:
            logger.error(
                f"[{source_name}] insert failed {path.name}: {insert_error}",
                exc_info=True,
            )
            failed += 1

    logger.info(
        f"[{source_name}] done: inserted={inserted}, skipped={skipped}, failed={failed}"
    )


async def _main_async(args: argparse.Namespace) -> int:
    manager = RAGManager(args.config)
    if not manager.is_enabled():
        logger.error("rag.enabled=false，无法索引")
        return 1

    available = list(manager.rag_cfg.get("knowledge_sources", {}).keys())
    targets = [args.source] if args.source else available
    if args.source and args.source not in available:
        logger.error(f"source '{args.source}' not in rag.yaml. available={available}")
        return 1

    logger.info(f"smoke index sources: {targets}")
    for src in targets:
        await _index_one_source(manager, src, force=args.force)
    logger.info("ALL DONE ✅")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="冒烟索引：绕过 MinerU 直接灌纯文本到 LightRAG")
    parser.add_argument("--config", default="config/rag.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
