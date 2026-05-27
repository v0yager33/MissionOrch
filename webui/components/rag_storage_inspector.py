"""RAG 索引产物统计器。

把 ``knowledge_base/rag_storage/<source>/`` 下的关键文件读出来，给前端展示：
- vdb_chunks.json         向量化片段数（核心指标）
- vdb_entities.json       实体数（朴素 RAG 模式应该是 0）
- vdb_relationships.json  关系数（朴素 RAG 模式应该是 0）
- graph_chunk_entity_relation.graphml  图节点 / 边数（朴素 RAG 模式应该都是 0）
- kv_store_full_docs.json 已索引文档数
- kv_store_text_chunks.json chunk 总数
- kv_store_llm_response_cache.json LLM 响应缓存条数（多查询改写产物）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _safe_json_count(file_path: Path) -> int:
    """解析 JSON 文件，返回顶层条目数（dict→len(dict) / list→len(list)）。"""
    if not file_path.exists():
        return 0
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as parse_error:
        logger.warning("inspector: 无法解析 %s: %s", file_path, parse_error)
        return 0
    # nano-vectordb 的 vdb_*.json 是 {data: [...], ...}
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return len(data["data"])
        return len(data)
    if isinstance(data, list):
        return len(data)
    return 0


def _safe_file_size_kb(file_path: Path) -> float:
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / 1024.0


def _graphml_counts(file_path: Path) -> Dict[str, int]:
    """轻量统计 graphml 中 <node> / <edge> 出现次数（不做完整 XML 解析）。"""
    if not file_path.exists():
        return {"nodes": 0, "edges": 0}
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"nodes": 0, "edges": 0}
    return {
        "nodes": text.count("<node "),
        "edges": text.count("<edge "),
    }


def inspect_source(storage_root: str | Path, source_name: str) -> Dict[str, Any]:
    """统计单一 source 的索引产物。"""
    base = Path(storage_root) / source_name

    chunks_file = base / "vdb_chunks.json"
    entities_file = base / "vdb_entities.json"
    relations_file = base / "vdb_relationships.json"
    graph_file = base / "graph_chunk_entity_relation.graphml"
    full_docs = base / "kv_store_full_docs.json"
    text_chunks = base / "kv_store_text_chunks.json"
    llm_cache = base / "kv_store_llm_response_cache.json"
    indexed_record = base / ".indexed_files.txt"

    graph_stats = _graphml_counts(graph_file)

    indexed_count = 0
    if indexed_record.exists():
        indexed_count = sum(
            1 for line in indexed_record.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    return {
        "source": source_name,
        "exists": base.exists(),
        "path": str(base),
        "vdb_chunks": _safe_json_count(chunks_file),
        "vdb_chunks_kb": round(_safe_file_size_kb(chunks_file), 1),
        "vdb_entities": _safe_json_count(entities_file),
        "vdb_relationships": _safe_json_count(relations_file),
        "graph_nodes": graph_stats["nodes"],
        "graph_edges": graph_stats["edges"],
        "full_docs": _safe_json_count(full_docs),
        "text_chunks": _safe_json_count(text_chunks),
        "llm_cache_entries": _safe_json_count(llm_cache),
        "indexed_files": indexed_count,
    }


def inspect_all(storage_root: str | Path, source_names: List[str]) -> List[Dict[str, Any]]:
    """统计多个 source。"""
    return [inspect_source(storage_root, name) for name in source_names]


def list_indexed_files(storage_root: str | Path, source_name: str) -> List[Dict[str, Any]]:
    """从 kv_store_full_docs.json 反查已入库的文件名 + 字符长度。"""
    full_docs_path = Path(storage_root) / source_name / "kv_store_full_docs.json"
    if not full_docs_path.exists():
        return []
    try:
        data = json.loads(full_docs_path.read_text(encoding="utf-8"))
    except Exception as parse_error:
        logger.warning("list_indexed_files: 解析失败 %s: %s", full_docs_path, parse_error)
        return []
    if not isinstance(data, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for doc_id, payload in data.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "file_path": payload.get("file_path", ""),
                "content_length": len(payload.get("content", "") or ""),
            }
        )
    return sorted(rows, key=lambda row: row["file_path"])
