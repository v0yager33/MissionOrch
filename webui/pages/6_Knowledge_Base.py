"""知识库管理页 —— 列出 source、上传 / 删除文件、跑索引、查看索引产物。"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import streamlit as st

from webui.components.config_io import load_yaml
from webui.components.rag_storage_inspector import (
    inspect_all,
    list_indexed_files,
)
from webui.components.ui_helpers import render_header
from webui.state import init_state

init_state()
render_header(
    "📚 知识库管理",
    "查看 / 维护各 source：上传文件、跑索引、查看 RAG 索引产物。",
)

CONFIG_PATH = "config/rag.yaml"
raw_cfg = load_yaml(CONFIG_PATH)
rag_cfg = raw_cfg.get("rag") or {}
sources_cfg = rag_cfg.get("knowledge_sources") or {}
storage_root = rag_cfg.get("working_dir", "./knowledge_base/rag_storage")


# ── 内部：跑索引（复用 scripts/index_knowledge_base.py 的核心逻辑） ──
def _run_indexing(source_name: str, force: bool) -> tuple[bool, str]:
    """同步触发一次索引，返回 (是否成功, 文本日志)。

    复用 ``scripts/index_knowledge_base._index_one_source``，让 UI 与 CLI 行为一致。
    """
    import io
    import logging as _logging

    # 把 logger 输出捕获到内存
    log_buffer = io.StringIO()
    handler = _logging.StreamHandler(log_buffer)
    handler.setFormatter(
        _logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root_logger = _logging.getLogger()
    root_logger.addHandler(handler)

    try:
        from missionorch_lc.core.rag_manager import RAGManager
        from scripts.index_knowledge_base import _index_one_source

        manager = RAGManager(CONFIG_PATH)
        if not manager.is_enabled():
            return False, "RAG 未启用，无法索引（rag.enabled=false）"

        # 在新 loop 里跑（Streamlit 主线程没有 running loop）
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(
                _index_one_source(manager, source_name, force=force)
            )
        finally:
            new_loop.close()
        return True, log_buffer.getvalue()
    except Exception as run_error:
        return False, log_buffer.getvalue() + f"\n[ERROR] {type(run_error).__name__}: {run_error}"
    finally:
        root_logger.removeHandler(handler)


# ── 内部：删除源文件 + 同步清理 .indexed_files.txt 中的签名 ──
def _file_signature(path: Path, base_dir: Path) -> str:
    """与 ``scripts/index_knowledge_base._file_signature`` 完全一致的签名算法。

    必须保持一致，否则清理 .indexed_files.txt 时签名对不上，下次跑增量索引
    会把已经被删的文件当作"未索引过"再跑一次（不会出错，但浪费）。
    """
    stat = path.stat()
    try:
        rel = path.relative_to(base_dir).as_posix()
    except ValueError:
        rel = path.name
    raw = f"{rel}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _delete_source_files(
    files_to_remove: List[Path],
    indexed_record_path: Path,
    source_base_dir: Path,
) -> Tuple[int, int, List[str]]:
    """批量删除文件 + 同步清理签名记录。

    Returns:
        (成功数, 失败数, 错误信息列表)
    """
    # 第 1 步：先把要删文件的签名都算出来（删完就没法 stat 了）
    sigs_to_remove = set()
    for path in files_to_remove:
        if path.exists():
            try:
                sigs_to_remove.add(_file_signature(path, base_dir=source_base_dir))
            except OSError:
                pass

    # 第 2 步：真正删文件
    success = 0
    failed = 0
    errors: List[str] = []
    for path in files_to_remove:
        try:
            if path.is_file():
                path.unlink()
                success += 1
            elif path.exists():
                errors.append(f"{path.name}: 不是普通文件，跳过")
                failed += 1
        except Exception as remove_error:
            errors.append(f"{path.name}: {remove_error}")
            failed += 1

    # 第 3 步：把对应签名从 .indexed_files.txt 移除
    if sigs_to_remove and indexed_record_path.exists():
        try:
            kept = [
                line.strip()
                for line in indexed_record_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and line.strip() not in sigs_to_remove
            ]
            indexed_record_path.write_text(
                "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
            )
        except Exception as record_error:
            errors.append(f"清理 .indexed_files.txt 失败: {record_error}")

    return success, failed, errors


if not sources_cfg:
    st.warning("`config/rag.yaml` 中没有任何 knowledge_sources。请到 **⚙️ RAG 配置** 页添加。")
    st.stop()


# ── Section 1: 概览 ──
st.markdown("### 📊 索引产物概览")
stats = inspect_all(storage_root, list(sources_cfg.keys()))
overview_rows = []
for stat in stats:
    overview_rows.append(
        {
            "Source": stat["source"],
            "已索引文件数": stat["indexed_files"],
            "Chunks": stat["vdb_chunks"],
            "Chunks 大小(KB)": stat["vdb_chunks_kb"],
            "Entities": stat["vdb_entities"],
            "Relations": stat["vdb_relationships"],
            "图节点": stat["graph_nodes"],
            "图边": stat["graph_edges"],
            "LLM 缓存条数": stat["llm_cache_entries"],
        }
    )
st.dataframe(overview_rows, hide_index=True, use_container_width=True)
st.caption(
    "🟢 **朴素 RAG 验收口径**：Entities / Relations / 图节点 / 图边 应全为 0；"
    "Chunks > 0 即说明向量库构建正常。"
)

st.divider()

# ── Section 2: 选一个 source 详细操作 ──
st.markdown("### 📁 选择 source 进行管理")
selected_source = st.selectbox("source", list(sources_cfg.keys()))
source_def = sources_cfg.get(selected_source) or {}
source_dir = Path(source_def.get("path", "")).resolve()
allowed_types: List[str] = source_def.get("file_types") or []

cols = st.columns(3)
cols[0].metric("源文件目录", str(source_dir))
cols[1].metric("支持格式", ", ".join(allowed_types) or "—")
cols[2].metric("描述", source_def.get("description", "") or "—")

# 列出源目录里的文件 + 多选删除
st.markdown("#### 📂 源目录现有文件")
if not source_dir.exists():
    st.info(f"目录不存在：{source_dir}")
    raw_files: List[Path] = []
else:
    raw_files = []
    for ext in allowed_types:
        raw_files.extend(source_dir.rglob(f"*{ext}"))
    raw_files = sorted(raw_files)

if not raw_files:
    st.info("（目录为空或没有支持的文件类型）")
else:
    st.caption(
        f"共 **{len(raw_files)}** 个文件。勾选下方文件后点底部"
        "『🗑 删除选中文件』批量删除（同时清理 `.indexed_files.txt` 中的签名）。"
    )

    # 表头
    head_cols = st.columns([0.5, 4, 2.5, 1.2])
    head_cols[0].markdown("**☑️**")
    head_cols[1].markdown("**相对路径**")
    head_cols[2].markdown("**修改时间**")
    head_cols[3].markdown("**大小(KB)**")

    # 每一行：checkbox + 文件信息
    selected_files: List[Path] = []
    for f in raw_files:
        row_cols = st.columns([0.5, 4, 2.5, 1.2])
        # 用 source 名 + 相对路径 hash 做 checkbox key，避免切换 source 时 state 串
        rel_path = str(f.relative_to(source_dir))
        ck_key = f"del_ck_{selected_source}_{hashlib.md5(rel_path.encode()).hexdigest()}"
        checked = row_cols[0].checkbox(
            "select",
            key=ck_key,
            label_visibility="collapsed",
        )
        row_cols[1].markdown(f"`{rel_path}`")
        row_cols[2].markdown(
            time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
        )
        row_cols[3].markdown(f"{round(f.stat().st_size / 1024, 1)}")
        if checked:
            selected_files.append(f)

    # 批量操作按钮
    op_col1, op_col2, op_col3 = st.columns([1.2, 1.2, 4])
    if op_col1.button(
        f"🗑 删除选中文件（{len(selected_files)}）",
        type="secondary",
        disabled=(len(selected_files) == 0),
        key="delete_selected_files",
    ):
        record_path = Path(storage_root) / selected_source / ".indexed_files.txt"
        success, failed, errors = _delete_source_files(
            selected_files, record_path, source_base_dir=source_dir
        )
        if success > 0:
            st.success(
                f"✅ 已删除 {success} 个文件并同步清理签名"
                f"（如需让向量库也彻底丢掉这些文件，请到下方点 force 重建索引）。"
            )
        if failed > 0:
            st.error(f"❌ {failed} 个文件删除失败")
        for err in errors:
            st.warning(err)
        if success > 0 or failed > 0:
            st.rerun()

    if op_col2.button("☑️ 全选当前列表", key="select_all_files"):
        # 把当前所有 checkbox 的 state 强制改为 True，再 rerun
        for f in raw_files:
            rel_path = str(f.relative_to(source_dir))
            ck_key = f"del_ck_{selected_source}_{hashlib.md5(rel_path.encode()).hexdigest()}"
            st.session_state[ck_key] = True
        st.rerun()

    op_col3.caption(
        "💡 删除源文件**不会**自动从向量库 (`vdb_chunks.json`) 移除已嵌入的 chunk。"
        "完全清理需要 force 重建索引（或上方危险区清空整个 source）。"
    )

# ── 上传 ──
st.markdown("#### ⬆️ 上传文件到该 source")
uploaded = st.file_uploader(
    "支持后缀：" + (", ".join(allowed_types) or "—"),
    accept_multiple_files=True,
    type=[ext.lstrip(".") for ext in allowed_types if ext.startswith(".")] or None,
)
if uploaded:
    if st.button(f"💾 保存 {len(uploaded)} 个文件到 {source_dir}"):
        try:
            source_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for upload in uploaded:
                target_path = source_dir / upload.name
                with target_path.open("wb") as f:
                    f.write(upload.getbuffer())
                saved += 1
            st.success(f"✅ 已保存 {saved} 个文件到 {source_dir}")
            st.rerun()
        except Exception as save_error:
            st.error(f"❌ 保存失败：{save_error}")

# ── 索引 ──
st.markdown("#### 🔄 跑索引（写入 RAG 向量库 vdb_chunks）")
idx_col1, idx_col2 = st.columns([1, 1])
with idx_col1:
    force_rebuild = st.checkbox(
        "force（忽略已索引签名重建）", value=False, key="force_rebuild"
    )
with idx_col2:
    if st.button(f"🚀 索引 source `{selected_source}`", type="primary"):
        with st.spinner(f"正在索引 {selected_source} … 第一次会加载本地 Qwen3 嵌入模型"):
            ok, log_text = _run_indexing(selected_source, force_rebuild)
        if ok:
            st.success(f"✅ source `{selected_source}` 索引完成")
        else:
            st.error("❌ 索引失败，请看下方日志")
        st.code(log_text, language="text")
        st.rerun()

# ── 已入库文件清单（从 RAG kv_store_full_docs.json 反查） ──
st.markdown("#### 📋 已入库文档（从 RAG kv 读取）")
indexed_rows = list_indexed_files(storage_root, selected_source)
if not indexed_rows:
    st.info("（没有已入库文档，或 kv_store_full_docs.json 不存在）")
else:
    st.dataframe(indexed_rows, hide_index=True, use_container_width=True)

# ── 危险区：清空索引产物 ──
with st.expander("☠️ 危险区：清空 source 的索引产物", expanded=False):
    st.warning(
        f"将删除 `{Path(storage_root) / selected_source}` 整个目录（不影响源文件）。"
    )
    confirm_text = st.text_input(
        f"为确认请输入 source 名 `{selected_source}`：",
        key="confirm_clear",
    )
    if st.button("🗑️ 清空索引", key="clear_index"):
        if confirm_text != selected_source:
            st.error("确认串不匹配，操作取消。")
        else:
            target = Path(storage_root) / selected_source
            try:
                if target.exists():
                    shutil.rmtree(target)
                st.success(f"✅ 已删除 {target}")
                st.rerun()
            except Exception as remove_error:
                st.error(f"❌ 删除失败：{remove_error}")
