"""历史运行页 —— 浏览 output/ 目录里的 JSON 结果，复现一次完整展示。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from webui.components.ui_helpers import (
    code_block,
    render_header,
    render_history_table,
    render_kv_grid,
    render_timing,
    render_token_usage,
)
from webui.state import init_state

init_state()
render_header(
    "📜 历史运行",
    "浏览 `output/` 目录里以前导出的运行结果（main.py --output-json 或本 UI 下载的 JSON）。",
)


OUTPUT_DIR = Path("output")


def _scan_json_files() -> List[Dict[str, Any]]:
    """扫描 output/ 下所有 .json，外层附带 mtime / size 信息。"""
    if not OUTPUT_DIR.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(OUTPUT_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        rows.append(
            {
                "path": path,
                "name": path.name,
                "rel": str(path.relative_to(OUTPUT_DIR)),
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    return rows


json_files = _scan_json_files()

if not json_files:
    st.info(f"`{OUTPUT_DIR}/` 下还没有 JSON 文件。")
    st.stop()

# ── 列表 ──
st.markdown(f"### 共找到 {len(json_files)} 份历史结果")
table_rows = [
    {"文件": item["rel"], "大小(KB)": item["size_kb"], "修改时间": item["mtime"]}
    for item in json_files
]
st.dataframe(table_rows, hide_index=True, use_container_width=True)

# 选择一份
selected_rel = st.selectbox(
    "选择要查看的结果",
    [item["rel"] for item in json_files],
)
selected_meta = next(item for item in json_files if item["rel"] == selected_rel)
selected_path: Path = selected_meta["path"]

# ── 解析 ──
try:
    payload: Dict[str, Any] = json.loads(selected_path.read_text(encoding="utf-8"))
except Exception as parse_error:
    st.error(f"❌ 解析失败：{parse_error}")
    st.stop()

st.divider()
st.markdown(f"#### 📂 `{selected_rel}` 结果详情")

render_kv_grid(
    {
        "迭代次数": payload.get("iterations", 0),
        "最终得分": round(float(payload.get("final_score", 0)), 2),
        "最佳得分": round(float(payload.get("best_score", 0)), 2),
        "解析": "✅" if payload.get("parse_success") else "❌",
        "RAG": "启用" if payload.get("rag_enabled") else "未启用",
        "Sources": ", ".join(payload.get("rag_sources") or []) or "—",
    },
    columns=6,
)

tabs = st.tabs(
    [
        "1️⃣ 任务分析",
        "2️⃣ 研究简报",
        "3️⃣ COA 矩阵",
        "4️⃣ 4 种结构化格式",
        "5️⃣ 验证 / 历史",
        "6️⃣ Token / 计时",
        "7️⃣ 原始 JSON",
    ]
)

with tabs[0]:
    analysis = payload.get("mission_analysis") or {}
    if not analysis:
        st.info("没有 mission_analysis。")
    else:
        st.markdown(f"**意图**: {analysis.get('mission_intent', '—')}")
        objectives = analysis.get("objectives") or []
        if objectives:
            st.dataframe(
                [
                    {
                        "ID": obj.get("id"),
                        "优先级": obj.get("priority"),
                        "描述": obj.get("description", ""),
                    }
                    for obj in objectives
                ],
                hide_index=True,
                use_container_width=True,
            )
        for constraint in analysis.get("constraints") or []:
            st.markdown(f"- {constraint}")
        for query in analysis.get("research_queries") or []:
            st.markdown(f"- {query}")

with tabs[1]:
    brief = payload.get("research_brief") or ""
    if brief:
        st.markdown(brief)
    else:
        st.info("没有 research_brief。")

with tabs[2]:
    coa = payload.get("coa_table") or ""
    if coa:
        st.markdown(coa)
    else:
        st.info("没有 COA 文本。")

with tabs[3]:
    outputs = payload.get("outputs") or {}
    final_coa = payload.get("final_coa") or {}
    sub_tabs = st.tabs(["JSON", "YAML", "扁平矩阵", "压缩格式", "原始 final_coa"])
    with sub_tabs[0]:
        code_block(outputs.get("json_format", ""), "json")
    with sub_tabs[1]:
        code_block(outputs.get("yaml_format", ""), "yaml")
    with sub_tabs[2]:
        flat = outputs.get("flat_matrix")
        if isinstance(flat, list) and flat:
            st.dataframe(flat, hide_index=True, use_container_width=True)
        else:
            code_block(json.dumps(flat, ensure_ascii=False, indent=2), "json")
    with sub_tabs[3]:
        code_block(outputs.get("condensed_format", ""), "text")
    with sub_tabs[4]:
        code_block(json.dumps(final_coa, ensure_ascii=False, indent=2), "json")

with tabs[4]:
    validation = payload.get("validation") or {}
    if validation:
        st.markdown(f"- **通过**: {'✅' if validation.get('is_valid') else '❌'}")
        feedback = validation.get("validation_feedback") or ""
        if feedback:
            st.code(feedback, language="text")
        for issue in validation.get("issues_found") or []:
            st.markdown(f"- {issue}")
    render_history_table(payload.get("history") or [])

with tabs[5]:
    render_token_usage(payload.get("token_usage") or {})
    render_timing(payload.get("timing") or [])

with tabs[6]:
    code_block(json.dumps(payload, ensure_ascii=False, indent=2), "json")
