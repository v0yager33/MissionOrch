"""RAG 直查页 —— 直接调 retrieve_with_rewrite，看 Multi-Query 改写 + 命中片段。"""

from __future__ import annotations

import time

import streamlit as st

from webui.components.rag_query import get_or_create_rag_manager, query_with_rewrite
from webui.components.ui_helpers import render_header
from webui.state import (
    append_rag_history,
    get_rag_history,
    init_state,
)

init_state()
render_header(
    "🔍 RAG 直查",
    "绕过 Researcher，直接调用 RAGManager 的 Multi-Query 检索，看每条子 query 命中了什么。",
)


# ── 取（或缓存）RAGManager 单例 ──
@st.cache_resource(show_spinner="🔄 正在加载 RAGManager（首次会加载本地 Qwen3 权重）…")
def _load_rag_manager_cached(config_path: str = "config/rag.yaml"):
    return get_or_create_rag_manager(config_path)


try:
    rag_manager = _load_rag_manager_cached()
except Exception as load_error:
    st.error(f"❌ RAGManager 加载失败：{load_error}")
    st.stop()

if not rag_manager.is_enabled():
    st.warning(
        "RAG 当前未启用。请到 **⚙️ RAG 配置** 页把 `rag.enabled` 设为 true 并配好 embedding 模型路径。"
    )
    st.stop()

sources = rag_manager.list_sources()
if not sources:
    st.warning("RAGManager 没有可用 source（rag.knowledge_sources 为空）。")
    st.stop()

# ── 输入 ──
with st.form("rag_query_form", border=True):
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(
            "查询语句",
            value="SEAD 的三个阶段是什么？",
            help="可以是中文或英文。Multi-Query 会自动改写成 3 条子 query 并发检索。",
        )
    with col2:
        source = st.selectbox("知识源", sources, index=0)
    with col3:
        mode = st.selectbox("检索模式", ["naive", "hybrid", "local", "global"], index=0)

    submitted = st.form_submit_button("🔍 开始检索", type="primary", use_container_width=True)


# ── 执行 ──
if submitted:
    if not query.strip():
        st.error("查询不能为空。")
    else:
        with st.spinner("正在检索…"):
            started = time.time()
            result = query_with_rewrite(
                rag_manager=rag_manager,
                query=query.strip(),
                source=source,
                mode=mode,
            )

        if result.error:
            st.error(f"❌ 检索失败：{result.error}")
        else:
            append_rag_history(
                {
                    "query": result.query,
                    "source": result.source,
                    "mode": result.mode,
                    "elapsed": result.elapsed_seconds,
                    "rewrite_enabled": result.rewrite_enabled,
                    "sub_queries": result.sub_queries,
                    "raw_text": result.raw_text,
                    "ts": time.time(),
                }
            )

            # 顶部摘要
            st.markdown("### 检索结果")
            sumcols = st.columns(4)
            sumcols[0].metric("Source", result.source)
            sumcols[1].metric("Mode", result.mode)
            sumcols[2].metric(
                "Multi-Query",
                "✅ 启用" if result.rewrite_enabled else "❌ 关闭",
            )
            sumcols[3].metric("耗时", f"{result.elapsed_seconds:.2f}s")

            # 改写后的子 queries
            if result.sub_queries:
                st.markdown("**🔁 改写后的子 query：**")
                for index, sub in enumerate(result.sub_queries, start=1):
                    st.markdown(f"{index}. `{sub}`")
            else:
                st.info("（没有 Multi-Query 改写，使用原 query 直接检索）")

            # 分组展示每条子 query 的命中片段
            st.markdown("**📚 命中片段：**")
            sections = result.sub_sections or [{"sub_query": "", "body": result.raw_text}]
            for index, section in enumerate(sections, start=1):
                title = section.get("sub_query") or f"片段 {index}"
                with st.expander(f"🔹 {title}", expanded=(index == 1)):
                    body = section.get("body") or ""
                    if not body:
                        st.info("（无内容）")
                    else:
                        st.markdown(body)


# ── 历史 ──
st.divider()
st.markdown("### 📜 最近的查询历史")
history = get_rag_history()
if not history:
    st.info("还没有历史。")
else:
    rows = []
    for entry in history[-20:][::-1]:
        rows.append(
            {
                "时间": time.strftime("%H:%M:%S", time.localtime(entry.get("ts", 0))),
                "Source": entry.get("source"),
                "Mode": entry.get("mode"),
                "Query": (entry.get("query") or "")[:80],
                "改写": "✅" if entry.get("rewrite_enabled") else "—",
                "耗时": f"{entry.get('elapsed', 0):.2f}s",
                "结果长度": len(entry.get("raw_text") or ""),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
