"""结果详情页 —— 展示最近一次运行的完整产物。"""

from __future__ import annotations

import json

import streamlit as st

from webui.components.ui_helpers import (
    code_block,
    render_header,
    render_history_table,
    render_kv_grid,
    render_timing,
    render_token_usage,
    warn_if_no_run,
)
from webui.state import get_run_result, init_state

init_state()
render_header(
    "📊 结果详情",
    "最近一次 6-Agent 流水线产物：mission_analysis / research_brief / COA / Token / 验证 …",
)

if warn_if_no_run():
    st.stop()

result = get_run_result() or {}

# ── 顶部摘要 ──
mission = st.session_state.get("last_run_mission", "")
params = st.session_state.get("last_run_params", {})

with st.expander("📌 本次任务输入", expanded=False):
    st.markdown(f"**Mission**:\n\n{mission}")
    st.json(params)

render_kv_grid(
    {
        "迭代次数": result.get("iterations", 0),
        "最终得分": round(float(result.get("final_score", 0)), 2),
        "最佳得分": round(float(result.get("best_score", 0)), 2),
        "解析": "✅" if result.get("parse_success") else "❌",
        "RAG": "启用" if result.get("rag_enabled") else "未启用",
        "RAG sources": ", ".join(result.get("rag_sources") or []) or "—",
    },
    columns=6,
)

# ── 6 个分区 Tab ──
tabs = st.tabs(
    [
        "1️⃣ 任务分析",
        "2️⃣ 研究简报",
        "3️⃣ COA 矩阵",
        "4️⃣ 4 种结构化格式",
        "5️⃣ 验证 / 历史",
        "6️⃣ Token / 计时",
    ]
)


# ── Tab 1: Analyst ──
with tabs[0]:
    analysis = result.get("mission_analysis") or {}
    if not analysis:
        st.info("没有 mission_analysis（可能用了经典 4-Agent 模式）。")
    else:
        st.markdown(f"**意图**: {analysis.get('mission_intent', '—')}")
        objectives = analysis.get("objectives") or []
        if objectives:
            st.markdown("**目标**")
            rows = [
                {
                    "ID": obj.get("id"),
                    "优先级": obj.get("priority"),
                    "描述": obj.get("description", ""),
                }
                for obj in objectives
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        constraints = analysis.get("constraints") or []
        if constraints:
            st.markdown("**约束**")
            for constraint in constraints:
                st.markdown(f"- {constraint}")
        queries = analysis.get("research_queries") or []
        if queries:
            st.markdown("**建议研究问题**")
            for query in queries:
                st.markdown(f"- {query}")


# ── Tab 2: Researcher ──
with tabs[1]:
    brief = result.get("research_brief", "") or ""
    if not brief:
        st.info("没有 research_brief（可能 RAG 未启用）。")
    else:
        st.markdown(brief)


# ── Tab 3: COA 矩阵原文 ──
with tabs[2]:
    coa_text = result.get("coa_table", "") or ""
    if not coa_text:
        st.info("没有生成 COA 文本。")
    else:
        st.markdown("**Planner 最终输出（COA Markdown 矩阵）**")
        st.markdown(coa_text)


# ── Tab 4: 4 种结构化格式 ──
with tabs[3]:
    outputs = result.get("outputs") or {}
    final_coa = result.get("final_coa") or {}
    if not outputs and not final_coa:
        st.info("解析失败或没有结构化输出。")
    else:
        sub_tabs = st.tabs(["JSON", "YAML", "扁平矩阵", "压缩格式", "原始 final_coa"])
        with sub_tabs[0]:
            code_block(outputs.get("json_format", "") or "", "json")
        with sub_tabs[1]:
            code_block(outputs.get("yaml_format", "") or "", "yaml")
        with sub_tabs[2]:
            flat = outputs.get("flat_matrix")
            if isinstance(flat, list):
                if flat:
                    st.dataframe(flat, hide_index=True, use_container_width=True)
                else:
                    st.info("扁平矩阵为空。")
            elif isinstance(flat, str):
                code_block(flat, "json")
            else:
                code_block(json.dumps(flat, ensure_ascii=False, indent=2), "json")
        with sub_tabs[3]:
            code_block(outputs.get("condensed_format", "") or "", "text")
        with sub_tabs[4]:
            code_block(json.dumps(final_coa, ensure_ascii=False, indent=2), "json")


# ── Tab 5: 验证 + 历史 ──
with tabs[4]:
    validation = result.get("validation") or {}
    if validation:
        st.markdown("**Validator 校验结果**")
        is_valid = validation.get("is_valid")
        st.markdown(f"- **通过**: {'✅' if is_valid else '❌'}")
        feedback = validation.get("validation_feedback", "") or ""
        if feedback:
            st.markdown("**反馈**：")
            st.code(feedback, language="text")
        issues = validation.get("issues_found") or []
        if issues:
            st.markdown("**发现的问题**")
            for issue in issues:
                st.markdown(f"- {issue}")
    else:
        st.info("没有验证记录。")

    st.markdown("**Judge 迭代历史**")
    render_history_table(result.get("history") or [])


# ── Tab 6: Token & Timing ──
with tabs[5]:
    st.markdown("**Token 用量**")
    render_token_usage(result.get("token_usage") or {})
    st.markdown("**阶段耗时**")
    render_timing(result.get("timing") or [])
    st.markdown("**完整阶段日志（stage_log）**")
    code_block(json.dumps(result.get("stage_log") or [], ensure_ascii=False, indent=2), "json")

# 底部下载完整 JSON
st.divider()
st.download_button(
    label="💾 下载完整结果 JSON",
    data=json.dumps(result, ensure_ascii=False, indent=2, default=str),
    file_name="missionorch_result.json",
    mime="application/json",
    use_container_width=True,
)
