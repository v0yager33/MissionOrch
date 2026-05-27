"""Streamlit 公共渲染辅助。

把页面里反复出现的 UI 片段抽出来，避免每个页面里重写表头 / 标签徽章。
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_header(title: str, subtitle: str = "") -> None:
    """统一的页面顶部标题。"""
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def render_status_badge(status: str) -> str:
    """根据 status 字符串返回带 emoji 的徽章文字。"""
    return {
        "idle": "⚪ 空闲",
        "running": "🟡 运行中",
        "done": "🟢 已完成",
        "error": "🔴 出错",
    }.get(status, status)


def render_kv_grid(items: Dict[str, Any], columns: int = 4) -> None:
    """把一个 dict 渲染成多列的 metric 块。"""
    if not items:
        return
    cols = st.columns(columns)
    keys = list(items.keys())
    for index, key in enumerate(keys):
        with cols[index % columns]:
            value = items[key]
            display = "—" if value is None else str(value)
            st.metric(label=key, value=display)


def render_history_table(history: List[Dict[str, Any]]) -> None:
    """Judge 的迭代历史表。"""
    if not history:
        st.info("暂无迭代历史。")
        return
    rows = []
    for entry in history:
        rows.append(
            {
                "迭代": entry.get("iteration"),
                "得分": round(float(entry.get("score", 0)), 2),
                "判定": entry.get("verdict", ""),
                "反馈摘要": (entry.get("feedback") or "")[:120],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_token_usage(usage: Dict[str, Any]) -> None:
    """Token 用量摘要 + 按模型拆分。"""
    if not usage:
        st.info("没有 Token 用量数据。")
        return
    total = usage.get("total_tokens", 0)
    st.metric("Total Tokens", f"{total:,}")
    by_model = usage.get("by_model", {})
    if not by_model:
        return
    rows = []
    for model_name, stats in by_model.items():
        rows.append(
            {
                "模型": model_name,
                "Prompt": f"{stats.get('prompt_tokens', 0):,}",
                "Completion": f"{stats.get('completion_tokens', 0):,}",
                "Total": f"{stats.get('total_tokens', 0):,}",
                "调用次数": stats.get("call_count", 0),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_timing(timing: List[Dict[str, Any]]) -> None:
    if not timing:
        st.info("没有阶段耗时数据。")
        return
    rows = []
    for stage in timing:
        elapsed = stage.get("elapsed")
        if elapsed is None:
            continue
        rows.append(
            {
                "阶段": stage.get("run_name") or ", ".join(stage.get("tags", []) or []) or "—",
                "耗时(秒)": round(float(elapsed), 2),
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def code_block(text: str, language: str = "text") -> None:
    """统一的代码块渲染。"""
    if not text:
        st.info("（空）")
        return
    st.code(text, language=language)


def warn_if_no_run() -> bool:
    """没有最近一次运行时给出引导，返回是否应该早退。"""
    from ..state import get_run_result

    result = get_run_result()
    if result is None:
        st.warning("还没有运行过任务。请先到 **🚀 任务执行** 页跑一次。")
        return True
    return False
