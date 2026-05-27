"""Streamlit session_state 的强类型包装。

把所有 UI 跨页面共享的状态集中在这里，避免页面里到处写
``st.session_state.setdefault(...)`` 的散乱代码。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


# ── 状态键常量（避免拼写错误）──
KEY_LAST_RUN_RESULT = "last_run_result"          # 最近一次 run_graph 的完整 dict
KEY_LAST_RUN_MISSION = "last_run_mission"        # 最近一次任务描述
KEY_LAST_RUN_PARAMS = "last_run_params"          # 最近一次运行参数
KEY_RUN_LOGS = "run_logs"                        # 阶段日志（List[str]）
KEY_RUN_STATUS = "run_status"                    # idle / running / done / error
KEY_RUN_ERROR = "run_error"                      # 错误信息
KEY_RAG_QUERY_HISTORY = "rag_query_history"      # RAG 直查历史 [{query, source, result, ts}]
KEY_RAG_MANAGER = "rag_manager_singleton"        # 进程内 RAGManager 单例


def init_state() -> None:
    """在 app 入口调用一次：初始化所有需要的 session_state。"""
    defaults: Dict[str, Any] = {
        KEY_LAST_RUN_RESULT: None,
        KEY_LAST_RUN_MISSION: "",
        KEY_LAST_RUN_PARAMS: {},
        KEY_RUN_LOGS: [],
        KEY_RUN_STATUS: "idle",
        KEY_RUN_ERROR: "",
        KEY_RAG_QUERY_HISTORY: [],
        KEY_RAG_MANAGER: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_run_result(
    *,
    result: Dict[str, Any],
    mission: str,
    params: Dict[str, Any],
) -> None:
    """记录最近一次完整运行结果。"""
    st.session_state[KEY_LAST_RUN_RESULT] = result
    st.session_state[KEY_LAST_RUN_MISSION] = mission
    st.session_state[KEY_LAST_RUN_PARAMS] = params
    st.session_state[KEY_RUN_STATUS] = "done"


def get_run_result() -> Optional[Dict[str, Any]]:
    return st.session_state.get(KEY_LAST_RUN_RESULT)


def append_run_log(message: str) -> None:
    logs: List[str] = st.session_state.setdefault(KEY_RUN_LOGS, [])
    logs.append(message)


def reset_run_logs() -> None:
    st.session_state[KEY_RUN_LOGS] = []


def get_run_logs() -> List[str]:
    return list(st.session_state.get(KEY_RUN_LOGS) or [])


def append_rag_history(entry: Dict[str, Any]) -> None:
    history: List[Dict[str, Any]] = st.session_state.setdefault(KEY_RAG_QUERY_HISTORY, [])
    history.append(entry)
    # 最多保留 50 条
    if len(history) > 50:
        del history[0 : len(history) - 50]


def get_rag_history() -> List[Dict[str, Any]]:
    return list(st.session_state.get(KEY_RAG_QUERY_HISTORY) or [])
