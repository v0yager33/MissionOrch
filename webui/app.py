"""MissionOrch-LC Streamlit Web UI 主入口。

启动方式：
    streamlit run webui/app.py
    # 或一键脚本：
    python run_ui.py

页面通过 ``pages/`` 目录自动注册（Streamlit 多页面机制）。本文件只负责：
1. 全局状态初始化
2. 项目根路径注入到 sys.path（让页面可以直接 ``from missionorch_lc...``）
3. 渲染欢迎主页 + 概览面板
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ── 把项目 src/ 加入 sys.path，让所有页面都能 import missionorch_lc.* ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# 让 webui 内部的 from .components 也能用
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 切到项目根目录，让所有相对路径（config/, knowledge_base/, output/）一致
os.chdir(PROJECT_ROOT)

from webui.state import init_state                                # noqa: E402
from webui.components.env_io import KNOWN_API_KEYS, load_env, merge_env_into_process  # noqa: E402
from webui.components.config_io import load_yaml                  # noqa: E402
from webui.components.rag_storage_inspector import inspect_all    # noqa: E402

# ── 页面级配置（必须在任何 st.* 之前） ──
st.set_page_config(
    page_title="MissionOrch-LC Web UI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# ── 把 .env 注入进程，让 ModelRouter 能读到 API key ──
env_pairs = load_env(PROJECT_ROOT / ".env")
if env_pairs:
    merge_env_into_process(env_pairs)


# ── 主页内容 ──
def render_home() -> None:
    st.title("🎯 MissionOrch-LC — 6-Agent COA 编排 Web UI")
    st.caption("基于 LangChain + LangGraph + 朴素 RAG 的多智能体作战方案生成系统")

    st.markdown(
        """
左侧侧边栏选择对应页面：

- **Mission Run** — 输入任务描述，跑一次 6-Agent 流水线，看实时阶段流
- **Results** — 上一次运行的完整产物：分析 / 简报 / COA 矩阵 / 4 种格式 / Token / 计时
- **RAG Query** — 直接调 `retrieve_with_rewrite`，看 Multi-Query 改写 + 命中片段
- **RAG Config** — 在线编辑 `config/rag.yaml`（embedding / reranker / parser / 检索参数）
- **LLM Config** — 编辑 `config/models.yaml` + `.env` 的 API keys，支持 ping 测试
- **Knowledge Base** — 上传文件、跑索引、查看 `rag_storage/` 索引产物统计
- **History** — 浏览 `output/` 里以前跑过的 JSON 结果
        """
    )

    st.divider()
    st.subheader("⚡ 系统快速概览")

    # ── 概览 1: 配置文件状态 ──
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        rag_cfg = load_yaml("config/rag.yaml")
        rag_enabled = (rag_cfg.get("rag") or {}).get("enabled", False)
        st.metric(
            "RAG 启用",
            "✅ 是" if rag_enabled else "❌ 否",
            help="config/rag.yaml → rag.enabled",
        )
    with cfg_col2:
        models_cfg = load_yaml("config/models.yaml")
        n_models = len((models_cfg.get("models") or {}))
        st.metric("已配置 LLM", f"{n_models} 个")
    with cfg_col3:
        env_keys_set = sum(1 for key in KNOWN_API_KEYS if os.getenv(key))
        st.metric("API Keys 注入", f"{env_keys_set}/{len(KNOWN_API_KEYS)} 个")

    # ── 概览 2: 知识库 sources ──
    st.markdown("#### 📚 知识库 source 索引产物")
    sources_cfg = (rag_cfg.get("rag") or {}).get("knowledge_sources") or {}
    storage_root = (rag_cfg.get("rag") or {}).get(
        "working_dir", "./knowledge_base/rag_storage"
    )
    if sources_cfg:
        stats = inspect_all(storage_root, list(sources_cfg.keys()))
        rows = []
        for stat in stats:
            rows.append(
                {
                    "Source": stat["source"],
                    "已索引文件": stat["indexed_files"],
                    "Chunks": stat["vdb_chunks"],
                    "Entities": stat["vdb_entities"],
                    "Relations": stat["vdb_relationships"],
                    "图节点": stat["graph_nodes"],
                    "图边": stat["graph_edges"],
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(
            "🟢 **朴素 RAG 模式**预期：Entities / Relations / 图节点 / 图边 全部为 0；"
            "Chunks 大于 0 即说明向量库正常。"
        )
    else:
        st.info("`config/rag.yaml` 中没有配置任何 knowledge_sources。")

    # ── 概览 3: 最近一次运行 ──
    st.markdown("#### 📊 最近一次运行")
    last = st.session_state.get("last_run_result")
    if not last:
        st.info("尚无运行记录。请到 **🚀 任务执行** 页发起一次。")
        return
    cols = st.columns(4)
    cols[0].metric("迭代次数", last.get("iterations", 0))
    cols[1].metric("最终得分", round(float(last.get("final_score", 0)), 2))
    cols[2].metric("最佳得分", round(float(last.get("best_score", 0)), 2))
    cols[3].metric(
        "解析", "✅ 成功" if last.get("parse_success") else "❌ 失败"
    )


render_home()
