"""任务执行页 —— 输入 mission，跑 6-Agent 流水线，实时看阶段进度。"""

from __future__ import annotations

import time
from typing import Dict, List

import streamlit as st

from webui.components.runner import RunHandle, STAGE_LABELS, start_run
from webui.components.ui_helpers import render_header, render_status_badge
from webui.state import init_state, set_run_result

init_state()
render_header(
    "🚀 任务执行",
    "输入任务描述并启动 6-Agent COA 编排流水线，下方实时显示各阶段进度。",
)


# ── 输入表单 ──
# 推荐"自由文本但有结构"输入：用 `## 段名` 分段，缺哪段不写哪段；
# 也兼容纯一段自由中文（Analyst 会自动 fallback）。
DEFAULT_MISSION = """## 任务
我方一个航母打击群须于 72 小时内对 X 海域的敌方综合防空网络实施压制，确保后续打击群安全突防。

## 我方力量
- 1 个航母打击群 (CSG-1)，包含：
  - EA-18G x4（电子战）
  - F-18E x8（对地打击）
  - E-2D x1（预警指挥）
- 出发母港：Yokosuka

## 敌方力量
- X 海域综合防空网络 (IADS)，包含：
  - 远程预警雷达 x2
  - S-400 SAM 阵地 x3
  - J-16 拦截机 x12

## 战场地理
- 作战海域：X 海域，中心约 38.5°N 124.2°E
- 我方航母距 X 海域约 300 海里
- 突防方向：从东南方向进入

## 时间约束
- 72 小时内完成
- 突防窗口：拂晓前

## 其他约束
- 不得越过敌方 12nm 领海线
- 接敌前保持 EMCON 静默

## 期望产出
- 一份完整 COA 方案，至少 4 个阶段
- 重点覆盖：电磁压制 → 制空 → 打击 → 安全返航
"""


with st.expander("📋 推荐输入格式说明（点击展开）"):
    st.markdown(
        """
本系统接受 **"自由文本但有结构"** 的任务描述。**所有段都是可选的**，缺哪段不写哪段；
段名匹配大小写不敏感、允许同义词；如果一段 `## ` 都没用，会自动按纯自由文本解析。

| 段名 | 同义词 | 写什么 |
| :--- | :--- | :--- |
| `## 任务` | `## 任务描述` `## 指挥意图` `## 任务目标` | 一段话讲清最高指挥意图 |
| `## 我方力量` | `## 我方` `## 友军` `## 蓝方` | 蓝方关键力量 / 平台 / 母港，可用 `- bullet` 列举 |
| `## 敌方力量` | `## 敌方` `## 红方` | 红方威胁清单 |
| `## 战场地理` | `## 地理` `## 作战区域` `## 战场环境` | 作战海域 / 距离 / 经纬度 / 突防方向 |
| `## 时间约束` | `## 时间` `## 时限` | 任务窗口 / 拂晓 / 截止时间 |
| `## 其他约束` | `## 约束` `## 限制条件` | 政治 / 法律 / 通信 / 弹药等硬约束 |
| `## 期望产出` | `## 期望` `## 输出要求` | 想要几阶段 / 重点关注什么 |

> **提示**：写得越具体、坐标 / 平台型号 / 距离越完整，生成的 COA 越对 AFSIM 仿真友好。
"""
    )

with st.form("mission_form", border=True):
    mission = st.text_area(
        "任务描述（mission_input）",
        value=DEFAULT_MISSION,
        height=320,
        help="可以是中文或英文，可分段写也可纯自由文本。Analyst 会自动抽取意图 / 实体 / 地理锚点。",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_iter = st.number_input("最大迭代", 1, 10, 3)
    with col2:
        threshold = st.slider("质量阈值", 0.0, 10.0, 8.0, step=0.5)
    with col3:
        use_rag = st.checkbox("启用 RAG", value=True, help="关闭后 Researcher 节点降级")
    with col4:
        legacy = st.checkbox("经典 4-Agent", value=False, help="退回旧版 Orchestrator")

    submitted = st.form_submit_button("🚀 开始执行", type="primary", use_container_width=True)


# ── 启动新一次运行 ──
if submitted:
    if not mission.strip():
        st.error("任务描述不能为空。")
    else:
        params = {
            "mission": mission.strip(),
            "use_rag": use_rag,
            "legacy": legacy,
            "max_iterations": int(max_iter),
            "quality_threshold": float(threshold),
        }
        handle = start_run(**params)
        st.session_state["_active_run_handle"] = handle
        st.session_state["_active_run_params"] = params
        st.session_state["run_status"] = "running"
        st.session_state["run_logs"] = []
        st.rerun()


# ── 渲染当前运行的实时进度 ──
handle: RunHandle | None = st.session_state.get("_active_run_handle")
status: str = st.session_state.get("run_status", "idle")

st.markdown("### 运行状态")
status_col, ctrl_col = st.columns([3, 1])
status_col.markdown(f"**状态**: {render_status_badge(status)}")

# 阶段流可视化容器
stage_container = st.container()
log_container = st.container()


def _render_stage_timeline(stage_log: List[Dict]) -> None:
    """根据 stage_log 渲染 6 阶段的步骤条。"""
    seen_stages = {entry.get("stage") for entry in stage_log}
    cols = stage_container.columns(len(STAGE_LABELS))
    for index, (stage_key, label) in enumerate(STAGE_LABELS.items()):
        with cols[index]:
            # 该阶段是否已经在 stage_log 里出现过
            if stage_key in seen_stages:
                st.success(label, icon="✅")
                # 显示该阶段的关键信息
                related = [e for e in stage_log if e.get("stage") == stage_key]
                last = related[-1] if related else {}
                meta_lines: List[str] = []
                for key in (
                    "objectives",
                    "research_queries",
                    "tools_available",
                    "brief_len",
                    "iteration",
                    "score",
                    "verdict",
                    "coa_len",
                    "parse_success",
                    "is_valid",
                ):
                    if key in last:
                        meta_lines.append(f"- **{key}**: {last[key]}")
                if meta_lines:
                    st.caption("\n".join(meta_lines))
            else:
                st.info(label, icon="⏳")


if handle is not None:
    # 拉一波最新事件
    new_events = handle.poll_logs()
    log_buffer: List[str] = st.session_state.setdefault("run_logs", [])
    for event in new_events:
        kind = event.get("type", "")
        if kind == "stage_start":
            log_buffer.append(f"▶ {event.get('label', event.get('stage'))}")
        elif kind == "info":
            log_buffer.append(f"ℹ {event.get('message', '')}")
        elif kind == "tool_start":
            log_buffer.append(
                f"🔧 tool: {event.get('tool')} ← {event.get('input', '')[:80]}"
            )
        elif kind == "tool_end":
            log_buffer.append(f"✅ tool result: {event.get('output', '')[:120]}")
        elif kind == "warn":
            log_buffer.append(f"⚠ {event.get('message', '')}")
        elif kind == "error":
            log_buffer.append(f"❌ {event.get('message', '')}")
        elif kind == "done":
            log_buffer.append(
                f"🎉 完成 | iter={event.get('iterations')} "
                f"score={event.get('final_score')} parse={event.get('parse_success')}"
            )

    # 用最新 stage_log 渲染时间线
    if handle.finished and handle.result is not None:
        _render_stage_timeline(handle.result.get("stage_log") or [])
    else:
        # 运行中时只能从 log_buffer 推断已经走到哪一阶段
        # 先把本轮 new_events 里的 stage_start 累积进 _seen_stage_set（用 set 去重，
        # 防止同一阶段被多次 poll 时反复 append 导致 list 无限增长）。
        seen_stages_set: set = st.session_state.setdefault("_seen_stage_set", set())
        for event in new_events:
            if event.get("type") == "stage_start":
                stage_name = event.get("stage")
                if stage_name:
                    seen_stages_set.add(stage_name)
        # 用累积后的 set 渲染当前已经走到的阶段
        _render_stage_timeline(
            [{"stage": stage_name} for stage_name in seen_stages_set]
        )

    # 实时日志窗口
    with log_container:
        st.markdown("### 实时日志")
        st.code("\n".join(log_buffer[-200:]) or "(等待事件…)", language="text")

    # 还在跑就轮询
    if not handle.finished:
        time.sleep(0.5)
        st.rerun()
    else:
        # 完成 → 写回 session_state，清掉 active handle
        if handle.result is not None and handle.error is None:
            set_run_result(
                result=handle.result,
                mission=st.session_state["_active_run_params"]["mission"],
                params=st.session_state["_active_run_params"],
            )
            st.success(
                f"✅ 运行完成。前往 **📊 结果详情** 查看完整产物。"
                f"  迭代={handle.result.get('iterations')} "
                f"得分={handle.result.get('final_score')}"
            )
            st.session_state["run_status"] = "done"
        else:
            st.session_state["run_status"] = "error"
            st.error(f"❌ 运行失败：{handle.error}")
        st.session_state["_active_run_handle"] = None
        st.session_state["_seen_stage_set"] = set()
else:
    st.info("提交一次任务即可开始。运行过程中会在此处实时刷新阶段进度。")
