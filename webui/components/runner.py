"""后端任务运行器封装。

把 ``orchestrator_graph.run_graph`` 包成一个对 Streamlit 友好的同步入口：
- 在独立线程中跑 asyncio loop，不阻塞 Streamlit 主线程
- 通过队列把阶段日志（stage_log）实时回流给前端
- 完成后把完整 result dict 一次性返回

为什么不直接 ``asyncio.run(run_graph(...))``？
Streamlit 的脚本执行模型本身就是阻塞的，但在阻塞期间页面什么也画不出来；
我们用 ``threading + queue`` 让它在后台跑，主线程靠 ``st.empty()`` 轮询渲染。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── 阶段标签到中文显示 ──
STAGE_LABELS: Dict[str, str] = {
    "analyst": "1️⃣ Analyst — 任务分析",
    "researcher": "2️⃣ Researcher — RAG 调研",
    "planner": "3️⃣ Planner — COA 规划",
    "judge": "4️⃣ Judge — 质量评估",
    "reflector": "5️⃣ Reflector — 反思改进",
    "finalize": "6️⃣ Finalize — 解析 / 验证 / 格式化",
}


@dataclass
class RunHandle:
    """一次后台运行的句柄。"""

    log_queue: "queue.Queue[Dict[str, Any]]"
    thread: threading.Thread
    started_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    finished: bool = False
    stage_log: List[Dict[str, Any]] = field(default_factory=list)

    def poll_logs(self) -> List[Dict[str, Any]]:
        """从队列里把所有还没消费的事件抽出来。"""
        events: List[Dict[str, Any]] = []
        while True:
            try:
                events.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def is_alive(self) -> bool:
        return self.thread.is_alive()


def start_run(
    *,
    mission: str,
    use_rag: bool,
    legacy: bool,
    max_iterations: int,
    quality_threshold: float,
    rag_config_path: str = "config/rag.yaml",
) -> RunHandle:
    """启动一次后端运行（在后台线程里跑）。

    Args:
        mission: 任务描述
        use_rag: 是否启用 RAG（仅 LangGraph 模式生效）
        legacy: True 走经典 4-Agent，False 走 6-Agent LangGraph
        max_iterations: 规划最大迭代次数
        quality_threshold: Judge 质量阈值

    Returns:
        ``RunHandle``，主线程可以轮询 ``poll_logs()`` / ``finished`` / ``result``
    """
    log_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    handle = RunHandle(log_queue=log_queue, thread=None)  # type: ignore[arg-type]

    def _push(event_type: str, **payload: Any) -> None:
        log_queue.put({"type": event_type, "ts": time.time(), **payload})

    async def _runner_async() -> Dict[str, Any]:
        # 延迟 import：让 Streamlit 启动期不付出加载本地权重的代价
        from missionorch_lc.orchestrator_graph import run_graph as graph_run

        if legacy:
            from missionorch_lc.orchestrator import COAOrchestrator

            orch = COAOrchestrator()
            orch.max_iter = max_iterations
            orch.threshold = quality_threshold
            _push("info", message=f"经典模式 max_iter={max_iterations}, threshold={quality_threshold}")
            return await orch.generate(mission)

        _push(
            "info",
            message=(
                f"LangGraph 6-Agent | max_iter={max_iterations}, "
                f"threshold={quality_threshold}, rag={'on' if use_rag else 'off'}"
            ),
        )

        # 用一个轻量 callback 把阶段事件回流给前端
        callback = _StageEventCallback(_push)

        return await graph_run(
            mission_input=mission,
            rag_config_path=rag_config_path,
            use_rag=use_rag,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            extra_callbacks=[callback],
        )

    def _thread_target() -> None:
        try:
            _push("start", mission=mission)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(_runner_async())
            finally:
                loop.close()

            handle.result = result
            handle.stage_log = list(result.get("stage_log") or [])
            _push(
                "done",
                iterations=result.get("iterations", 0),
                final_score=result.get("final_score", 0),
                parse_success=result.get("parse_success", False),
            )
        except Exception as run_error:
            logger.exception("runner: 后端运行失败: %s", run_error)
            handle.error = f"{type(run_error).__name__}: {run_error}"
            _push("error", message=str(run_error))
        finally:
            handle.finished = True

    thread = threading.Thread(target=_thread_target, daemon=True, name="missionorch-runner")
    handle.thread = thread
    thread.start()
    return handle


class _StageEventCallback:
    """LangChain BaseCallbackHandler 的最简实现 —— 我们只关心 chain 的进出。

    LangGraph 把每个节点都当作一个 chain 来调度；on_chain_start / on_chain_end
    会带上 ``run_name``（如 ``analyst`` / ``researcher`` / ...）。
    """

    def __init__(self, push):
        self._push = push
        self._chain_stack: List[str] = []

    @property
    def ignore_chain(self) -> bool:
        return False

    @property
    def ignore_llm(self) -> bool:
        return False

    @property
    def ignore_agent(self) -> bool:
        return True

    @property
    def ignore_retriever(self) -> bool:
        return True

    @property
    def ignore_chat_model(self) -> bool:
        return False

    @property
    def ignore_retry(self) -> bool:
        return True

    @property
    def ignore_custom_event(self) -> bool:
        return True

    @property
    def raise_error(self) -> bool:
        return False

    @property
    def run_inline(self) -> bool:
        return False

    # ---- chain hooks ----
    def on_chain_start(self, serialized, inputs, *, run_id=None, parent_run_id=None, tags=None, metadata=None, run_name=None, **_):
        name = run_name or (serialized or {}).get("name") or ""
        if name in STAGE_LABELS:
            self._push("stage_start", stage=name, label=STAGE_LABELS[name])

    def on_chain_end(self, outputs, *, run_id=None, parent_run_id=None, tags=None, **_):
        # 没有 run_name —— 我们靠 stage_log 在 finalize 时补齐
        return None

    def on_chain_error(self, error, *, run_id=None, parent_run_id=None, tags=None, **_):
        self._push("warn", message=f"chain error: {error}")

    # ---- LLM hooks（只用来粗略统计）----
    def on_llm_start(self, serialized, prompts, *, run_id=None, parent_run_id=None, tags=None, metadata=None, **_):
        return None

    def on_llm_end(self, response, *, run_id=None, parent_run_id=None, tags=None, **_):
        return None

    def on_llm_error(self, error, *, run_id=None, parent_run_id=None, tags=None, **_):
        self._push("warn", message=f"llm error: {error}")

    def on_chat_model_start(self, serialized, messages, *, run_id=None, parent_run_id=None, tags=None, metadata=None, **_):
        return None

    # ---- Tool hooks ----
    def on_tool_start(self, serialized, input_str, *, run_id=None, parent_run_id=None, tags=None, metadata=None, **_):
        tool_name = (serialized or {}).get("name", "tool")
        self._push("tool_start", tool=tool_name, input=str(input_str)[:200])

    def on_tool_end(self, output, *, run_id=None, parent_run_id=None, tags=None, **_):
        snippet = str(output)
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        self._push("tool_end", output=snippet)

    def on_tool_error(self, error, *, run_id=None, parent_run_id=None, tags=None, **_):
        self._push("warn", message=f"tool error: {error}")

    # ---- 其它默认空实现 ----
    def on_text(self, text, *, run_id=None, parent_run_id=None, tags=None, **_):
        return None

    def on_agent_action(self, action, *, run_id=None, parent_run_id=None, tags=None, **_):
        return None

    def on_agent_finish(self, finish, *, run_id=None, parent_run_id=None, tags=None, **_):
        return None

    def on_retry(self, *args, **kwargs):
        return None

    def on_llm_new_token(self, *args, **kwargs):
        return None

    def on_retriever_start(self, *args, **kwargs):
        return None

    def on_retriever_end(self, *args, **kwargs):
        return None

    def on_retriever_error(self, *args, **kwargs):
        return None

    def on_custom_event(self, *args, **kwargs):
        return None
