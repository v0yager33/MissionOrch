"""LangChain Callback Handlers —— 用于 token 统计与调试。

用法：
    callback = TokenUsageCallback()
    chain.ainvoke(..., config={"callbacks": [callback]})
    print(callback.summary())
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class TokenUsage:
    """按模型累计的 token 用量。"""

    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens", "call_count")

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        call_count: int = 0,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.call_count = call_count

    def merge(self, usage: Dict[str, Any]) -> None:
        """合并来自不同 provider 的 usage 字典。

        兼容字段：
          - OpenAI: prompt_tokens / completion_tokens / total_tokens
          - Anthropic / Gemini: input_tokens / output_tokens
        """
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.call_count += 1

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


class TokenUsageCallback(AsyncCallbackHandler):
    """累计每个模型的 token 消耗（协程安全）。"""

    # LangChain 会检查这些类属性
    raise_error: bool = False
    run_inline: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.usage_by_model: Dict[str, TokenUsage] = {}
        self._lock = Lock()

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """LLM 调用结束时收集 usage。"""
        llm_output = response.llm_output or {}
        model_name = (
            llm_output.get("model_name")
            or llm_output.get("model")
            or "unknown"
        )

        # OpenAI / OpenAI-compatible 把 usage 放在 llm_output["token_usage"]
        token_usage: Dict[str, Any] = dict(
            llm_output.get("token_usage") or llm_output.get("usage") or {}
        )

        # Anthropic / Gemini / 新版 LangChain 把 usage 放在 AIMessage.usage_metadata
        if not token_usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    message = getattr(gen, "message", None)
                    usage_metadata = getattr(message, "usage_metadata", None) if message else None
                    if not usage_metadata:
                        generation_info = getattr(gen, "generation_info", None) or {}
                        usage_metadata = generation_info.get("usage_metadata")
                    if usage_metadata:
                        token_usage = dict(usage_metadata)
                        break
                if token_usage:
                    break

        if not token_usage:
            return

        with self._lock:
            usage = self.usage_by_model.setdefault(model_name, TokenUsage())
            usage.merge(token_usage)

    def total_tokens(self) -> int:
        with self._lock:
            return sum(u.total_tokens for u in self.usage_by_model.values())

    def summary(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {model: u.as_dict() for model, u in self.usage_by_model.items()}

    def reset(self) -> None:
        with self._lock:
            self.usage_by_model.clear()


class StageTimingCallback(AsyncCallbackHandler):
    """记录每个顶层 chain 的耗时，供进度展示使用。"""

    raise_error: bool = False
    run_inline: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.stages: List[Dict[str, Any]] = []
        self._lock = Lock()

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is not None:
            return
        stage = {
            "run_id": str(run_id),
            "tags": list(tags or []),
            "run_name": kwargs.get("name", ""),
            "start": time.time(),
            "end": None,
            "elapsed": None,
        }
        with self._lock:
            self.stages.append(stage)

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is not None:
            return
        now = time.time()
        rid = str(run_id)
        with self._lock:
            for stage in reversed(self.stages):
                if stage["run_id"] == rid and stage["end"] is None:
                    stage["end"] = now
                    stage["elapsed"] = now - stage["start"]
                    break

    def summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.stages)

    def reset(self) -> None:
        with self._lock:
            self.stages.clear()

