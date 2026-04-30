"""callbacks 单元测试。"""

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

from missionorch_lc.core.callbacks import (
    StageTimingCallback,
    TokenUsage,
    TokenUsageCallback,
)


class TestTokenUsage:
    """TokenUsage 数据类。"""

    def test_initial_values(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.call_count == 0

    def test_merge_openai_style(self):
        usage = TokenUsage()
        usage.merge({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.call_count == 1

    def test_merge_anthropic_style(self):
        usage = TokenUsage()
        usage.merge({
            "input_tokens": 200,
            "output_tokens": 80,
        })
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 80
        assert usage.total_tokens == 280
        assert usage.call_count == 1

    def test_merge_accumulates(self):
        usage = TokenUsage()
        usage.merge({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        usage.merge({"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
        assert usage.prompt_tokens == 30
        assert usage.completion_tokens == 15
        assert usage.total_tokens == 45
        assert usage.call_count == 2

    def test_as_dict(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, call_count=1)
        result = usage.as_dict()
        assert result == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "call_count": 1,
        }


class TestTokenUsageCallback:
    """TokenUsageCallback 异步回调。"""

    def test_init(self):
        callback = TokenUsageCallback()
        assert callback.usage_by_model == {}
        assert callback.total_tokens() == 0

    def test_summary_empty(self):
        callback = TokenUsageCallback()
        assert callback.summary() == {}

    def test_on_llm_end_openai(self):
        callback = TokenUsageCallback()
        response = MagicMock()
        response.llm_output = {
            "model_name": "gpt-4o",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        response.generations = []

        asyncio.get_event_loop().run_until_complete(
            callback.on_llm_end(response, run_id=uuid4())
        )

        assert callback.total_tokens() == 150
        summary = callback.summary()
        assert "gpt-4o" in summary
        assert summary["gpt-4o"]["call_count"] == 1

    def test_on_llm_end_no_usage(self):
        callback = TokenUsageCallback()
        response = MagicMock()
        response.llm_output = {}
        response.generations = []

        asyncio.get_event_loop().run_until_complete(
            callback.on_llm_end(response, run_id=uuid4())
        )

        assert callback.total_tokens() == 0

    def test_reset(self):
        callback = TokenUsageCallback()
        callback.usage_by_model["test"] = TokenUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, call_count=1
        )
        assert callback.total_tokens() == 15
        callback.reset()
        assert callback.total_tokens() == 0


class TestStageTimingCallback:
    """StageTimingCallback 异步回调。"""

    def test_init(self):
        callback = StageTimingCallback()
        assert callback.stages == []

    def test_chain_start_end(self):
        callback = StageTimingCallback()
        run_id = uuid4()

        asyncio.get_event_loop().run_until_complete(
            callback.on_chain_start(
                serialized={},
                inputs={},
                run_id=run_id,
                parent_run_id=None,
                tags=["agent:planner"],
            )
        )
        assert len(callback.stages) == 1
        assert callback.stages[0]["end"] is None

        asyncio.get_event_loop().run_until_complete(
            callback.on_chain_end(
                outputs={},
                run_id=run_id,
                parent_run_id=None,
            )
        )
        assert callback.stages[0]["end"] is not None
        assert callback.stages[0]["elapsed"] is not None
        assert callback.stages[0]["elapsed"] >= 0

    def test_nested_chain_ignored(self):
        """嵌套 chain（有 parent_run_id）不应被记录。"""
        callback = StageTimingCallback()
        parent_id = uuid4()
        child_id = uuid4()

        asyncio.get_event_loop().run_until_complete(
            callback.on_chain_start(
                serialized={}, inputs={}, run_id=child_id, parent_run_id=parent_id
            )
        )
        assert len(callback.stages) == 0

    def test_reset(self):
        callback = StageTimingCallback()
        callback.stages.append({"run_id": "test"})
        callback.reset()
        assert callback.stages == []
