"""PlannerAgent 的 section_patch 修订与 prompt 组装（不调用 LLM）。"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from missionorch_lc.agents.planner import PlannerAgent
from missionorch_lc.core.expected_outcomes import coerce_outcomes
from missionorch_lc.core.settings import resolve_path

from .conftest import FULL_COA_TEXT

PATCH = (
    "## 一、方案概述\n"
    "改为夜间低空突防：电子战编队先压制预警雷达与火控雷达，打击编队随后对 SAM 阵地实施精确打击，"
    "预警机全程指挥协同，完成任务后各编队按序脱离并返航着舰。\n"
)


def _planner(monkeypatch, reply: str, *, revision_mode: str = "section_patch", finish: str = "stop"):
    planner = PlannerAgent.__new__(PlannerAgent)
    planner.agent_type = "planner"
    planner.model_id = "fake-model"
    planner.prompt_path = str(resolve_path("src/missionorch_lc/prompts/planner.txt"))
    planner.revision_mode = revision_mode
    planner.last_finish_reason = ""
    planner.last_truncated = False
    planner.last_patched_sections = []
    planner.prompts: List[str] = []

    def _reply(prompt_value: Any) -> AIMessage:
        planner.prompts.append(prompt_value.to_string())
        return AIMessage(content=reply, response_metadata={"finish_reason": finish})

    monkeypatch.setattr(planner, "bind_model", lambda tools=None, **_: RunnableLambda(_reply))
    monkeypatch.setattr(planner, "log_interaction", lambda *a, **k: None)
    return planner


def _generate(planner, **kwargs):
    return asyncio.run(planner.generate_coa("压制X海域防空", **kwargs))


def test_patch_mode_merges_sections(monkeypatch):
    planner = _planner(monkeypatch, PATCH)
    out = _generate(planner, previous_coa_text=FULL_COA_TEXT, reflection="1. 改为夜间突防")
    assert planner.last_patched_sections == ["一、方案概述"]
    assert "夜间低空突防" in out
    # 未修改的章节（含第八节预期战果）原样保留
    assert "## 八、预期战果（Expected Outcomes）" in out
    assert "| EO-2 | 蓝方战损可控 |" in out
    assert out.startswith("# COA方案：X海域防空压制")
    assert "【输出方式：只输出修改的章节】" in planner.prompts[0]
    assert "【上一版COA" in planner.prompts[0]


def test_patch_mode_accepts_full_rewrite(monkeypatch):
    rewritten = FULL_COA_TEXT.replace("X海域防空压制", "X海域防空压制（重写）")
    planner = _planner(monkeypatch, rewritten)
    out = _generate(planner, previous_coa_text=FULL_COA_TEXT)
    assert out == rewritten
    assert planner.last_patched_sections == []


def test_full_mode_ignores_patch(monkeypatch):
    planner = _planner(monkeypatch, FULL_COA_TEXT, revision_mode="full")
    assert not planner._use_patch(FULL_COA_TEXT, None)
    assert planner._use_patch(FULL_COA_TEXT, "section_patch")
    # 首轮没有上一版，即使配置为 section_patch 也是全量输出
    assert not _planner(monkeypatch, FULL_COA_TEXT)._use_patch(None, None)


def test_prompt_contains_expected_outcomes_and_structure(monkeypatch):
    planner = _planner(monkeypatch, FULL_COA_TEXT, revision_mode="full")
    outcomes = coerce_outcomes("击落 J-16 ≥ 4 架", source="user")
    _generate(planner, expected_outcomes=outcomes, structure_report="- 必须修正：阶段数 4 < 5")
    prompt = planner.prompts[0]
    assert "击落 J-16" in prompt and "输入给定的预期战果" in prompt
    assert "阶段数 4 < 5" in prompt
    assert "【输出方式：只输出修改的章节】" not in prompt


def test_truncation_flag(monkeypatch):
    planner = _planner(monkeypatch, FULL_COA_TEXT, revision_mode="full", finish="length")
    _generate(planner)
    assert planner.last_truncated is True


def test_too_short_reply_raises(monkeypatch):
    planner = _planner(monkeypatch, "太短", revision_mode="full")
    with pytest.raises(ValueError):
        _generate(planner)
