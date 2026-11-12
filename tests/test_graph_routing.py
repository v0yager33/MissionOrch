"""LangGraph 编排：路由纯函数 + fake agents 端到端（离线，不调用任何 LLM）。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from missionorch_lc import orchestrator_graph as og
from missionorch_lc.core.expected_outcomes import coerce_outcomes

from .conftest import FULL_COA_TEXT

V2_TEXT = FULL_COA_TEXT.replace("X海域防空压制", "X海域防空压制（修订版）")


# ── 纯函数 ──


def _stop(**kw):
    base = dict(
        score=6.0, structure_passed=True, iteration=1, max_iterations=3,
        threshold=8.0, early_stop=True, no_improve_count=0, patience=1,
    )
    base.update(kw)
    return og._stop_reason(**base)


def test_stop_reason_quality_met():
    assert _stop(score=8.5) == "quality_met"


def test_stop_reason_requires_structure_pass():
    assert _stop(score=9.0, structure_passed=False) == ""


def test_stop_reason_early_stop_disabled():
    assert _stop(score=9.0, early_stop=False) == ""
    assert _stop(early_stop=False, no_improve_count=5) == ""


def test_stop_reason_max_iterations_and_plateau():
    assert _stop(iteration=3) == "max_iterations"
    assert _stop(no_improve_count=1) == "plateau"
    assert _stop(no_improve_count=1, patience=0) == ""


def test_routers():
    assert og._after_planner({"stop_reason": "planner_error"}) == "finalize"
    assert og._after_planner({}) == "judge"
    assert og._should_continue({"stop_reason": "plateau"}) == "finalize"
    assert og._should_continue({"use_reflector": True}) == "reflector"
    assert og._should_continue({"use_reflector": False}) == "planner"


def test_format_instructions():
    assert og._format_instructions(["补充返航阶段", " ", "细化决策点"], "fb") == (
        "1. 补充返航阶段\n2. 细化决策点"
    )
    assert og._format_instructions([], "原始反馈") == "原始反馈"


def test_compose_planner_knowledge_skips_outcomes():
    text = og._compose_planner_knowledge(
        {
            "mission_analysis": {
                "mission_intent": "压制防空",
                "expected_outcomes": [{"description": "不应出现在知识里"}],
                "research_queries": ["也不应出现"],
            },
            "research_brief": "## 研究简报\n证据",
        }
    )
    assert "压制防空" in text and "研究简报" in text
    assert "不应出现在知识里" not in text and "也不应出现" not in text


def test_rule_only_analysis():
    analysis = og._rule_only_analysis("任务\n## 预期战果\n- 击落 J-16 ≥ 4 架\n")
    assert analysis["skipped_llm"] is True
    assert analysis["expected_outcomes"][0]["source"] == "mission_input"


# ── fake agents ──


class _Fake:
    profile = None
    model_id = "fake-model"


class FakeAnalyst(_Fake):
    def __init__(self):
        self.calls = 0

    async def analyze(self, mission):
        self.calls += 1
        return {"mission_intent": "压制防空", "objectives": ["o1"], "research_queries": ["q"],
                "expected_outcomes": []}


class FakeResearcher(_Fake):
    def _fallback_brief(self, analysis, reason=""):
        return f"（跳过调研：{reason}）"

    async def research(self, analysis, tools):  # pragma: no cover - use_rag=False 时不走
        raise AssertionError("research should be skipped")


class FakePlanner(_Fake):
    revision_mode = "full"
    last_truncated = False
    last_patched_sections: List[str] = []

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: List[Dict[str, Any]] = []

    async def generate_coa(self, mission, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeJudge(_Fake):
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    async def evaluate(self, coa_text, mission, structure_report=""):
        self.calls += 1
        score = self.scores.pop(0)
        return score, f"score={score}", {
            "verdict": "ACCEPT" if score >= 8 else "REVISE",
            "revision_instructions": ["细化第三阶段行动"],
        }


class FakeReflector(_Fake):
    async def reflect(self, coa_text, feedback, iteration):  # pragma: no cover
        return feedback


class FakeValidator(_Fake):
    async def validate_and_extract_matrix(self, coa_obj, mission, raw_text=""):
        return True, "ok", {}


def _install(monkeypatch, *, planner_outputs, scores, workflow=None):
    agents = {
        "analyst": FakeAnalyst(),
        "researcher": FakeResearcher(),
        "planner": FakePlanner(planner_outputs),
        "judge": FakeJudge(scores),
        "reflector": FakeReflector(),
        "validator": FakeValidator(),
    }
    wf = {"use_reflector": False, "use_analyst": True, "min_improvement": 0.2, "patience": 1}
    wf.update(workflow or {})
    monkeypatch.delenv("MISSIONORCH_PROFILE", raising=False)
    monkeypatch.setattr(og, "get_agents", lambda profile=None: agents)
    monkeypatch.setattr(og, "load_agents_config", lambda *a, **k: {"workflow": wf})
    monkeypatch.setattr(og, "save_run_memory", lambda *a, **k: None)
    return agents


def _run(mission="压制X海域防空", **kwargs):
    params = dict(use_rag=False, max_iterations=3, quality_threshold=8.0, early_stop=True)
    params.update(kwargs)
    return asyncio.run(og.run_graph(mission, **params))


def test_graph_quality_met_passes_outcomes_through(monkeypatch):
    agents = _install(monkeypatch, planner_outputs=[FULL_COA_TEXT], scores=[8.5])
    result = _run(expected_outcomes=["击落 J-16 ≥ 4 架"])

    assert result["stop_reason"] == "quality_met"
    assert result["iterations"] == 1 and result["best_iteration"] == 1
    assert result["final_score"] == 8.5
    assert result["parse_success"] is True
    assert agents["judge"].calls == 1

    # 输入侧：用户给定条目进入 planner，来源保持 user
    sent = agents["planner"].calls[0]["expected_outcomes"]
    assert [o["source"] for o in sent] == ["user"]
    assert result["input_expected_outcomes"][0]["description"] == "击落 J-16 ≥ 4 架"

    # 输出侧：方案第八节 + 输入条目，一条不丢；预留字段未被填
    outcomes = result["expected_outcomes"]
    assert [o["outcome_id"] for o in outcomes] == ["EO-1", "EO-2", "EO-3"]
    assert outcomes[-1]["source"] == "user"
    assert all(o["actual_value"] is None and o["status"] == "PENDING" for o in outcomes)
    assert len(result["final_coa"]["expected_outcomes"]) == 3
    assert "预期战果" in result["markdown"] and "J-16" in result["markdown"]

    assert set(result["stage_timing"]) >= {"analyst", "researcher", "planner", "judge", "finalize"}
    assert result["elapsed_seconds"] >= 0


def test_graph_plateau_keeps_best(monkeypatch):
    agents = _install(monkeypatch, planner_outputs=[FULL_COA_TEXT, V2_TEXT], scores=[7.0, 6.5])
    result = _run()

    assert result["stop_reason"] == "plateau"
    assert result["iterations"] == 2
    assert result["best_iteration"] == 1 and result["final_score"] == 7.0
    assert result["coa_table"] == FULL_COA_TEXT
    # 第二轮在 v1 上修订，并收到 judge 的修订指令
    second = agents["planner"].calls[1]
    assert second["previous_coa_text"] == FULL_COA_TEXT
    assert "细化第三阶段行动" in second["reflection"]
    assert [h["is_best"] for h in result["history"]] == [True, False]


def test_graph_max_iterations(monkeypatch):
    _install(
        monkeypatch,
        planner_outputs=[FULL_COA_TEXT, V2_TEXT, V2_TEXT],
        scores=[6.0, 7.0, 7.5],
        workflow={"patience": 0},
    )
    result = _run()
    assert result["stop_reason"] == "max_iterations"
    assert result["best_iteration"] == 3 and result["final_score"] == 7.5


def test_graph_planner_error_keeps_best(monkeypatch):
    _install(
        monkeypatch,
        planner_outputs=[FULL_COA_TEXT, RuntimeError("timeout")],
        scores=[7.0],
    )
    result = _run()
    assert result["stop_reason"] == "planner_error"
    assert result["coa_table"] == FULL_COA_TEXT
    assert result["final_score"] == 7.0 and result["parse_success"] is True
    assert any(e.get("error") for e in result["stage_log"] if e["stage"] == "planner")


def test_graph_first_planner_error_raises(monkeypatch):
    _install(monkeypatch, planner_outputs=[RuntimeError("boom")], scores=[])
    with pytest.raises(RuntimeError):
        _run()


def test_graph_rule_only_analyst(monkeypatch):
    agents = _install(
        monkeypatch,
        planner_outputs=[FULL_COA_TEXT],
        scores=[9.0],
        workflow={"use_analyst": False},
    )
    mission = "压制X海域防空\n\n## 预期战果\n- 击落 J-16 ≥ 4 架\n"
    result = _run(mission)
    assert agents["analyst"].calls == 0
    assert result["mission_analysis"]["skipped_llm"] is True
    assert result["input_expected_outcomes"][0]["source"] == "mission_input"
    assert any("J-16" in o["description"] for o in result["expected_outcomes"])


def test_user_outcomes_win_over_mission_text(monkeypatch):
    _install(
        monkeypatch,
        planner_outputs=[FULL_COA_TEXT],
        scores=[9.0],
        workflow={"use_analyst": False},
    )
    mission = "压制X海域防空\n\n## 预期战果\n- 击落 J-16 ≥ 4 架\n"
    result = _run(mission, expected_outcomes="击落 J-16 ≥ 4 架")
    inputs = result["input_expected_outcomes"]
    assert len(inputs) == 1 and inputs[0]["source"] == "user"


def test_coerce_matches_run_graph_input():
    # run_graph 内部用 source=user 归一化，webui / CLI 传字符串列表即可
    assert coerce_outcomes(["击落 J-16 ≥ 4 架"], source="user")[0]["comparator"] == ">="
