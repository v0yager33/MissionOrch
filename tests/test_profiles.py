"""运行档位（profile）、配置一致性与 webui 常驻 loop 的测试。"""

from __future__ import annotations

import asyncio
import re
import threading

import pytest
import yaml

from missionorch_lc.agents.analyst import MissionAnalystAgent
from missionorch_lc.agents.base import (
    AgentConfigError,
    apply_profile,
    load_agents_config,
    resolve_profile_name,
)
from missionorch_lc.core.settings import resolve_path

_BASE = {
    "agents": {"planner": {"model_id": "pro", "max_tokens": 6000}, "judge": {"model_id": "flash"}},
    "workflow": {"max_iterations": 3, "use_analyst": True, "default_profile": "balanced"},
    "profiles": {
        "fast": {"agents": {"planner": {"model_id": "flash"}}, "workflow": {"use_analyst": False}},
        "balanced": {},
    },
}


@pytest.fixture(autouse=True)
def _no_env_profile(monkeypatch):
    monkeypatch.delenv("MISSIONORCH_PROFILE", raising=False)


def test_apply_profile_deep_merges():
    merged = apply_profile(_BASE, "fast")
    assert merged["active_profile"] == "fast"
    assert merged["agents"]["planner"] == {"model_id": "flash", "max_tokens": 6000}
    assert merged["agents"]["judge"]["model_id"] == "flash"
    assert merged["workflow"]["use_analyst"] is False
    assert merged["workflow"]["max_iterations"] == 3
    # 原配置不被修改
    assert _BASE["agents"]["planner"]["model_id"] == "pro"


def test_profile_resolution_order(monkeypatch):
    assert resolve_profile_name(_BASE) == "balanced"
    monkeypatch.setenv("MISSIONORCH_PROFILE", "fast")
    assert resolve_profile_name(_BASE) == "fast"
    assert resolve_profile_name(_BASE, "balanced") == "balanced"


def test_unknown_profile_raises():
    with pytest.raises(AgentConfigError):
        apply_profile(_BASE, "turbo")


def test_no_profiles_section_is_noop():
    cfg = {"agents": {"planner": {"model_id": "x"}}, "workflow": {}}
    assert apply_profile(cfg) is cfg


def test_real_agents_yaml_profiles():
    fast = load_agents_config(profile="fast")
    assert fast["active_profile"] == "fast"
    assert fast["workflow"]["use_analyst"] is False
    balanced = load_agents_config(profile="balanced")
    assert balanced["workflow"]["use_reflector"] is False
    assert balanced["workflow"]["revision_mode"] == "section_patch"
    assert balanced["agents"]["planner"]["model_id"] == "deepseek_v4_pro_nothink"
    quality = load_agents_config(profile="quality")
    assert quality["workflow"]["use_reflector"] is True
    assert quality["agents"]["planner"]["model_id"] == "deepseek_v4_pro"


def test_planner_models_thinking_modes():
    with open(resolve_path("config/models.yaml"), encoding="utf-8") as f:
        models = (yaml.safe_load(f) or {}).get("models") or {}
    assert models["deepseek_v4_pro_nothink"]["extra_body"]["thinking"]["type"] == "disabled"
    assert "reasoning_effort" not in models["deepseek_v4_pro_nothink"]
    assert models["deepseek_v4_pro"]["extra_body"]["thinking"]["type"] == "enabled"


def test_all_model_ids_exist():
    with open(resolve_path("config/models.yaml"), encoding="utf-8") as f:
        models = set((yaml.safe_load(f) or {}).get("models") or {})
    for name in ("fast", "balanced", "quality"):
        cfg = load_agents_config(profile=name)
        for agent, agent_cfg in cfg["agents"].items():
            assert agent_cfg["model_id"] in models, f"{name}/{agent}: {agent_cfg['model_id']}"


def test_models_yaml_has_no_inline_keys():
    text = resolve_path("config/models.yaml").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9]{16,}", text)
    for line in text.splitlines():
        if re.match(r"\s*api_key\s*:", line):
            value = line.split(":", 1)[1].strip().strip("'\"")
            # 允许留空、环境变量引用，或本地服务（ollama/vLLM）常用的非密钥占位值
            assert value == "" or value.startswith("${") or value.lower() in ("ollama", "empty", "none"), line.strip()


# ── analyst 输出归一化 ──


def test_analyst_normalize_tolerates_loose_llm_output():
    result = {
        "mission_intent": None,
        "objectives": ["压制雷达", {"description": "打击阵地"}],
        "constraints": "夜间行动",
        "uncertainties": None,
        "key_entities": {"blue_forces": [{"name": "EA-18G"}], "red_forces": "S-400", "areas": None},
    }
    MissionAnalystAgent._normalize(result, "压制X海域防空")
    assert result["mission_intent"] == "压制X海域防空"
    assert [o["id"] for o in result["objectives"]] == ["OBJ-1", "OBJ-2"]
    assert result["objectives"][0]["description"] == "压制雷达"
    assert result["constraints"] == ["夜间行动"]
    assert result["uncertainties"] == [] and result["research_queries"] == []
    assert result["key_entities"] == {
        "blue_forces": ["EA-18G"], "red_forces": ["S-400"], "areas": []
    }


# ── webui 常驻 loop ──


def test_background_loop_is_shared():
    from webui.components import runner

    loop = runner._background_loop()
    assert loop is runner._background_loop()

    async def _probe():
        return threading.current_thread().name

    name = asyncio.run_coroutine_threadsafe(_probe(), loop).result(timeout=5)
    assert name == "missionorch-loop"
    assert loop.is_running()
