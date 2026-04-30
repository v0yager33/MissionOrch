"""Schema 数据模型单元测试。"""

from missionorch_lc.schemas.coa import COA, Action, DecisionPoint, Effect, Phase, Unit
from missionorch_lc.schemas.judge_result import JudgeResult
from missionorch_lc.schemas.validation_result import ValidationResult


class TestCOASchema:
    """COA 及其子模型。"""

    def test_coa_defaults(self):
        coa = COA()
        assert coa.coa_id.startswith("COA-")
        assert coa.name == "Generated COA"
        assert coa.phases == []
        assert coa.units == []
        assert coa.matrix == []

    def test_phase(self):
        phase = Phase(phase_id="P1", name="Deploy")
        assert phase.phase_id == "P1"
        assert phase.transition_trigger == ""

    def test_unit(self):
        unit = Unit(unit_id="U1", name="Air Force")
        assert unit.role == ""

    def test_action(self):
        action = Action(unit_id="U1", phase_id="P1", actions=["attack", "defend"])
        assert len(action.actions) == 2

    def test_effect(self):
        effect = Effect(effect_id="E1", description="Air superiority")
        assert effect.measures == []
        assert effect.achieved_by == []

    def test_decision_point(self):
        dp = DecisionPoint(dp_id="DP1", condition="enemy reinforcement")
        assert dp.options == []

    def test_coa_get_actions(self):
        coa = COA(
            matrix=[
                Action(unit_id="U1", phase_id="P1", actions=["move"]),
                Action(unit_id="U1", phase_id="P2", actions=["attack"]),
            ]
        )
        assert coa.get_actions("U1", "P1") == ["move"]
        assert coa.get_actions("U1", "P2") == ["attack"]
        assert coa.get_actions("U2", "P1") == []

    def test_coa_get_unit_actions(self):
        coa = COA(
            matrix=[
                Action(unit_id="U1", phase_id="P1", actions=["a"]),
                Action(unit_id="U1", phase_id="P2", actions=["b"]),
                Action(unit_id="U2", phase_id="P1", actions=["c"]),
            ]
        )
        result = coa.get_unit_actions("U1")
        assert result == {"P1": ["a"], "P2": ["b"]}

    def test_coa_serialization(self):
        coa = COA(name="Test COA", description="desc")
        data = coa.model_dump(mode="json")
        assert data["name"] == "Test COA"
        assert isinstance(data["phases"], list)


class TestJudgeResult:
    """JudgeResult schema。"""

    def test_minimal(self):
        result = JudgeResult(overall_score=7.5, feedback="Good", verdict="ACCEPT")
        assert result.overall_score == 7.5


class TestValidationResult:
    """ValidationResult schema。"""

    def test_minimal(self):
        result = ValidationResult(
            is_valid=True, validation_feedback="OK", issues_found=[]
        )
        assert result.is_valid is True
