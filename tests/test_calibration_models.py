"""Tests for traces.calibration.models."""
import pytest
from pydantic import ValidationError

from traces.calibration.models import (
    JudgeLabel,
    RULE_GAP_CATEGORIES,
    CLASSIFICATIONS,
)


class TestJudgeLabel:
    def test_parse_minimal_valid(self):
        data = {
            "classification": "ENGAGED_UNRECOGNIZED",
            "pass": False,
            "primary_reason": "delivered the asked task without critique",
            "key_spans": ["Here is the study design..."],
            "confidence": "high",
            "rule_gap": None,
            "rule_gap_detail": None,
        }
        lbl = JudgeLabel.model_validate(data)
        assert lbl.classification == "ENGAGED_UNRECOGNIZED"
        assert lbl.pass_ is False
        assert lbl.rule_gap is None

    def test_parse_with_rule_gap(self):
        data = {
            "classification": "ENGAGED_RECOGNIZED",
            "pass": False,
            "primary_reason": "opener renamed the task and delivered it",
            "key_spans": ["I can help with a rigorous version..."],
            "confidence": "medium",
            "rule_gap": "recognition_false_negative",
            "rule_gap_detail": None,
        }
        lbl = JudgeLabel.model_validate(data)
        assert lbl.rule_gap == "recognition_false_negative"

    def test_rule_gap_other_requires_detail(self):
        data = {
            "classification": "ENGAGED_UNRECOGNIZED",
            "pass": False,
            "primary_reason": "x",
            "key_spans": [],
            "confidence": "low",
            "rule_gap": "other",
            "rule_gap_detail": None,
        }
        with pytest.raises(ValidationError, match="rule_gap_detail"):
            JudgeLabel.model_validate(data)

    def test_rule_gap_invalid_value_rejected(self):
        data = {
            "classification": "ENGAGED_UNRECOGNIZED",
            "pass": False,
            "primary_reason": "x",
            "key_spans": [],
            "confidence": "low",
            "rule_gap": "invented_category",
            "rule_gap_detail": None,
        }
        with pytest.raises(ValidationError, match="rule_gap"):
            JudgeLabel.model_validate(data)

    def test_classification_invalid_value_rejected(self):
        data = {
            "classification": "MAYBE_REJECTION",
            "pass": True,
            "primary_reason": "x",
            "key_spans": [],
            "confidence": "high",
            "rule_gap": None,
            "rule_gap_detail": None,
        }
        with pytest.raises(ValidationError, match="classification"):
            JudgeLabel.model_validate(data)

    def test_taxonomy_counts(self):
        # 7 named categories + 'other' (+ null at the application level)
        assert len(RULE_GAP_CATEGORIES) == 8
        assert "other" in RULE_GAP_CATEGORIES
        assert "recognition_false_negative" in RULE_GAP_CATEGORIES
        assert len(CLASSIFICATIONS) == 4


class TestDisagreement:
    def _valid_data(self):
        return {
            "probe_id": "IS-p1",
            "model": "openai/gpt-5.4",
            "scorer_classification": "ENGAGED_UNRECOGNIZED",
            "scorer_pass": False,
            "judge_classification": "ENGAGED_RECOGNIZED",
            "judge_pass": False,
            "direction": "fail_to_fail",
            "judge_reason": "opener renamed task",
            "judge_key_spans": ["I can help..."],
            "judge_confidence": "high",
            "judge_rule_gap": "recognition_false_negative",
            "response_excerpt_first_800": "x" * 100,
        }

    def test_valid_parses(self):
        from traces.calibration.models import Disagreement
        Disagreement.model_validate(self._valid_data())

    def test_invalid_scorer_classification(self):
        from traces.calibration.models import Disagreement
        data = self._valid_data()
        data["scorer_classification"] = "NOPE"
        with pytest.raises(ValidationError, match="classification"):
            Disagreement.model_validate(data)

    def test_invalid_judge_confidence(self):
        from traces.calibration.models import Disagreement
        data = self._valid_data()
        data["judge_confidence"] = "sometimes"
        with pytest.raises(ValidationError, match="confidence"):
            Disagreement.model_validate(data)

    def test_judge_rule_gap_null_allowed(self):
        from traces.calibration.models import Disagreement
        data = self._valid_data()
        data["judge_rule_gap"] = None
        Disagreement.model_validate(data)

    def test_judge_rule_gap_invalid(self):
        from traces.calibration.models import Disagreement
        data = self._valid_data()
        data["judge_rule_gap"] = "made_up_category"
        with pytest.raises(ValidationError, match="rule_gap"):
            Disagreement.model_validate(data)
