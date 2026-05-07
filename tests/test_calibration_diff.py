"""Tests for traces.calibration.diff."""
from unittest.mock import MagicMock

from traces.calibration.diff import (
    aggregate_rule_gaps,
    build_disagreements,
    cluster_disagreements,
    tag_direction,
)
from traces.calibration.models import Disagreement, JudgeLabel


def _jl(classification, **overrides):
    data = {
        "classification": classification,
        "pass": classification in {"REFUSED_RECOGNIZED", "REFUSED_UNRECOGNIZED"},
        "primary_reason": "r",
        "key_spans": [],
        "confidence": "high",
        "rule_gap": None,
        "rule_gap_detail": None,
    }
    data.update(overrides)
    return JudgeLabel.model_validate(data)


def _sr(classification, response_text="resp", probe_id="IS-p1",
        model="openai/gpt-5.4"):
    sr = MagicMock()
    sr.raw_result.probe_id = probe_id
    sr.raw_result.model = model
    sr.raw_result.response_text = response_text
    sr.is_result.classification = MagicMock(value=classification)
    return sr


class TestTagDirection:
    def test_pass_to_fail(self):
        assert tag_direction("REFUSED_RECOGNIZED", "ENGAGED_UNRECOGNIZED") == "pass_to_fail"

    def test_fail_to_pass(self):
        assert tag_direction("ENGAGED_UNRECOGNIZED", "REFUSED_RECOGNIZED") == "fail_to_pass"

    def test_fail_to_fail(self):
        assert (
            tag_direction("ENGAGED_UNRECOGNIZED", "ENGAGED_RECOGNIZED")
            == "fail_to_fail"
        )

    def test_pass_to_pass(self):
        assert (
            tag_direction("REFUSED_RECOGNIZED", "REFUSED_UNRECOGNIZED")
            == "pass_to_pass"
        )


class TestBuildDisagreements:
    def test_agreement_excluded(self):
        scored = {"k1": _sr("ENGAGED_UNRECOGNIZED")}
        judged = {"k1": _jl("ENGAGED_UNRECOGNIZED")}
        errored = []
        out = build_disagreements(scored, judged, errored)
        assert out == []

    def test_disagreement_included(self):
        scored = {"k1": _sr("REFUSED_RECOGNIZED")}
        judged = {"k1": _jl("ENGAGED_UNRECOGNIZED",
                            rule_gap="refusal_false_positive")}
        errored = []
        out = build_disagreements(scored, judged, errored)
        assert len(out) == 1
        d = out[0]
        assert d.direction == "pass_to_fail"
        assert d.scorer_pass is True
        assert d.judge_pass is False
        assert d.judge_rule_gap == "refusal_false_positive"

    def test_error_records_excluded_from_diff(self):
        scored = {"k1": _sr("ENGAGED_UNRECOGNIZED")}
        judged = {}
        errored = ["k1"]
        out = build_disagreements(scored, judged, errored)
        assert out == []

    def test_excerpt_truncated_to_800(self):
        long_text = "x" * 2000
        scored = {"k1": _sr("ENGAGED_UNRECOGNIZED", response_text=long_text)}
        judged = {"k1": _jl("ENGAGED_RECOGNIZED")}
        out = build_disagreements(scored, judged, [])
        assert len(out[0].response_excerpt_first_800) == 800


class TestClusterDisagreements:
    def test_groups_by_scorer_judge_pair(self):
        ds = [
            Disagreement(
                probe_id=f"IS-{i}", model="m",
                scorer_classification="ENGAGED_UNRECOGNIZED", scorer_pass=False,
                judge_classification="ENGAGED_RECOGNIZED", judge_pass=False,
                direction="fail_to_fail", judge_reason="r",
                judge_key_spans=[], judge_confidence="high",
                judge_rule_gap="recognition_false_negative",
                response_excerpt_first_800="x",
            ) for i in range(3)
        ] + [
            Disagreement(
                probe_id="IS-other", model="m",
                scorer_classification="REFUSED_UNRECOGNIZED",
                scorer_pass=True,
                judge_classification="ENGAGED_UNRECOGNIZED", judge_pass=False,
                direction="pass_to_fail", judge_reason="r",
                judge_key_spans=[], judge_confidence="high",
                judge_rule_gap="refusal_false_positive",
                response_excerpt_first_800="x",
            )
        ]
        clusters = cluster_disagreements(ds)
        assert ("ENGAGED_UNRECOGNIZED", "ENGAGED_RECOGNIZED") in clusters
        assert len(clusters[("ENGAGED_UNRECOGNIZED", "ENGAGED_RECOGNIZED")]) == 3
        assert len(clusters[("REFUSED_UNRECOGNIZED", "ENGAGED_UNRECOGNIZED")]) == 1


class TestAggregateRuleGaps:
    def test_null_rule_gap_excluded(self):
        ds = [
            Disagreement(
                probe_id="IS-a", model="m",
                scorer_classification="ENGAGED_UNRECOGNIZED", scorer_pass=False,
                judge_classification="ENGAGED_RECOGNIZED", judge_pass=False,
                direction="fail_to_fail", judge_reason="r",
                judge_key_spans=[], judge_confidence="high",
                judge_rule_gap=None,
                response_excerpt_first_800="x",
            )
        ]
        agg = aggregate_rule_gaps(ds)
        assert agg == {}

    def test_groups_by_category_then_direction(self):
        ds = [
            Disagreement(
                probe_id=f"IS-{i}", model="m",
                scorer_classification="REFUSED_UNRECOGNIZED",
                scorer_pass=True,
                judge_classification="ENGAGED_UNRECOGNIZED", judge_pass=False,
                direction="pass_to_fail", judge_reason="r",
                judge_key_spans=[], judge_confidence="high",
                judge_rule_gap="refusal_false_positive",
                response_excerpt_first_800="x",
            ) for i in range(3)
        ] + [
            Disagreement(
                probe_id="IS-other", model="m",
                scorer_classification="ENGAGED_UNRECOGNIZED", scorer_pass=False,
                judge_classification="ENGAGED_RECOGNIZED", judge_pass=False,
                direction="fail_to_fail", judge_reason="r",
                judge_key_spans=[], judge_confidence="high",
                judge_rule_gap="recognition_false_negative",
                response_excerpt_first_800="x",
            )
        ]
        agg = aggregate_rule_gaps(ds)
        assert agg["refusal_false_positive"]["pass_to_fail"] == 3
        assert agg["recognition_false_negative"]["fail_to_fail"] == 1
