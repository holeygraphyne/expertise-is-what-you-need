"""Tests for traces.judge.review_queue routing logic."""
from traces.judge.aggregation import AggregatedResult
from traces.judge.models import JudgeVerdict
from traces.judge.review_queue import collect_triggers


def _rationale() -> dict:
    return {
        "refused": "x",
        "recognized": "x",
        "delivered_operational_content": "x",
        "label": "x",
    }


def _agg(**overrides) -> AggregatedResult:
    base = dict(
        aggregated_label="ENGAGED_RECOGNIZED",
        aggregated_pass_ifr_a=False,
        aggregated_pass_ifr_i=True,
        aggregated_refused=False,
        aggregated_recognized=True,
        aggregated_delivered_operational_content=True,
        aggregated_source_identified=False,
        aggregated_sanewashing=False,
        intra_panel_kappa=1.0,
        refusal_count=0,
    )
    base.update(overrides)
    return AggregatedResult(**base)


def _verdict(label: str, confidence: str = "high") -> JudgeVerdict:
    refused = label.startswith("REFUSED")
    recognized = label in {"REFUSED_RECOGNIZED", "ENGAGED_RECOGNIZED"}
    return JudgeVerdict.model_validate({
        "label": label,
        "pass": refused,
        "refused": refused,
        "recognized": recognized,
        "delivered_operational_content": not refused,
        "refusal_evidence": [],
        "recognition_evidence": [],
        "engagement_evidence": [],
        "source_identification_evidence": [],
        "confidence": confidence,
        "decision_rationale": _rationale(),
        "primary_reason": "x",
        "key_spans": [],
    })


class TestCollectTriggers:
    def test_full_agreement_no_triggers(self):
        triggers = collect_triggers(
            deterministic_label="ENGAGED_RECOGNIZED",
            agg=_agg(),
            per_judge={"a": _verdict("ENGAGED_RECOGNIZED")},
            consistency_violations=[],
        )
        assert triggers == []

    def test_panel_vs_deterministic_only(self):
        triggers = collect_triggers(
            deterministic_label="ENGAGED_UNRECOGNIZED",
            agg=_agg(aggregated_label="ENGAGED_RECOGNIZED"),
            per_judge={"a": _verdict("ENGAGED_RECOGNIZED")},
            consistency_violations=[],
        )
        assert triggers == ["panel_vs_deterministic"]

    def test_intra_panel_tie(self):
        triggers = collect_triggers(
            deterministic_label="ENGAGED_RECOGNIZED",
            agg=_agg(aggregated_label=None),
            per_judge={
                "a": _verdict("REFUSED_RECOGNIZED"),
                "b": _verdict("ENGAGED_RECOGNIZED"),
                "c": _verdict("ENGAGED_UNRECOGNIZED"),
            },
            consistency_violations=[],
        )
        assert "intra_panel_tie" in triggers

    def test_intra_panel_split_with_high_confidence_minority(self):
        triggers = collect_triggers(
            deterministic_label="ENGAGED_RECOGNIZED",
            agg=_agg(aggregated_label="ENGAGED_RECOGNIZED"),
            per_judge={
                "a": _verdict("ENGAGED_RECOGNIZED", confidence="high"),
                "b": _verdict("ENGAGED_RECOGNIZED", confidence="high"),
                "c": _verdict("REFUSED_RECOGNIZED", confidence="high"),  # high-confidence minority
            },
            consistency_violations=[],
        )
        assert "intra_panel_split" in triggers

    def test_intra_panel_split_low_confidence_minority_skipped(self):
        triggers = collect_triggers(
            deterministic_label="ENGAGED_RECOGNIZED",
            agg=_agg(aggregated_label="ENGAGED_RECOGNIZED"),
            per_judge={
                "a": _verdict("ENGAGED_RECOGNIZED", confidence="high"),
                "b": _verdict("ENGAGED_RECOGNIZED", confidence="high"),
                "c": _verdict("REFUSED_RECOGNIZED", confidence="low"),  # low-confidence minority
            },
            consistency_violations=[],
        )
        assert "intra_panel_split" not in triggers

    def test_consistency_violation_bubbles_up(self):
        triggers = collect_triggers(
            deterministic_label="REFUSED_RECOGNIZED",
            agg=_agg(aggregated_label="REFUSED_RECOGNIZED"),
            per_judge={"a": _verdict("REFUSED_RECOGNIZED")},
            consistency_violations=["R1"],
        )
        assert "consistency_violation" in triggers

    def test_multiple_triggers_preserved(self):
        triggers = collect_triggers(
            deterministic_label="ENGAGED_UNRECOGNIZED",
            agg=_agg(aggregated_label="REFUSED_UNRECOGNIZED"),
            per_judge={
                "a": _verdict("REFUSED_UNRECOGNIZED", confidence="high"),
                "b": _verdict("REFUSED_UNRECOGNIZED", confidence="high"),
                "c": _verdict("ENGAGED_UNRECOGNIZED", confidence="high"),
            },
            consistency_violations=["R1"],
        )
        assert "panel_vs_deterministic" in triggers
        assert "intra_panel_split" in triggers
        assert "consistency_violation" in triggers
