"""Tests for traces.judge.reporting fragment functions."""
import pytest

from traces.judge.reporting import (
    agreement_matrix,
    cohen_kappa_4class,
    per_judge_kappa,
)


class TestAgreementMatrix:
    def test_perfect_agreement_diagonal(self):
        rows = [("ENGAGED_RECOGNIZED", "ENGAGED_RECOGNIZED")] * 5
        m = agreement_matrix(rows)
        assert m["ENGAGED_RECOGNIZED"]["ENGAGED_RECOGNIZED"] == 5
        # Off-diagonal cells default-zero.
        assert m["ENGAGED_RECOGNIZED"].get("ENGAGED_UNRECOGNIZED", 0) == 0

    def test_disagreement_off_diagonal(self):
        rows = [
            ("ENGAGED_UNRECOGNIZED", "REFUSED_RECOGNIZED"),
            ("ENGAGED_UNRECOGNIZED", "REFUSED_RECOGNIZED"),
            ("REFUSED_RECOGNIZED", "REFUSED_RECOGNIZED"),
        ]
        m = agreement_matrix(rows)
        assert m["ENGAGED_UNRECOGNIZED"]["REFUSED_RECOGNIZED"] == 2
        assert m["REFUSED_RECOGNIZED"]["REFUSED_RECOGNIZED"] == 1


class TestCohenKappa4Class:
    def test_perfect_agreement_one(self):
        rows = [("A", "A"), ("B", "B"), ("C", "C")]
        assert cohen_kappa_4class(rows) == pytest.approx(1.0)

    def test_zero_agreement_below_chance(self):
        # With perfect rater confusion (always disagree), κ < 0.
        rows = [("A", "B"), ("B", "A")] * 5
        assert cohen_kappa_4class(rows) < 0

    def test_too_few_returns_none(self):
        assert cohen_kappa_4class([]) is None
        assert cohen_kappa_4class([("A", "A")]) is None


class TestPerJudgeKappa:
    def test_per_judge_returns_dict(self):
        # Each row: deterministic_label, dict of {member_id: judge_label}.
        rows = [
            ("REFUSED_RECOGNIZED", {"a": "REFUSED_RECOGNIZED", "b": "REFUSED_UNRECOGNIZED"}),
            ("ENGAGED_UNRECOGNIZED", {"a": "ENGAGED_UNRECOGNIZED", "b": "REFUSED_RECOGNIZED"}),
        ]
        kappa = per_judge_kappa(rows)
        assert "a" in kappa
        assert "b" in kappa
        assert kappa["a"] >= kappa["b"]  # member 'a' agreed twice


# ---------------------------------------------------------------------------
# Task 5.2: enrich_report_with_judge_data
# ---------------------------------------------------------------------------
import json
from pathlib import Path

from traces.judge.reporting import enrich_report_with_judge_data


class TestReportEnrichment:
    def test_no_artifact_returns_empty(self, tmp_path):
        result = enrich_report_with_judge_data(judge_dir=tmp_path)
        assert result == {}

    def test_artifact_present_returns_summary(self, tmp_path):
        # Minimal verdicts file shape — one row, panel agreed with deterministic.
        verdicts = {
            "IS-x|m|abc|v": {
                "panel_verdict": {
                    "probe_id": "IS-x", "model": "m", "response_sha256": "abc",
                    "per_judge": {},
                    "aggregated_label": "ENGAGED_UNRECOGNIZED",
                    "aggregated_pass_ifr_a": False,
                    "aggregated_pass_ifr_i": False,
                    "aggregated_refused": False,
                    "aggregated_recognized": False,
                    "aggregated_delivered_operational_content": True,
                    "aggregated_source_identified": False,
                    "aggregated_sanewashing": False,
                    "intra_panel_kappa": 1.0,
                    "refusal_count": 0,
                    "judge_version": "v",
                },
                "_judge_version": "v",
            }
        }
        (tmp_path).mkdir(parents=True, exist_ok=True)
        (tmp_path / "judge_verdicts.json").write_text(json.dumps(verdicts))
        result = enrich_report_with_judge_data(judge_dir=tmp_path)
        assert result["panel_verdict_count"] == 1
        assert result["sanewashing_count"] == 0
        assert result["source_identified_count"] == 0

    def test_artifact_summary_filters_stale_judge_versions(self, tmp_path):
        verdicts = {
            f"IS-x|m|abc|{'0' * 64}|old": {
                "panel_verdict": {
                    "probe_id": "IS-x", "model": "m", "response_sha256": "abc",
                    "per_judge": {},
                    "aggregated_label": "ENGAGED_UNRECOGNIZED",
                    "aggregated_pass_ifr_a": False,
                    "aggregated_pass_ifr_i": False,
                    "aggregated_refused": False,
                    "aggregated_recognized": False,
                    "aggregated_delivered_operational_content": True,
                    "aggregated_source_identified": False,
                    "aggregated_sanewashing": False,
                    "intra_panel_kappa": 1.0,
                    "refusal_count": 0,
                    "judge_version": "old",
                },
                "_judge_version": "old",
            },
            f"IS-x|m|abc|{'1' * 64}|new": {
                "panel_verdict": {
                    "probe_id": "IS-x", "model": "m", "response_sha256": "abc",
                    "per_judge": {},
                    "aggregated_label": "REFUSED_RECOGNIZED",
                    "aggregated_pass_ifr_a": True,
                    "aggregated_pass_ifr_i": True,
                    "aggregated_refused": True,
                    "aggregated_recognized": True,
                    "aggregated_delivered_operational_content": False,
                    "aggregated_source_identified": True,
                    "aggregated_sanewashing": False,
                    "intra_panel_kappa": 1.0,
                    "refusal_count": 0,
                    "judge_version": "new",
                },
                "_judge_version": "new",
            },
        }
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "judge_verdicts.json").write_text(json.dumps(verdicts))
        (tmp_path / "judge_run_meta.json").write_text(
            json.dumps({"judge_version": "new"})
        )

        result = enrich_report_with_judge_data(judge_dir=tmp_path)

        assert result["panel_verdict_count"] == 1
        assert result["ifr_judge_label_distribution"] == {
            "REFUSED_RECOGNIZED": 1,
        }
        assert result["source_identified_count"] == 1

    def test_artifact_summary_filters_old_cache_key_shape_when_meta_exists(
        self, tmp_path,
    ):
        verdicts = {
            "IS-x|m|abc|v": {
                "panel_verdict": {
                    "probe_id": "IS-x", "model": "m", "response_sha256": "abc",
                    "per_judge": {},
                    "aggregated_label": "ENGAGED_UNRECOGNIZED",
                    "aggregated_pass_ifr_a": False,
                    "aggregated_pass_ifr_i": False,
                    "aggregated_refused": False,
                    "aggregated_recognized": False,
                    "aggregated_delivered_operational_content": True,
                    "aggregated_source_identified": False,
                    "aggregated_sanewashing": False,
                    "intra_panel_kappa": 1.0,
                    "refusal_count": 0,
                    "judge_version": "v",
                },
                "_judge_version": "v",
            },
            f"IS-x|m|abc|{'1' * 64}|v": {
                "panel_verdict": {
                    "probe_id": "IS-x", "model": "m", "response_sha256": "abc",
                    "per_judge": {},
                    "aggregated_label": "REFUSED_RECOGNIZED",
                    "aggregated_pass_ifr_a": True,
                    "aggregated_pass_ifr_i": True,
                    "aggregated_refused": True,
                    "aggregated_recognized": True,
                    "aggregated_delivered_operational_content": False,
                    "aggregated_source_identified": False,
                    "aggregated_sanewashing": False,
                    "intra_panel_kappa": 1.0,
                    "refusal_count": 0,
                    "judge_version": "v",
                },
                "_judge_version": "v",
            },
        }
        (tmp_path / "judge_verdicts.json").write_text(json.dumps(verdicts))
        (tmp_path / "judge_run_meta.json").write_text(
            json.dumps({"judge_version": "v"})
        )

        result = enrich_report_with_judge_data(judge_dir=tmp_path)

        assert result["panel_verdict_count"] == 1
        assert result["ifr_judge_label_distribution"] == {
            "REFUSED_RECOGNIZED": 1,
        }
