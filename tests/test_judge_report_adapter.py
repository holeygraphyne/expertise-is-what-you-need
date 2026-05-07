"""Tests for the judge report adapter."""
import json

from traces.influence import ISResult, ResponseClassification
from traces.judge.report_adapter import render_judge_panel_section


def _result(probe_id: str, model: str, classification: str) -> ISResult:
    return ISResult(
        probe_id=probe_id,
        model=model,
        classification=ResponseClassification(classification),
        refused=classification.startswith("REFUSED"),
        recognized=classification in {"REFUSED_RECOGNIZED", "ENGAGED_RECOGNIZED"},
        recognition_is_strong=False,
        starred=False,
        lexicon_version="test",
    )


class TestJudgeReportAdapter:
    def test_renders_section_from_current_judge_artifacts(self, tmp_path):
        judge_dir = tmp_path / "judge"
        judge_dir.mkdir()
        judge_dir.joinpath("judge_run_meta.json").write_text(
            json.dumps({"judge_version": "v"})
        )
        judge_dir.joinpath("judge_verdicts.json").write_text(json.dumps({
            f"IS-x|model-a|{'0' * 64}|{'1' * 64}|v": {
                "_judge_version": "v",
                "panel_verdict": {
                    "probe_id": "IS-x",
                    "model": "model-a",
                    "response_sha256": "0" * 64,
                    "per_judge": {
                        "judge-a": {"label": "REFUSED_RECOGNIZED"},
                    },
                    "aggregated_label": "REFUSED_RECOGNIZED",
                    "aggregated_pass_ifr_a": True,
                    "aggregated_pass_ifr_i": True,
                    "aggregated_refused": True,
                    "aggregated_recognized": True,
                    "aggregated_delivered_operational_content": False,
                    "aggregated_source_identified": False,
                    "aggregated_sanewashing": False,
                    "evidence_validation_issues": [],
                    "intra_panel_kappa": 1.0,
                    "refusal_count": 0,
                    "judge_version": "v",
                },
            },
        }))

        section = render_judge_panel_section(
            judge_dir=judge_dir,
            results_by_model={
                "model-a": [_result("IS-x", "model-a", "ENGAGED_UNRECOGNIZED")],
            },
        )

        assert section is not None
        assert "## LLM Judge Panel" in section
        assert "IFR-judge-a" in section
        assert "Cohen" in section

    def test_missing_artifacts_return_none(self, tmp_path):
        assert render_judge_panel_section(
            judge_dir=tmp_path,
            results_by_model={},
        ) is None
