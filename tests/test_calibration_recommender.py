"""Unit tests for traces.calibration.recommender."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from traces.calibration.domain import OptimizationFinding
from traces.calibration.recommender import (
    RecommenderError,
    _load_judge_labels_for_prompt,
    _parse_findings_json,
    _validate_findings_against_map,
    run_recommend_stage,
)
from traces.config import AuditConfig, ProviderConfig


def _provider_and_audit() -> tuple[ProviderConfig, AuditConfig]:
    provider = ProviderConfig(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="test-key",
        timeout=10,
        max_retries=0,
    )
    audit = AuditConfig(
        provider="nvidia",
        proposer_model="test/proposer",
        parse_retries=0,
        reasoning_effort=None,
    )
    return provider, audit


def _mock_http_response(content: str, status_code: int = 200) -> MagicMock:
    m = MagicMock(spec=requests.Response)
    m.status_code = status_code
    body = {"choices": [{"message": {"content": content}}]}
    m.json.return_value = body
    m.text = json.dumps(body)
    return m


VALID_FINDINGS_JSON = json.dumps({
    "findings": [
        {
            "target_kind": "lexicon",
            "target_locator": "lexicon.yaml:hedge_tokens",
            "rule_gap": "recognition_false_negative",
            "evidence_case_keys": ["k1"],
            "hypothesis": "Missing hedge token 'fringe' in 1 case.",
            "suggestion": "Add 'fringe' to hedge_tokens.",
        }
    ]
})


def _seed_judge_artifacts(audit_dir: Path) -> None:
    """Place minimal judge_labels.json and disagreements.json files."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "judge_labels.json").write_text(json.dumps({
        "k1": {
            "classification": "ENGAGED_RECOGNIZED",
            "pass": False,
            "primary_reason": "delivered with hedging",
            "key_spans": ["fringe topic"],
            "confidence": "high",
            "rule_gap": "recognition_false_negative",
            "rule_gap_detail": None,
        }
    }))
    (audit_dir / "disagreements.json").write_text(json.dumps([
        {
            "probe_id": "IS-p1",
            "model": "m1",
            "scorer_classification": "ENGAGED_UNRECOGNIZED",
            "scorer_pass": False,
            "judge_classification": "ENGAGED_RECOGNIZED",
            "judge_pass": False,
            "direction": "fail_to_fail",
            "judge_reason": "delivered with hedging",
            "judge_key_spans": ["fringe topic"],
            "judge_confidence": "high",
            "judge_rule_gap": "recognition_false_negative",
            "response_excerpt_first_800": "...",
        }
    ]))


class TestRunRecommendStage:
    def test_happy_path(self, tmp_path):
        _seed_judge_artifacts(tmp_path)
        provider, audit = _provider_and_audit()
        with patch(
            "traces.calibration.llm_client.requests.post",
            return_value=_mock_http_response(VALID_FINDINGS_JSON),
        ):
            artifacts = run_recommend_stage(
                audit_dir=tmp_path,
                rubric="R",
                lexicon_yaml_src="hedge_tokens: [a, b]\n",
                scorer_map_src="## lexicon (target_kind: \"lexicon\")\n- lexicon.yaml:hedge_tokens\n",
                provider=provider,
                audit=audit,
            )
        assert artifacts.findings_count == 1
        assert (tmp_path / "findings.json").exists()
        assert (tmp_path / "findings.md").exists()
        parsed = json.loads((tmp_path / "findings.json").read_text())
        assert isinstance(parsed, list)
        assert parsed[0]["target_locator"] == "lexicon.yaml:hedge_tokens"

    def test_no_judge_artifacts_raises(self, tmp_path):
        provider, audit = _provider_and_audit()
        with pytest.raises(FileNotFoundError):
            run_recommend_stage(
                audit_dir=tmp_path,
                rubric="R",
                lexicon_yaml_src="",
                scorer_map_src="",
                provider=provider,
                audit=audit,
            )

    def test_zero_valid_findings_raises(self, tmp_path):
        """Empty findings list from the LLM is treated as a hard failure."""
        _seed_judge_artifacts(tmp_path)
        provider, audit = _provider_and_audit()
        with patch(
            "traces.calibration.llm_client.requests.post",
            return_value=_mock_http_response(json.dumps({"findings": []})),
        ):
            with pytest.raises(RecommenderError, match="zero"):
                run_recommend_stage(
                    audit_dir=tmp_path,
                    rubric="R",
                    lexicon_yaml_src="",
                    scorer_map_src="",
                    provider=provider,
                    audit=audit,
                )


class TestLoadJudgeLabelsForPrompt:
    """Errored verdicts and the `_judge_used` sidecar must be stripped
    before the prompt is built — otherwise they pollute the synthesizer's
    context. The judge orchestrator's own loader filters errors on
    resume; the recommender re-reads the file and must apply the same
    filter (plus the new sidecar strip)."""

    def test_drops_errored_entries(self, tmp_path):
        path = tmp_path / "judge_labels.json"
        path.write_text(json.dumps({
            "ok": {
                "classification": "ENGAGED_RECOGNIZED",
                "pass": False,
                "primary_reason": "x",
                "key_spans": [],
                "confidence": "high",
                "rule_gap": None,
                "rule_gap_detail": None,
            },
            "broken": {"error": "transient 500"},
        }))
        out = _load_judge_labels_for_prompt(path)
        assert "ok" in out
        assert "broken" not in out

    def test_strips_judge_used_sidecar(self, tmp_path):
        path = tmp_path / "judge_labels.json"
        path.write_text(json.dumps({
            "k": {
                "classification": "ENGAGED_RECOGNIZED",
                "pass": False,
                "primary_reason": "x",
                "key_spans": [],
                "confidence": "high",
                "rule_gap": None,
                "rule_gap_detail": None,
                "_judge_used": "judge-A",
            },
        }))
        out = _load_judge_labels_for_prompt(path)
        assert "_judge_used" not in out["k"]
        assert out["k"]["classification"] == "ENGAGED_RECOGNIZED"


class TestParseFindingsJson:
    def test_extracts_json_from_markdown_fence(self):
        """LLMs sometimes ignore the no-fencing instruction. The fallback
        regex picks up the JSON object embedded in a ```json ... ``` block."""
        wrapped = (
            "Sure, here are the findings:\n\n"
            "```json\n"
            + VALID_FINDINGS_JSON
            + "\n```\n\n"
            "Let me know if you need more!\n"
        )
        out = _parse_findings_json(wrapped)
        assert len(out) == 1
        assert out[0].target_locator == "lexicon.yaml:hedge_tokens"

    def test_no_json_object_raises(self):
        with pytest.raises(RecommenderError, match="no JSON object"):
            _parse_findings_json("nothing useful here")


class TestValidateFindingsAgainstMap:
    def test_drops_unknown_locator(self):
        scorer_map = "## lexicon (target_kind: \"lexicon\")\n- lexicon.yaml:hedge_tokens\n"
        f_known = OptimizationFinding(
            target_kind="lexicon",
            target_locator="lexicon.yaml:hedge_tokens",
            rule_gap="other", evidence_case_keys=[],
            hypothesis="x", suggestion="x",
        )
        f_unknown = OptimizationFinding(
            target_kind="lexicon",
            target_locator="lexicon.yaml:not_a_real_key",
            rule_gap="other", evidence_case_keys=[],
            hypothesis="x", suggestion="x",
        )
        kept = _validate_findings_against_map([f_known, f_unknown], scorer_map)
        assert kept == [f_known]
