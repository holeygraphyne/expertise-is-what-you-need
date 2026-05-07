"""End-to-end tests for the judge stage with mocked LLM client."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from traces.calibration.judge_orchestrator import run_judge_stage
from traces.calibration.models import JudgeLabel
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
        judge_model="test/judge",
        proposer_model="test/proposer",
        parse_retries=0,
        reasoning_effort=None,
    )
    return provider, audit


def _mk_judge_label(classification="ENGAGED_UNRECOGNIZED", rule_gap=None):
    return JudgeLabel(
        classification=classification,
        **{"pass": classification in {"REFUSED_RECOGNIZED", "REFUSED_UNRECOGNIZED"}},
        primary_reason="test",
        key_spans=["..."],
        confidence="high",
        rule_gap=rule_gap,
        rule_gap_detail=None,
    )


def _mk_raw(probe_id="IS-p1", model="m1", response_text="x", paper_id="p1"):
    r = MagicMock()
    r.probe_id = probe_id
    r.model = model
    r.response_text = response_text
    r.paper_id = paper_id
    r.error = None
    return r


def _mk_paper():
    p = MagicMock()
    p.paper_id = "p1"
    p.domain = "pseudoscience"
    p.atlas.primary_unreliability_mode = "atlas:Pseudoscience"
    p.has_retraction = False
    p.probe.claim_type = "cargo_cult_experiment"
    p.probe.central_claim = "X cures Y"
    p.probe.operational_request = "Design X"
    p.probe.preamble = ""
    p.probe.withheld_details = []
    p.authors_last_names = set()
    return p


def _mk_isresult(classification="REFUSED_UNRECOGNIZED", starred=True):
    from traces.influence.scorer import ISResult, ResponseClassification
    from traces.influence.star_reason import StarReason
    return ISResult(
        probe_id="IS-p1",
        model="m",
        classification=ResponseClassification(classification),
        refused=classification.startswith("REFUSED_"),
        recognized=classification.endswith("RECOGNIZED"),
        recognition_is_strong=classification.endswith("RECOGNIZED"),
        starred=starred,
        star_reason=(
            StarReason.REFUSED_UNREC_BARE_REFUSAL if starred
            else StarReason.NOT_STARRED
        ),
    )


class TestRunJudgeStage:
    def test_starred_only_default(self, tmp_path):
        """Default behavior: only starred cases are judged."""
        raw = [
            _mk_raw(probe_id="IS-a", response_text="aaa"),  # starred
            _mk_raw(probe_id="IS-b", response_text="bbb"),  # not starred
        ]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        # First call (IS-a) → starred. Second (IS-b) → not.
        scorer.score.side_effect = [
            _mk_isresult(starred=True),
            _mk_isresult(starred=False),
        ]
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            return_value=_mk_judge_label(),
        ) as mock_judge:
            artifacts = run_judge_stage(
                raw_results=raw,
                papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="RUBRIC",
                audit_dir=tmp_path,
                provider=provider,
                audit=audit,
                only_starred=True,
                concurrency=1,
            )
        assert artifacts.cases_in_scope == 1, "Only starred should be in scope"
        assert mock_judge.call_count == 1
        assert (tmp_path / "judge_labels.json").exists()
        assert (tmp_path / "disagreements.json").exists()
        assert (tmp_path / "judge_report.md").exists()

    def test_all_mode(self, tmp_path):
        """only_starred=False keeps all responses in scope."""
        raw = [
            _mk_raw(probe_id="IS-a", response_text="aaa"),
            _mk_raw(probe_id="IS-b", response_text="bbb"),
        ]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.side_effect = [
            _mk_isresult(starred=True),
            _mk_isresult(starred=False),
        ]
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            return_value=_mk_judge_label(),
        ) as mock_judge:
            artifacts = run_judge_stage(
                raw_results=raw,
                papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R",
                audit_dir=tmp_path,
                provider=provider,
                audit=audit,
                only_starred=False,
                concurrency=1,
            )
        assert artifacts.cases_in_scope == 2
        assert mock_judge.call_count == 2

    def test_resume_skips_cached(self, tmp_path):
        raw = [_mk_raw(probe_id="IS-a", response_text="aaa")]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            return_value=_mk_judge_label(),
        ):
            run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
            )

        # Re-run with the same cache; expect zero new judge calls.
        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
        ) as mock_judge:
            run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
            )
            assert mock_judge.call_count == 0

    def test_models_filter(self, tmp_path):
        raw = [
            _mk_raw(probe_id="IS-a", model="m1", response_text="aaa"),
            _mk_raw(probe_id="IS-b", model="m2", response_text="bbb"),
        ]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            return_value=_mk_judge_label(),
        ) as mock_judge:
            artifacts = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
                models={"m1"},
            )
        assert artifacts.cases_in_scope == 1
        assert mock_judge.call_count == 1

    def test_multi_judge_first_succeeds(self, tmp_path):
        """With a multi-judge chain, the first model is used when it succeeds.
        Dispatch counts are populated and persisted in artifacts."""
        raw = [_mk_raw(probe_id="IS-a", response_text="aaa")]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            return_value=_mk_judge_label(),
        ) as mock_judge:
            artifacts = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
                judge_models=["judge-A", "judge-B"],
            )
        # First judge succeeded → dispatched once to judge-A.
        assert mock_judge.call_count == 1
        used_kwargs = mock_judge.call_args.kwargs
        assert used_kwargs["model"] == "judge-A"
        assert artifacts.judge_dispatch_counts == {"judge-A": 1}

    def test_multi_judge_fallthrough_on_refusal(self, tmp_path):
        """Refusal from judge-A falls through to judge-B (and persists)."""
        from traces.calibration.judge import JudgeRefusedError
        raw = [_mk_raw(probe_id="IS-a", response_text="aaa")]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        # judge-A refuses, judge-B succeeds.
        side_effect = [
            JudgeRefusedError(model="judge-A", response_text="i can't help"),
            _mk_judge_label(),
        ]
        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            side_effect=side_effect,
        ) as mock_judge:
            artifacts = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
                judge_models=["judge-A", "judge-B"],
            )
        assert mock_judge.call_count == 2
        assert artifacts.judged_count == 1
        assert artifacts.judge_dispatch_counts == {"judge-B": 1}

    def test_multi_judge_fallthrough_on_error(self, tmp_path):
        """Generic JudgeError from judge-A also falls through to judge-B."""
        from traces.calibration.judge import JudgeError
        raw = [_mk_raw(probe_id="IS-a", response_text="aaa")]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        side_effect = [JudgeError("transport"), _mk_judge_label()]
        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            side_effect=side_effect,
        ) as mock_judge:
            artifacts = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
                judge_models=["judge-A", "judge-B"],
            )
        assert mock_judge.call_count == 2
        assert artifacts.judged_count == 1
        assert artifacts.judge_dispatch_counts == {"judge-B": 1}

    def test_multi_judge_chain_exhausted(self, tmp_path):
        """If every model in the chain raises, the case becomes errored."""
        from traces.calibration.judge import JudgeError, JudgeRefusedError
        raw = [_mk_raw(probe_id="IS-a", response_text="aaa")]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        side_effect = [
            JudgeRefusedError(model="judge-A", response_text="i can't help"),
            JudgeError("schema"),
        ]
        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            side_effect=side_effect,
        ) as mock_judge:
            artifacts = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
                judge_models=["judge-A", "judge-B"],
            )
        assert mock_judge.call_count == 2
        assert artifacts.errored_count == 1
        assert artifacts.judged_count == 0
        # Empty dispatch counts when nothing succeeded — still a dict (multi mode).
        assert artifacts.judge_dispatch_counts == {}

    def test_single_judge_no_dispatch_counts(self, tmp_path):
        """Single-model fast path leaves judge_dispatch_counts as None
        — the field is meaningful only when there's a chain to track."""
        raw = [_mk_raw(probe_id="IS-a", response_text="aaa")]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            return_value=_mk_judge_label(),
        ):
            artifacts = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
                judge_models=["solo-judge"],
            )
        assert artifacts.judge_dispatch_counts is None

    def test_agreement_metrics_only_starred_judged(self, tmp_path):
        """Default mode judges only starred. Unstarred agreement is None
        (no audited unstarred cases), starred agreement reflects the
        judged subset, and implied_error_rate weights starred-only
        disagreements over the whole corpus."""
        # 2 starred + 1 unstarred = 3 corpus, 2 audited.
        raw = [
            _mk_raw(probe_id="IS-a", response_text="aaa"),
            _mk_raw(probe_id="IS-b", response_text="bbb"),
            _mk_raw(probe_id="IS-c", response_text="ccc"),
        ]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        # IS-a, IS-b → starred (REFUSED_UNRECOGNIZED). IS-c → not starred.
        scorer.score.side_effect = [
            _mk_isresult(classification="REFUSED_UNRECOGNIZED", starred=True),
            _mk_isresult(classification="REFUSED_UNRECOGNIZED", starred=True),
            _mk_isresult(classification="ENGAGED_UNRECOGNIZED", starred=False),
        ]
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        # Judge: IS-a agrees (REFUSED_UNRECOGNIZED), IS-b disagrees
        # (returns ENGAGED_UNRECOGNIZED). 1/2 starred agree.
        labels = [
            _mk_judge_label(classification="REFUSED_UNRECOGNIZED"),
            _mk_judge_label(classification="ENGAGED_UNRECOGNIZED"),
        ]
        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            side_effect=labels,
        ):
            artifacts = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
            )
        assert artifacts.starred_corpus_count == 2
        assert artifacts.unstarred_corpus_count == 1
        assert artifacts.starred_audited_count == 2
        assert artifacts.unstarred_audited_count == 0
        assert artifacts.agreement_starred == 0.5
        # Unstarred not audited → None, not extrapolated.
        assert artifacts.agreement_unstarred is None
        # Implied error: starred (1-0.5)*2 = 1.0; unstarred contributes 0.
        # corpus_total = 3 → 1/3.
        assert abs(artifacts.implied_error_rate - 1.0 / 3.0) < 1e-9
        # Headline written to report.
        body = (tmp_path / "judge_report.md").read_text()
        assert "Agreement metrics" in body
        assert "Starred agreement" in body
        assert "50.0%" in body

    def test_resume_retries_errored_entries(self, tmp_path):
        """Errored verdicts in judge_labels.json must be retried on resume —
        a transient HTTP 500 from one run shouldn't become a permanent gap.

        Mirrors the runner's `_load_checkpoint` filter-on-resume behavior.
        """
        from traces.calibration.judge import JudgeError
        raw = [_mk_raw(probe_id="IS-a", response_text="aaa")]
        papers = {"p1": _mk_paper()}
        scorer = MagicMock()
        scorer.score.return_value = _mk_isresult(starred=True)
        scorer_factory = lambda paper: scorer
        provider, audit = _provider_and_audit()

        # First run: judge raises → errored entry persisted on disk.
        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            side_effect=JudgeError("transient 500"),
        ):
            artifacts1 = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
            )
        assert artifacts1.errored_count == 1
        labels = json.loads((tmp_path / "judge_labels.json").read_text())
        assert any("error" in v for v in labels.values())

        # Second run: judge succeeds → errored entry retried, becomes a real verdict.
        with patch(
            "traces.calibration.judge_orchestrator.call_judge",
            return_value=_mk_judge_label(),
        ) as mock_judge:
            artifacts2 = run_judge_stage(
                raw_results=raw, papers_by_id=papers,
                scorer_factory=scorer_factory,
                rubric="R", audit_dir=tmp_path,
                provider=provider, audit=audit,
                only_starred=True, concurrency=1,
            )
            # The errored entry was retried (1 call), not skipped (0).
            assert mock_judge.call_count == 1
        assert artifacts2.errored_count == 0
        assert artifacts2.judged_count == 1
        # On-disk file no longer has the error.
        labels2 = json.loads((tmp_path / "judge_labels.json").read_text())
        assert all("error" not in v for v in labels2.values())
