# tests/test_judge_orchestrator.py
"""Integration test for the orchestrator with a faked panel adapter."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers import make_paper
from traces.config import AuditConfig, JudgePanelMember, ProviderConfig
from traces.judge.models import JudgeVerdict
from traces.judge.orchestrator import run_score_judge_stage


def _rationale() -> dict:
    return {
        "refused": "x",
        "recognized": "x",
        "delivered_operational_content": "x",
        "label": "x",
    }


def _verdict(label: str, **overrides) -> JudgeVerdict:
    refused = label.startswith("REFUSED")
    recognized = label in {"REFUSED_RECOGNIZED", "ENGAGED_RECOGNIZED"}
    base = {
        "label": label,
        "pass": refused,
        "refused": refused,
        "recognized": recognized,
        "delivered_operational_content": not refused,
        "refusal_evidence": [],
        "recognition_evidence": [],
        "engagement_evidence": [],
        "source_identification_evidence": [],
        "confidence": "high",
        "decision_rationale": _rationale(),
        "primary_reason": "x",
        "key_spans": [],
    }
    base.update(overrides)
    return JudgeVerdict.model_validate(base)


@pytest.fixture
def papers_by_id():
    return {"bem_psi_2011": make_paper("bem_psi_2011", domain="pseudoscience")}


@pytest.fixture
def panel():
    return [
        JudgePanelMember(provider="anthropic", model="claude-opus-4-7"),
        JudgePanelMember(provider="openai", model="gpt-5"),
        JudgePanelMember(provider="nvidia", model="deepseek-ai/deepseek-v4-pro"),
    ]


@pytest.fixture
def providers():
    pc = ProviderConfig(base_url="https://api.example.com/v1", api_key="x", timeout=10, max_retries=0)
    return {"anthropic": pc, "openai": pc, "nvidia": pc}


@pytest.fixture
def raw_results_path(tmp_path):
    src = Path(__file__).parent / "fixtures" / "judge" / "raw_results_small.json"
    dst = tmp_path / "raw_results.json"
    dst.write_text(src.read_text())
    return dst


class TestRunScoreJudgeStage:
    def test_two_response_run_writes_verdicts_and_review_queue(
        self, tmp_path, raw_results_path, papers_by_id, panel, providers,
    ):
        # Mock the per-judge call so the orchestrator does no real HTTP.
        # Different responses get different labels to exercise both
        # cases (response 1: refused-recognized; response 2: engaged-unrec).
        def fake_call_panel_judge(*, payload, rubric, provider, audit, model):
            if "I cannot help" in payload:
                return _verdict("REFUSED_RECOGNIZED")
            return _verdict("ENGAGED_UNRECOGNIZED")

        from traces.atlas import VocabularyLoader
        from traces.atlas.ontology_loader import ATLASGraph
        from traces.config import AtlasConfig
        # Use a fake scorer factory so we don't depend on real corpus loading.
        from traces.judge.aggregation import AggregatedResult  # noqa
        # The orchestrator needs a scorer_factory(paper) -> ISScorer-like
        # interface that exposes .score(...) returning an object with
        # .classification.value attribute.
        class FakeIS:
            def __init__(self, label):
                self.classification = type("X", (), {"value": label})()
        class FakeScorer:
            def score(self, *, response_text, probe_id, model, **kwargs):
                if "I cannot help" in response_text:
                    return FakeIS("REFUSED_RECOGNIZED")
                return FakeIS("ENGAGED_UNRECOGNIZED")
        def fake_scorer_factory(paper):
            return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={
                "anthropic/claude-opus-4-7": 0.04,
                "openai/gpt-5": 0.04,
                "nvidia/deepseek-ai/deepseek-v4-pro": 0.04,
            },
            harm_aggregation="max",
        )
        rubric = "rubric text"

        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric=rubric,
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,  # disable budget gate for the test
            )

        verdicts_path = tmp_path / "judge" / "judge_verdicts.json"
        review_queue_path = tmp_path / "judge" / "review_queue.jsonl"
        meta_path = tmp_path / "judge" / "judge_run_meta.json"

        assert verdicts_path.exists()
        verdicts = json.loads(verdicts_path.read_text())
        assert len(verdicts) == 2

        # Deterministic scorer and panel agree on both rows, so we do not
        # expect review-queue entries in this fixture.
        if review_queue_path.exists():
            lines = [
                line for line in review_queue_path.read_text().splitlines()
                if line.strip()
            ]
            assert artifacts.review_queue_count == len(lines)

        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "judge_version" in meta
        assert "panel_member_ids" in meta

    def test_resume_from_cache_skips_judged_rows(
        self, tmp_path, raw_results_path, papers_by_id, panel, providers,
    ):
        # Pre-populate the cache with one verdict.
        # ... (test that a second invocation does not re-call the panel
        # for cached rows). Implementation: count call_panel_judge invocations.
        call_counter = {"n": 0}

        def fake_call(*, payload, rubric, provider, audit, model):
            call_counter["n"] += 1
            return _verdict("ENGAGED_UNRECOGNIZED")

        class FakeIS:
            def __init__(self): self.classification = type("X", (), {"value": "ENGAGED_UNRECOGNIZED"})()
        class FakeScorer:
            def score(self, **kw): return FakeIS()
        def fake_scorer_factory(paper): return FakeScorer()

        audit = AuditConfig(
            provider="nvidia", judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.04 for m in panel},
            harm_aggregation="max",
        )
        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call(**kw),
        ):
            run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="r",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )
            first_run_calls = call_counter["n"]

            # Second invocation should be a full cache hit.
            run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="r",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )
            assert call_counter["n"] == first_run_calls

    def test_stale_judge_versions_are_purged_on_rerun(
        self, tmp_path, raw_results_path, papers_by_id, panel, providers,
    ):
        judge_dir = tmp_path / "judge"
        judge_dir.mkdir()
        (judge_dir / "judge_verdicts.json").write_text(json.dumps({
            "stale-key": {
                "panel_verdict": {
                    "probe_id": "IS-old",
                    "model": "old-model",
                    "response_sha256": "0" * 64,
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
        }))

        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "ENGAGED_UNRECOGNIZED"})()
        class FakeScorer:
            def score(self, **kw): return FakeIS()
        def fake_scorer_factory(paper): return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.04 for m in panel},
            harm_aggregation="max",
        )
        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            return_value=_verdict("ENGAGED_UNRECOGNIZED"),
        ):
            run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="r",
                judge_dir=judge_dir,
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )

        verdicts = json.loads((judge_dir / "judge_verdicts.json").read_text())
        assert "stale-key" not in verdicts

    def test_payload_metadata_change_invalidates_cache(
        self, tmp_path, raw_results_path, papers_by_id, panel, providers,
    ):
        call_counter = {"n": 0}

        def fake_call(*, payload, rubric, provider, audit, model):
            call_counter["n"] += 1
            return _verdict("ENGAGED_UNRECOGNIZED")

        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "ENGAGED_UNRECOGNIZED"})()
        class FakeScorer:
            def score(self, **kw): return FakeIS()
        def fake_scorer_factory(paper): return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.04 for m in panel},
            harm_aggregation="max",
        )
        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call(**kw),
        ):
            run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="r",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )
            first_run_calls = call_counter["n"]

            papers_by_id["bem_psi_2011"]._domain = "pathological_science"
            run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="r",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )

        assert call_counter["n"] == first_run_calls * 2

    def test_missing_paper_rows_skipped_before_dispatch(
        self, tmp_path, papers_by_id, panel, providers,
    ):
        rows = [
            {
                "probe_id": "IS-bem_psi_2011",
                "paper_id": "bem_psi_2011",
                "model": "test-model-ok",
                "response_text": "A valid row with a study design.",
                "latency_ms": 100.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "stop",
                "timestamp": "2026-04-30T00:00:00Z",
                "error": None,
            },
            {
                "probe_id": "IS-missing",
                "paper_id": "missing-paper",
                "model": "test-model-missing",
                "response_text": "This row should not reach judge dispatch.",
                "latency_ms": 100.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "stop",
                "timestamp": "2026-04-30T00:00:00Z",
                "error": None,
            },
        ]
        raw_results_path = tmp_path / "raw_results.json"
        raw_results_path.write_text(json.dumps(rows))
        call_counter = {"n": 0}

        def fake_call_panel_judge(*, payload, rubric, provider, audit, model):
            call_counter["n"] += 1
            assert "missing-paper" not in payload
            return _verdict("ENGAGED_UNRECOGNIZED")

        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "ENGAGED_UNRECOGNIZED"})()

        class FakeScorer:
            def score(self, **kw):
                return FakeIS()

        def fake_scorer_factory(paper):
            return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.01 for m in panel},
            harm_aggregation="max",
        )
        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )

        assert artifacts.cases_in_scope == 1
        assert artifacts.judged_count == 1
        assert artifacts.errored_count == 0
        assert call_counter["n"] == len(panel)
        meta = json.loads((tmp_path / "judge" / "judge_run_meta.json").read_text())
        assert meta["skipped_missing_paper"] == 1

    def test_starred_only_filters_to_scorer_starred_rows(
        self, tmp_path, raw_results_path, papers_by_id, panel, providers,
    ):
        call_counter = {"n": 0}

        def fake_call_panel_judge(*, payload, rubric, provider, audit, model):
            call_counter["n"] += 1
            assert "IAPS images" not in payload
            return _verdict("REFUSED_RECOGNIZED")

        class FakeIS:
            def __init__(self, label: str, starred: bool):
                self.classification = type("X", (), {"value": label})()
                self.starred = starred

        class FakeScorer:
            def score(self, *, response_text, **kw):
                if "I cannot help" in response_text:
                    return FakeIS("REFUSED_RECOGNIZED", True)
                return FakeIS("ENGAGED_UNRECOGNIZED", False)

        def fake_scorer_factory(paper):
            return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.01 for m in panel},
            harm_aggregation="max",
        )
        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
                starred_only=True,
            )

        assert artifacts.cases_in_scope == 1
        assert artifacts.judged_count == 1
        assert call_counter["n"] == len(panel)
        meta = json.loads((tmp_path / "judge" / "judge_run_meta.json").read_text())
        assert meta["starred_only"] is True
        assert meta["rows_before_starred_filter"] == 2

    def test_all_panel_failures_are_not_cached_as_judged(
        self, tmp_path, raw_results_path, papers_by_id, panel, providers,
    ):
        from traces.calibration.judge import JudgeError

        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "ENGAGED_UNRECOGNIZED"})()

        class FakeScorer:
            def score(self, **kw):
                return FakeIS()

        def fake_scorer_factory(paper):
            return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.01 for m in panel},
            harm_aggregation="max",
        )
        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=JudgeError("transport failed"),
        ):
            artifacts = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )

        assert artifacts.judged_count == 0
        assert artifacts.errored_count == 2
        verdicts_path = tmp_path / "judge" / "judge_verdicts.json"
        assert not verdicts_path.exists()

    def test_concurrent_review_queue_no_partial_lines(
        self, tmp_path, papers_by_id, panel, providers,
    ):
        """Concurrent workers that all trigger the review queue must produce
        well-formed JSONL — no partial/interleaved lines.

        Strategy: create 6 rows that all produce a panel_vs_deterministic
        trigger (panel says ENGAGED_UNRECOGNIZED, deterministic scorer says
        REFUSED_RECOGNIZED), then run with max_workers=4 and verify every
        line in review_queue.jsonl parses as valid JSON.
        """
        rows = [
            {
                "probe_id": f"IS-bem_psi_2011",
                "paper_id": "bem_psi_2011",
                "model": f"test-model-{i}",
                "response_text": f"Concurrent response number {i} with enough text " * 10,
                "latency_ms": 100.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "stop",
                "timestamp": "2026-04-30T00:00:00Z",
                "error": None,
            }
            for i in range(6)
        ]
        raw_results_path = tmp_path / "raw_results.json"
        raw_results_path.write_text(json.dumps(rows))

        # Panel always says ENGAGED_UNRECOGNIZED; scorer always says
        # REFUSED_RECOGNIZED → panel_vs_deterministic fires for every row.
        def fake_call_panel_judge(*, payload, rubric, provider, audit, model):
            # Simulate some work so threads overlap in time.
            time.sleep(0.01)
            return _verdict("ENGAGED_UNRECOGNIZED")

        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "REFUSED_RECOGNIZED"})()

        class FakeScorer:
            def score(self, **kw):
                return FakeIS()

        def fake_scorer_factory(paper):
            return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.01 for m in panel},
            harm_aggregation="max",
        )

        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=4,
                max_cost_usd=0.0,
            )

        review_queue_path = tmp_path / "judge" / "review_queue.jsonl"
        assert review_queue_path.exists(), "review_queue.jsonl must be created"

        raw_text = review_queue_path.read_text()
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]

        # Every line must parse as valid JSON (no partial writes / interleaving).
        parsed = []
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"review_queue.jsonl line {i} is not valid JSON: {exc!r}\n"
                    f"  raw: {line!r}"
                ) from exc
            parsed.append(obj)

        # All 6 rows triggered review queue (panel_vs_deterministic).
        assert len(parsed) == 6, (
            f"Expected 6 review-queue entries, got {len(parsed)}; "
            f"artifacts.review_queue_count={artifacts.review_queue_count}"
        )
        assert artifacts.review_queue_count == 6

    def test_review_queue_preserved_across_resume(
        self, tmp_path, papers_by_id, panel, providers,
    ):
        """Re-running with the same judge_dir (full cache hit) must preserve
        review-queue rows instead of silently truncating to zero.

        Setup: 2 rows where the panel says ENGAGED_UNRECOGNIZED but the
        deterministic scorer says REFUSED_RECOGNIZED → panel_vs_deterministic
        fires for both rows on the first run.  After the second run (full
        cache hit, no new HTTP calls) the queue must still have 2 rows.
        """
        rows = [
            {
                "probe_id": "IS-bem_psi_2011",
                "paper_id": "bem_psi_2011",
                "model": f"test-model-resume-{i}",
                "response_text": f"Resume test response {i} " * 20,
                "latency_ms": 100.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "finish_reason": "stop",
                "timestamp": "2026-04-30T00:00:00Z",
                "error": None,
            }
            for i in range(2)
        ]
        raw_results_path = tmp_path / "raw_results.json"
        raw_results_path.write_text(json.dumps(rows))

        # Panel → ENGAGED_UNRECOGNIZED; deterministic scorer → REFUSED_RECOGNIZED.
        # panel_vs_deterministic fires for every row.
        def fake_call_panel_judge(*, payload, rubric, provider, audit, model):
            return _verdict("ENGAGED_UNRECOGNIZED")

        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "REFUSED_RECOGNIZED"})()

        class FakeScorer:
            def score(self, **kw):
                return FakeIS()

        def fake_scorer_factory(paper):
            return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.01 for m in panel},
            harm_aggregation="max",
        )
        judge_dir = tmp_path / "judge"

        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts1 = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=judge_dir,
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )

        assert artifacts1.review_queue_count == 2, (
            f"First run should produce 2 review-queue rows, got {artifacts1.review_queue_count}"
        )

        # Second run: full cache hit, no new panel calls.
        # The review queue must still have 2 rows (rebuilt from cache).
        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts2 = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=judge_dir,
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
            )

        review_queue_path = judge_dir / "review_queue.jsonl"
        assert review_queue_path.exists()
        lines = [ln for ln in review_queue_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2, (
            f"After resume, review_queue.jsonl must still have 2 rows, got {len(lines)}"
        )
        assert artifacts2.review_queue_count == 2, (
            f"artifacts.review_queue_count should be 2 on resume, got {artifacts2.review_queue_count}"
        )

    def test_evidence_validation_review_row_preserved_across_resume(
        self, tmp_path, papers_by_id, panel, providers,
    ):
        rows = [{
            "probe_id": "IS-bem_psi_2011",
            "paper_id": "bem_psi_2011",
            "model": "test-model-evidence",
            "response_text": "The response gives a study design without naming a source.",
            "latency_ms": 100.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "stop",
            "timestamp": "2026-04-30T00:00:00Z",
            "error": None,
        }]
        raw_results_path = tmp_path / "raw_results.json"
        raw_results_path.write_text(json.dumps(rows))

        def fake_call_panel_judge(*, payload, rubric, provider, audit, model):
            return _verdict(
                "ENGAGED_UNRECOGNIZED",
                source_identification_evidence=[
                    {"kind": "author", "quote": "hallucinated author"},
                ],
            )

        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "ENGAGED_UNRECOGNIZED"})()

        class FakeScorer:
            def score(self, **kw):
                return FakeIS()

        def fake_scorer_factory(paper):
            return FakeScorer()

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.01 for m in panel},
            harm_aggregation="max",
        )
        judge_dir = tmp_path / "judge"

        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts1 = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=judge_dir,
                providers=providers,
                audit=audit,
                concurrency=1,
                max_cost_usd=0.0,
            )
            artifacts2 = run_score_judge_stage(
                raw_results_path=raw_results_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric text",
                judge_dir=judge_dir,
                providers=providers,
                audit=audit,
                concurrency=1,
                max_cost_usd=0.0,
            )

        assert artifacts1.review_queue_count == 1
        assert artifacts2.review_queue_count == 1
        lines = [
            json.loads(line)
            for line in (judge_dir / "review_queue.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert lines[0]["disagreement_kind"] == "consistency_violation"


class TestStratifiedSampling:
    """Tests for the --sample stratification feature."""

    def _make_raw_results_json(self, tmp_path, rows: list[dict]) -> "Path":
        p = tmp_path / "raw_results.json"
        p.write_text(json.dumps(rows))
        return p

    def _row(self, paper_id: str, model: str, idx: int) -> dict:
        return {
            "probe_id": f"IS-{paper_id}",
            "paper_id": paper_id,
            "model": model,
            "response_text": f"Response {paper_id}/{model}/{idx} " * 5,
            "latency_ms": 100.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "stop",
            "timestamp": "2026-04-30T00:00:00Z",
            "error": None,
        }

    def _fake_setup(self, panel):
        """Return a (fake_scorer_factory, fake_call_panel_judge, audit) triple."""
        class FakeIS:
            def __init__(self):
                self.classification = type("X", (), {"value": "ENGAGED_UNRECOGNIZED"})()
        class FakeScorer:
            def score(self, **kw): return FakeIS()
        def fake_scorer_factory(paper): return FakeScorer()

        def fake_call_panel_judge(*, payload, rubric, provider, audit, model):
            return _verdict("ENGAGED_UNRECOGNIZED")

        audit = AuditConfig(
            provider="nvidia",
            judge_panel=panel,
            cost_per_call_usd={m.member_id: 0.01 for m in panel},
            harm_aggregation="max",
        )
        return fake_scorer_factory, fake_call_panel_judge, audit

    def test_sample_takes_n_per_cell(self, tmp_path, panel, providers):
        """3 cells x 5 rows each; sample=2 => 6 rows judged."""
        papers_by_id = {
            "paper_a": make_paper("paper_a", domain="pseudoscience"),
            "paper_b": make_paper("paper_b", domain="pseudoscience"),
        }
        rows = (
            [self._row("paper_a", "model_x", i) for i in range(5)]
            + [self._row("paper_a", "model_y", i) for i in range(5)]
            + [self._row("paper_b", "model_x", i) for i in range(5)]
        )
        raw_path = self._make_raw_results_json(tmp_path, rows)
        fake_scorer_factory, fake_call_panel_judge, audit = self._fake_setup(panel)

        call_counter = {"n": 0}
        def counting_judge(**kw):
            call_counter["n"] += 1
            return fake_call_panel_judge(**kw)

        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: counting_judge(**kw),
        ):
            artifacts = run_score_judge_stage(
                raw_results_path=raw_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=2,
                max_cost_usd=0.0,
                sample=2,
            )

        # 3 cells × 2 samples = 6 rows in scope; each row calls 3 panel members.
        assert artifacts.cases_in_scope == 6
        assert call_counter["n"] == 6 * len(panel)

    def test_sample_handles_undersized_cell(self, tmp_path, panel, providers):
        """Cell with 1 row and sample=5 => that 1 row is taken (no error)."""
        papers_by_id = {
            "paper_a": make_paper("paper_a", domain="pseudoscience"),
        }
        rows = [self._row("paper_a", "model_x", 0)]  # only 1 row
        raw_path = self._make_raw_results_json(tmp_path, rows)
        fake_scorer_factory, fake_call_panel_judge, audit = self._fake_setup(panel)

        with patch(
            "traces.judge.orchestrator.call_panel_judge",
            side_effect=lambda **kw: fake_call_panel_judge(**kw),
        ):
            artifacts = run_score_judge_stage(
                raw_results_path=raw_path,
                papers_by_id=papers_by_id,
                scorer_factory=fake_scorer_factory,
                rubric="rubric",
                judge_dir=tmp_path / "judge",
                providers=providers,
                audit=audit,
                concurrency=1,
                max_cost_usd=0.0,
                sample=5,
            )

        assert artifacts.cases_in_scope == 1

    def test_sample_seed_is_deterministic(self, tmp_path, panel, providers):
        """Same seed => same selection across two fresh invocations."""
        papers_by_id = {
            "paper_a": make_paper("paper_a", domain="pseudoscience"),
        }
        rows = [self._row("paper_a", "model_x", i) for i in range(10)]
        raw_path = self._make_raw_results_json(tmp_path, rows)
        fake_scorer_factory, fake_call_panel_judge, audit = self._fake_setup(panel)

        judged_probe_ids_run = []

        def capturing_judge(*, payload, rubric, provider, audit, model):
            return fake_call_panel_judge(
                payload=payload, rubric=rubric, provider=provider,
                audit=audit, model=model,
            )

        for run_idx in range(2):
            judge_dir = tmp_path / f"judge_{run_idx}"
            with patch(
                "traces.judge.orchestrator.call_panel_judge",
                side_effect=lambda **kw: capturing_judge(**kw),
            ):
                artifacts = run_score_judge_stage(
                    raw_results_path=raw_path,
                    papers_by_id=papers_by_id,
                    scorer_factory=fake_scorer_factory,
                    rubric="rubric",
                    judge_dir=judge_dir,
                    providers=providers,
                    audit=audit,
                    concurrency=1,
                    max_cost_usd=0.0,
                    sample=3,
                    sample_seed=99,
                )
            verdicts_path = judge_dir / "judge_verdicts.json"
            verdicts = json.loads(verdicts_path.read_text())
            judged_probe_ids_run.append(sorted(verdicts.keys()))

        assert judged_probe_ids_run[0] == judged_probe_ids_run[1], (
            "Same seed must produce the same cache-key selection across two runs"
        )
