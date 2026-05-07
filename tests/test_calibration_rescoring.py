"""Tests for traces.calibration.rescoring."""
from unittest.mock import MagicMock

from traces.calibration import rescoring as rescoring_module

from traces.calibration.rescoring import (
    ScoredResponse,
    make_scorer_factory,
    rescore_responses,
)


def _mk_raw(probe_id="IS-p1", model="openai/gpt-5.4",
            response_text="Response.", paper_id="p1", error=None):
    r = MagicMock()
    r.probe_id = probe_id
    r.model = model
    r.response_text = response_text
    r.paper_id = paper_id
    r.error = error
    return r


def _mk_paper(paper_id="p1", family="pseudoscience"):
    p = MagicMock()
    p.paper_id = paper_id
    p.domain = family
    p.atlas.primary_unreliability_mode = "atlas:UnfalsifiableDesign"
    p.has_retraction = False
    p.probe.claim_type = "cargo_cult_experiment"
    p.probe.central_claim = "X cures Y."
    p.probe.operational_request = "Design X."
    p.probe.withheld_details = []
    return p


class TestRescoreResponses:
    def test_rescores_each_response(self):
        raws = [
            _mk_raw(probe_id="IS-p1", response_text="R1"),
            _mk_raw(probe_id="IS-p1", response_text="R2", model="qwen/a"),
        ]
        papers = {"p1": _mk_paper("p1")}

        mock_scorer = MagicMock()
        mock_scorer.score.side_effect = lambda **kw: MagicMock(
            probe_id=kw["probe_id"],
            model=kw["model"],
            classification=MagicMock(value="FULL_ENGAGEMENT"),
            blocks=[],
            response_text=kw["response_text"],
            cargo_cult_compliance_hits=0,
        )
        mock_scorer_factory = MagicMock(return_value=mock_scorer)

        out = rescore_responses(
            raw_results=raws,
            papers_by_id=papers,
            scorer_factory=mock_scorer_factory,
        )
        assert len(out) == 2
        for sr in out.values():
            assert isinstance(sr, ScoredResponse)
            assert sr.paper is papers["p1"]
            assert sr.is_result is not None

    def test_skips_errored_raws(self):
        raws = [
            _mk_raw(response_text="R1", error=None),
            _mk_raw(response_text="", error="upstream failure"),
        ]
        papers = {"p1": _mk_paper("p1")}
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MagicMock(
            classification=MagicMock(value="FULL_ENGAGEMENT"),
            blocks=[], cargo_cult_compliance_hits=0,
        )

        out = rescore_responses(
            raw_results=raws,
            papers_by_id=papers,
            scorer_factory=lambda paper: mock_scorer,
        )
        assert len(out) == 1

    def test_scores_empty_non_errored_raws(self):
        raws = [_mk_raw(response_text="", error=None)]
        papers = {"p1": _mk_paper("p1")}
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = MagicMock(
            classification=MagicMock(value="REFUSED_UNRECOGNIZED"),
            refused=True,
            recognized=False,
            null_content_kind="empty",
        )

        out = rescore_responses(
            raw_results=raws,
            papers_by_id=papers,
            scorer_factory=lambda paper: mock_scorer,
        )

        assert len(out) == 1
        mock_scorer.score.assert_called_once()

    def test_skips_orphans(self):
        raws = [_mk_raw(paper_id="missing", response_text="R")]
        papers = {"p1": _mk_paper("p1")}
        out = rescore_responses(
            raw_results=raws,
            papers_by_id=papers,
            scorer_factory=lambda paper: MagicMock(),
        )
        assert out == {}


def test_make_scorer_factory_builds_shared_resources_once(monkeypatch):
    fake_resources = object()
    build_calls = []
    scorer_calls = []

    class FakeScoringResources:
        @classmethod
        def build(cls):
            build_calls.append(True)
            return fake_resources

    class FakeScorer:
        def __init__(self, *, vocabulary, config, resources):
            scorer_calls.append((vocabulary, config, resources))

    monkeypatch.setattr(rescoring_module, "ScoringResources", FakeScoringResources)
    monkeypatch.setattr(rescoring_module, "ISScorer", FakeScorer)

    vocab_loader = MagicMock()
    vocab_loader.load_rejection_vocabulary.side_effect = ["vocab-a", "vocab-b"]
    factory = make_scorer_factory(vocab_loader, "config")

    paper_a = _mk_paper("p1")
    paper_b = _mk_paper("p2")
    paper_b.atlas.primary_unreliability_mode = "atlas:OtherMode"
    paper_b.has_retraction = True

    factory(paper_a)
    factory(paper_b)

    assert build_calls == [True]
    assert scorer_calls == [
        ("vocab-a", "config", fake_resources),
        ("vocab-b", "config", fake_resources),
    ]
