"""Tests for traces.calibration.payload."""
from unittest.mock import MagicMock

from traces.calibration.payload import build_judge_payload
from traces.influence.scorer import (
    ISResult,
    ResponseClassification,
)


def _mk_paper(paper_id: str = "p1", family: str = "pseudoscience"):
    paper = MagicMock()
    paper.paper_id = paper_id
    paper.domain = family
    paper.atlas.primary_unreliability_mode = "atlas:UnfalsifiableDesign"
    paper.probe.claim_type = "cargo_cult_experiment"
    paper.probe.central_claim = "X cures Y via aggregation."
    paper.probe.operational_request = "Design a follow-up study."
    return paper


def _mk_is_result(
    classification: ResponseClassification,
) -> ISResult:
    return ISResult(
        probe_id="IS-p1",
        model="openai/gpt-5.4",
        classification=classification,
        refused=classification in {
            ResponseClassification.REFUSED_RECOGNIZED,
            ResponseClassification.REFUSED_UNRECOGNIZED,
        },
        recognized=classification in {
            ResponseClassification.REFUSED_RECOGNIZED,
            ResponseClassification.ENGAGED_RECOGNIZED,
        },
        recognition_is_strong=classification in {
            ResponseClassification.REFUSED_RECOGNIZED,
            ResponseClassification.ENGAGED_RECOGNIZED,
        },
        starred=False,
        response_text="x" * 100,
    )


class TestBuildJudgePayload:
    def test_contains_probe_id_and_response(self):
        paper = _mk_paper()
        is_result = _mk_is_result(ResponseClassification.ENGAGED_UNRECOGNIZED)
        payload = build_judge_payload(
            probe_id="IS-p1",
            paper=paper,
            response_text="THE RESPONSE TEXT.",
            is_result=is_result,
        )
        assert "IS-p1" in payload
        assert "THE RESPONSE TEXT." in payload

    def test_contains_operational_request(self):
        paper = _mk_paper()
        is_result = _mk_is_result(ResponseClassification.ENGAGED_UNRECOGNIZED)
        payload = build_judge_payload(
            probe_id="IS-p1",
            paper=paper,
            response_text="resp",
            is_result=is_result,
        )
        assert "Design a follow-up study." in payload

    def test_contains_scorer_output(self):
        paper = _mk_paper()
        is_result = _mk_is_result(ResponseClassification.REFUSED_RECOGNIZED)
        payload = build_judge_payload(
            probe_id="IS-p1",
            paper=paper,
            response_text="resp",
            is_result=is_result,
        )
        assert "REFUSED_RECOGNIZED" in payload
        assert "refused: true" in payload
        assert "recognized: true" in payload
        assert "recognition_is_strong: true" in payload

    def test_contains_ifr_flags(self):
        paper = _mk_paper()
        is_result = _mk_is_result(ResponseClassification.REFUSED_UNRECOGNIZED)
        payload = build_judge_payload(
            probe_id="IS-p1", paper=paper,
            response_text="resp", is_result=is_result,
        )
        assert "pass_ifr_a: true" in payload
        assert "pass_ifr_i: true" in payload

    def test_excludes_withheld_details(self):
        paper = _mk_paper()
        paper.probe.withheld_details = ["secret_marker_xyzzy"]
        is_result = _mk_is_result(ResponseClassification.ENGAGED_UNRECOGNIZED)
        payload = build_judge_payload(
            probe_id="IS-p1", paper=paper,
            response_text="resp", is_result=is_result,
        )
        assert "xyzzy" not in payload
        assert "withheld" not in payload.lower()

    def test_excludes_preamble(self):
        paper = _mk_paper()
        paper.probe.preamble = "SENSITIVE PREAMBLE CONTENT SHOULD NOT APPEAR"
        is_result = _mk_is_result(ResponseClassification.ENGAGED_UNRECOGNIZED)
        payload = build_judge_payload(
            probe_id="IS-p1", paper=paper,
            response_text="resp", is_result=is_result,
        )
        assert "SENSITIVE PREAMBLE" not in payload


def test_payload_includes_star_reason_when_starred():
    """When the scorer starred a response, the payload surfaces which
    branch fired so the judge can reason about scorer hesitation."""
    from traces.influence.scorer import ResponseClassification
    from traces.influence.star_reason import StarReason
    paper = _mk_paper()
    is_result = _mk_is_result(
        ResponseClassification.REFUSED_UNRECOGNIZED,
    )
    is_result.starred = True
    is_result.star_reason = StarReason.REFUSED_UNREC_BARE_REFUSAL
    payload = build_judge_payload(
        probe_id="IS-p1",
        paper=paper,
        response_text="I'm sorry, but I can't help with that.",
        is_result=is_result,
    )
    assert "star_reason: bare_refusal" in payload


def test_payload_omits_star_reason_when_not_starred():
    from traces.influence.scorer import ResponseClassification
    paper = _mk_paper()
    is_result = _mk_is_result(ResponseClassification.ENGAGED_UNRECOGNIZED)
    # is_result.starred defaults to False; star_reason is NOT_STARRED.
    payload = build_judge_payload(
        probe_id="IS-p1",
        paper=paper,
        response_text="Here is a study design.",
        is_result=is_result,
    )
    assert "star_reason:" not in payload
