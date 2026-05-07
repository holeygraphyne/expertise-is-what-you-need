"""Unit tests for traces.calibration.starred_selection."""
from unittest.mock import MagicMock

import pytest

from traces.calibration.domain import PaperMetadata, StarReason, StarredCase
from traces.calibration.starred_selection import select_starred_cases


def _mk_scored(*, key: str, model: str, starred: bool, star_reason: StarReason,
               classification: str = "ENGAGED_UNRECOGNIZED",
               response_text: str = "x"):
    sr = MagicMock()
    sr.raw_result.probe_id = "IS-p1"
    sr.raw_result.model = model
    sr.raw_result.response_text = response_text
    sr.raw_result.paper_id = "p1"
    sr.is_result.starred = starred
    sr.is_result.star_reason = star_reason
    sr.is_result.classification.value = classification
    sr.paper.paper_id = "p1"
    sr.paper.domain = "pseudoscience"
    sr.paper.atlas.primary_unreliability_mode = "atlas:Pseudoscience"
    sr.paper.probe.claim_type = "cargo_cult_experiment"
    sr.paper.probe.central_claim = "X cures Y"
    sr.paper.probe.operational_request = "Design X"
    return sr


def test_starred_only_filters_unstarred(tmp_path):
    scored = {
        "k1": _mk_scored(
            key="k1", model="m1", starred=True,
            star_reason=StarReason.REFUSED_UNREC_BARE_REFUSAL,
            classification="REFUSED_UNRECOGNIZED",
        ),
        "k2": _mk_scored(
            key="k2", model="m1", starred=False,
            star_reason=StarReason.NOT_STARRED,
        ),
    }
    out = select_starred_cases(scored, only_starred=True)
    assert {c.cache_key for c in out} == {"k1"}
    assert out[0].star_reason == StarReason.REFUSED_UNREC_BARE_REFUSAL


def test_only_starred_false_keeps_all():
    scored = {
        "k1": _mk_scored(key="k1", model="m1", starred=True,
                         star_reason=StarReason.REFUSED_UNREC_BARE_REFUSAL,
                         classification="REFUSED_UNRECOGNIZED"),
        "k2": _mk_scored(key="k2", model="m1", starred=False,
                         star_reason=StarReason.NOT_STARRED),
    }
    out = select_starred_cases(scored, only_starred=False)
    assert {c.cache_key for c in out} == {"k1", "k2"}


def test_models_filter_intersects():
    scored = {
        "k1": _mk_scored(key="k1", model="m1", starred=True,
                         star_reason=StarReason.REFUSED_UNREC_BARE_REFUSAL,
                         classification="REFUSED_UNRECOGNIZED"),
        "k2": _mk_scored(key="k2", model="m2", starred=True,
                         star_reason=StarReason.REFUSED_UNREC_BARE_REFUSAL,
                         classification="REFUSED_UNRECOGNIZED"),
    }
    out = select_starred_cases(scored, only_starred=True, models={"m1"})
    assert {c.cache_key for c in out} == {"k1"}


def test_returns_starred_case_with_paper_metadata_projection():
    scored = {
        "k1": _mk_scored(key="k1", model="m1", starred=True,
                         star_reason=StarReason.REFUSED_UNREC_BARE_REFUSAL,
                         classification="REFUSED_UNRECOGNIZED"),
    }
    out = select_starred_cases(scored, only_starred=True)
    assert len(out) == 1
    case = out[0]
    assert isinstance(case, StarredCase)
    assert isinstance(case.paper_metadata, PaperMetadata)
    assert case.paper_metadata.domain == "pseudoscience"
    assert case.paper_metadata.atlas_unreliability_mode == "atlas:Pseudoscience"
    assert case.scorer_classification == "REFUSED_UNRECOGNIZED"
    # IFR-a/IFR-i derivation
    assert case.scorer_pass_ifr_a is True   # REFUSED_* counts for IFR-a
    assert case.scorer_pass_ifr_i is True   # and for IFR-i


def test_engaged_recognized_pass_flags():
    scored = {
        "k1": _mk_scored(
            key="k1", model="m1", starred=True,
            star_reason=StarReason.ENGAGED_REC_PERMISSIVE_R4_ONLY,
            classification="ENGAGED_RECOGNIZED",
        ),
    }
    out = select_starred_cases(scored, only_starred=True)
    assert out[0].scorer_pass_ifr_a is False
    assert out[0].scorer_pass_ifr_i is True


def test_empty_input_returns_empty_list():
    assert select_starred_cases({}, only_starred=True) == []
