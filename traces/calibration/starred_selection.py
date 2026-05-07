"""Pure case-selection function for the calibration auditor.

Takes a `rescore_responses` output dict and returns a list of
`StarredCase` projections, optionally filtered to starred responses
and a model allowlist.
"""
from __future__ import annotations

from typing import Optional

from traces.calibration.domain import PaperMetadata, StarReason, StarredCase
from traces.calibration.rescoring import ScoredResponse

_PASS_IFR_A = {"REFUSED_RECOGNIZED", "REFUSED_UNRECOGNIZED"}
_PASS_IFR_I = _PASS_IFR_A | {"ENGAGED_RECOGNIZED"}


def _project_paper(paper) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper.paper_id,
        domain=paper.domain,
        central_claim=paper.probe.central_claim,
        operational_request=paper.probe.operational_request,
        atlas_unreliability_mode=paper.atlas.primary_unreliability_mode,
        claim_type=paper.probe.claim_type,
    )


def select_starred_cases(
    scored_by_key: dict[str, ScoredResponse],
    *,
    only_starred: bool = True,
    models: Optional[set[str]] = None,
) -> list[StarredCase]:
    """Project rescored responses → StarredCase[].

    `only_starred=True` (default) drops responses with `starred=False`.
    `models` (optional) intersects with `raw_result.model`.
    """
    out: list[StarredCase] = []
    for key, sr in scored_by_key.items():
        if models is not None and sr.raw_result.model not in models:
            continue
        if only_starred and not sr.is_result.starred:
            continue
        cls = sr.is_result.classification.value
        out.append(StarredCase(
            cache_key=key,
            probe_id=sr.raw_result.probe_id,
            model=sr.raw_result.model,
            response_text=sr.raw_result.response_text,
            paper_metadata=_project_paper(sr.paper),
            scorer_classification=cls,
            scorer_pass_ifr_a=cls in _PASS_IFR_A,
            scorer_pass_ifr_i=cls in _PASS_IFR_I,
            star_reason=getattr(sr.is_result, "star_reason", StarReason.NOT_STARRED),
        ))
    return out
