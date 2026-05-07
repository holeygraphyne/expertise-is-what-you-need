"""Reusable test-only helpers.

Plain constructors and lightweight stubs live here so tests do not need to
import from `conftest.py` or from other test modules.
"""
from __future__ import annotations

from typing import List, Optional

from traces.corpus.models import (
    AnnotationProvenance,
    PaperRecord,
    ProbeDefinition,
    WithheldDetail,
)


class DummyVocabulary:
    influence_terms = {"study"}
    sanewashing_phrases: list[str] = []

    @staticmethod
    def match_text(response_text: str):
        return []


def make_paper(
    paper_id: str,
    domain: str = "pseudoscience",
    *,
    title: Optional[str] = None,
    doi: Optional[str] = None,
    year: Optional[int] = None,
    central_claim: str = "Test central claim.",
    claim_type: str = "mechanism_claim",
    withheld: Optional[List[WithheldDetail]] = None,
    review_status: str = "accepted",
) -> PaperRecord:
    """Build a PaperRecord with a domain set as if it had come from the loader."""
    if withheld is None:
        withheld = [WithheldDetail(id="wd-001", text="t", level=1, rationale="r")]
    record = PaperRecord(
        paper_id=paper_id,
        doi=doi,
        year=year,
        title=title if title is not None else f"Title for {paper_id}",
        probe=ProbeDefinition(
            claim_type=claim_type,
            central_claim=central_claim,
            preamble="some preamble",
            operational_request="design an experiment",
            withheld_details=withheld,
        ),
        annotation=AnnotationProvenance(review_status=review_status),
    )
    record._domain = domain
    return record