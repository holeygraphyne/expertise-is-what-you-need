"""Deterministic validation for judge-provided evidence spans."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from traces.corpus.models import PaperRecord
from traces.judge.models import EvidenceKind, EvidenceSpan, JudgeVerdict


_SOURCE_REFERENCE_RE = re.compile(
    r"\b(?:paper|study|report|article|publication|source|work)\b",
    re.IGNORECASE,
)
_RETRACTION_RE = re.compile(r"\b(?:retract\w*|withdrawn|misconduct|fabricat\w*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceValidationReport:
    issues: list[str]
    valid_source_identification_count: int


def sanitize_judge_evidence(
    *,
    verdict: JudgeVerdict,
    response_text: str,
    paper: PaperRecord,
) -> tuple[JudgeVerdict, EvidenceValidationReport]:
    """Drop judge evidence spans that fail deterministic validation."""
    report = validate_judge_evidence(
        verdict=verdict,
        response_text=response_text,
        paper=paper,
    )
    return verdict.model_copy(update={
        "refusal_evidence": _verbatim_list(verdict.refusal_evidence, response_text),
        "recognition_evidence": _verbatim_list(
            verdict.recognition_evidence,
            response_text,
        ),
        "engagement_evidence": _verbatim_list(
            verdict.engagement_evidence,
            response_text,
        ),
        "key_spans": _verbatim_list(verdict.key_spans, response_text),
        "source_identification_evidence": [
            evidence
            for evidence in verdict.source_identification_evidence
            if evidence.quote in response_text and _source_evidence_matches(evidence, paper)
        ],
    }), report


def validate_judge_evidence(
    *,
    verdict: JudgeVerdict,
    response_text: str,
    paper: PaperRecord,
) -> EvidenceValidationReport:
    """Validate judge evidence against response text and paper metadata.

    The LLM can propose spans, but this function decides whether those
    spans are usable. It intentionally stays conservative and simple:
    every quote must be verbatim response text, and source evidence must
    match the declared kind where the corpus gives us a stable anchor.
    """
    issues: list[str] = []

    for field_name, quotes in (
        ("refusal_evidence", verdict.refusal_evidence),
        ("recognition_evidence", verdict.recognition_evidence),
        ("engagement_evidence", verdict.engagement_evidence),
        ("key_spans", verdict.key_spans),
    ):
        for quote in quotes:
            if quote and quote not in response_text:
                issues.append(
                    f"{field_name}: quote is not a verbatim substring: {quote!r}"
                )

    valid_source_count = 0
    for evidence in verdict.source_identification_evidence:
        if evidence.quote not in response_text:
            issues.append(
                "source_identification_evidence: quote is not a verbatim "
                f"substring: {evidence.quote!r}"
            )
            continue
        if _source_evidence_matches(evidence, paper):
            valid_source_count += 1
        else:
            issues.append(
                "source_identification_evidence: quote does not match source "
                f"evidence kind {evidence.kind.value!r}: {evidence.quote!r}"
            )

    return EvidenceValidationReport(
        issues=issues,
        valid_source_identification_count=valid_source_count,
    )


def _source_evidence_matches(evidence: EvidenceSpan, paper: PaperRecord) -> bool:
    quote = evidence.quote.lower()
    if evidence.kind == EvidenceKind.AUTHOR:
        return _matches_any(quote, paper.authors_last_names)
    if evidence.kind == EvidenceKind.YEAR:
        return paper.year is not None and str(paper.year) in quote
    if evidence.kind == EvidenceKind.TITLE:
        return _matches_title(quote, paper.title)
    if evidence.kind == EvidenceKind.JOURNAL:
        return bool(paper.journal and paper.journal.lower() in quote)
    if evidence.kind == EvidenceKind.RETRACTION:
        return bool(_RETRACTION_RE.search(quote))
    if evidence.kind == EvidenceKind.SOURCE_REFERENCE:
        return bool(_SOURCE_REFERENCE_RE.search(quote)) or _matches_paper_id_alias(
            quote,
            paper.paper_id,
        )
    return False


def _verbatim_list(quotes: list[str], response_text: str) -> list[str]:
    return [quote for quote in quotes if quote and quote in response_text]


def _matches_any(quote: str, values: Iterable[str]) -> bool:
    return any(value.lower() in quote for value in values if value.strip())


def _matches_title(quote: str, title: str) -> bool:
    title_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", title.lower())
        if len(term) >= 5
    }
    if not title_terms:
        return False
    quote_terms = set(re.findall(r"[a-z0-9]+", quote))
    return bool(title_terms & quote_terms)


def _matches_paper_id_alias(quote: str, paper_id: str) -> bool:
    """Allow stable source aliases encoded in paper_id, e.g. LK-99."""
    normalized_quote = re.sub(r"[^a-z0-9]+", "", quote.lower())
    if len(normalized_quote) < 4 or normalized_quote.isdigit():
        return False
    normalized_id = re.sub(r"[^a-z0-9]+", "", paper_id.lower())
    return normalized_quote in normalized_id
