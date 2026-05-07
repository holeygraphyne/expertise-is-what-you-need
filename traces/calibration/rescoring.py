"""Re-score raw responses with the current ISScorer.

The audit tool re-scores fresh rather than reading probe_scores.json,
so its diff always reflects the scorer's present behavior — critical
for the patch → re-audit iteration loop.

Per-paper vocabulary construction follows the same pattern as
`traces/__main__.py::_report_is_for_run`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from traces.calibration.cache import cache_key
from traces.influence.scorer import (
    ISResult,
    ISScorer,
    ScoringResources,
    _sanewashing_author_last_names,
)

logger = logging.getLogger(__name__)

class _AtlasLike(Protocol):
    primary_unreliability_mode: str


class PaperLike(Protocol):
    atlas: _AtlasLike

    @property
    def has_retraction(self) -> bool: ...


@dataclass
class ScoredResponse:
    """Raw response + its fresh ISResult + paper metadata."""
    raw_result: object
    paper: object
    is_result: ISResult


def rescore_responses(
        *,
        raw_results: Iterable,
        papers_by_id: dict,
        scorer_factory: Callable[[PaperLike], ISScorer],
) -> dict[str, ScoredResponse]:
    """Re-score each raw result with a per-paper ISScorer.

    `scorer_factory(paper) -> ISScorer` is injected so tests can use
    lightweight stand-ins. In production it builds a `Vocabulary` via
    VocabularyLoader and instantiates `ISScorer(vocabulary, config)`.

    Returns a dict keyed by cache_key so downstream diff + payload
    builders can look up both the ISResult and the paper by the same
    stable key.

    Skips responses where the raw result is errored or the paper is
    missing from the corpus (orphan response). Empty/null-content
    responses are still scored so scorer/report semantics stay aligned.
    """
    out: dict[str, ScoredResponse] = {}
    orphan_count = 0
    scorer_cache: dict[str, ISScorer] = {}

    for raw in raw_results:
        if getattr(raw, "error", None):
            continue
        paper = papers_by_id.get(raw.paper_id)
        if paper is None:
            orphan_count += 1
            continue

        if raw.paper_id not in scorer_cache:
            scorer_cache[raw.paper_id] = scorer_factory(paper)
        scorer = scorer_cache[raw.paper_id]

        is_result = scorer.score(
            response_text=raw.response_text,
            probe_id=raw.probe_id,
            model=raw.model,
            preamble=paper.probe.preamble,
            operational_request=paper.probe.operational_request,
            central_claim=paper.probe.central_claim,
            withheld_details=list(paper.probe.withheld_details),
            probe_authors=paper.authors_last_names,
            paper_year=getattr(paper, "year", None),
            sanewashing_author_last_names=_sanewashing_author_last_names(
                getattr(paper, "authors", [])
            ),
        )

        key = cache_key(raw.probe_id, raw.model, raw.response_text)
        out[key] = ScoredResponse(
            raw_result=raw, paper=paper, is_result=is_result
        )

    if orphan_count:
        logger.info(
            "rescoring: skipped %d orphan responses (no matching paper)",
            orphan_count,
        )
    return out


def make_scorer_factory(
        vocab_loader, config_scoring
) -> Callable[[PaperLike], ISScorer]:
    """Build a scorer_factory closure suitable for production use.

    Signature matches `rescore_responses`'s `scorer_factory` parameter.
    """
    resources = ScoringResources.build()

    def _factory(paper: PaperLike) -> ISScorer:
        vocab = vocab_loader.load_rejection_vocabulary(
            mode=paper.atlas.primary_unreliability_mode,
            has_retraction=paper.has_retraction,
        )
        return ISScorer(
            vocabulary=vocab,
            config=config_scoring,
            resources=resources,
        )
    return _factory