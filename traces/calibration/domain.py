"""Pure domain types for the calibration bounded context.

No IO, no LLM client, no spaCy. Importable from anywhere in the codebase
without side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from pydantic import BaseModel, Field

# Re-export StarReason from the influence layer so calibration code
# imports from a single domain module.
from traces.influence.star_reason import StarReason

__all__ = [
    "OptimizationFinding",
    "PaperMetadata",
    "StarReason",
    "StarredCase",
]


@dataclass(frozen=True)
class PaperMetadata:
    """Projection of PaperRecord fields the auditor actually consumes."""
    paper_id: str
    domain: str
    central_claim: str
    operational_request: str
    atlas_unreliability_mode: str
    claim_type: str


@dataclass(frozen=True)
class StarredCase:
    """One unit of work for the judge stage.

    `cache_key` is `cache_key(probe_id, model, response_text)` from
    `traces.calibration.cache` — the same key the judge cache uses, so
    selection and verdict-lookup share semantics.
    """
    cache_key: str
    probe_id: str
    model: str
    response_text: str
    paper_metadata: PaperMetadata
    scorer_classification: str          # ResponseClassification.value
    scorer_pass_ifr_a: bool
    scorer_pass_ifr_i: bool
    star_reason: StarReason


class OptimizationFinding(BaseModel):
    """A single recommendation from the recommender stage.

    `target_kind` is closed; pydantic rejects values outside the set.
    `target_locator` is free-text but must reference a symbol listed
    in `traces/calibration/scorer_map.md` — checked downstream.
    """
    target_kind: Literal["lexicon", "matcher", "logic", "threshold"]
    target_locator: str = Field(min_length=1)
    rule_gap: str = Field(min_length=1)
    evidence_case_keys: List[str] = Field(default_factory=list)
    hypothesis: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)
