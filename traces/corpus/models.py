"""
Corpus data models for TRACES.

PaperRecord is the central type: one YAML file per paper,
containing all metadata, ATLAS classification, probe definition,
and withheld details.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr


class WithheldDetail(BaseModel):
    id: str
    text: str
    level: int  # 1=general domain, 2=paper-specific plausible, 3=paper-specific distinctive
    rationale: str = ""
    match_type: Literal["exact_list", "exact_list_case_insensitive", "phrase_match"] = "phrase_match"


class AssociatedPerson(BaseModel):
    name: str
    legal_consequences: str = "none"


class RetractionRecord(BaseModel):
    date: str
    reason: str
    notice_doi: Optional[str] = None
    severity: str  # "catastrophic", "major", "standard"
    associated_persons: List[AssociatedPerson] = Field(default_factory=list)


class ATLASAnnotation(BaseModel):
    primary_unreliability_mode: Optional[str] = None
    secondary_unreliability_modes: List[str] = Field(default_factory=list)
    claimed_domain: Optional[str] = None
    detection_markers: List[str] = Field(default_factory=list)
    default_severity: float = 0.0
    retraction: Optional[RetractionRecord] = None


class ProbeDefinition(BaseModel):
    claim_type: str  # mechanism_claim, efficacy_claim, existence_claim

    central_claim: str = ""

    preamble: str = ""
    operational_request: str = ""

    withheld_details: List[WithheldDetail] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class AnnotationProvenance(BaseModel):
    annotator: str = ""
    reviewer: str = ""
    review_status: str = "pending"  # accepted, disputed, pending
    date_annotated: Optional[str] = None
    date_reviewed: Optional[str] = None
    notes: str = ""


def extract_last_names(authors: list[str]) -> set[str]:
    """
    Extract last names from author strings like `First Last`, `F Last`,
    `F. M. Last`, or `Last`.
    """
    last_names: dict[str, str] = {}
    for author in authors:
        parts = [part.strip(".,") for part in author.split() if part.strip(".,")]
        if not parts:
            continue
        last = parts[-1]
        if len(last) < 3 or not last[0].isupper():
            continue
        last_names.setdefault(last.lower(), last)
    return set(last_names.values())


class PaperRecord(BaseModel):
    """Complete record for one paper in the TRACES corpus."""

    paper_id: str
    doi: Optional[str] = None
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    pdf_sha256: Optional[str] = None

    atlas: ATLASAnnotation = Field(default_factory=ATLASAnnotation)
    probe: ProbeDefinition
    annotation: AnnotationProvenance = Field(default_factory=AnnotationProvenance)

    # Set by CorpusLoader from the family folder name. Not part of YAML.
    _domain: str = PrivateAttr(default="")
    _authors_last_names: set[str] | None = PrivateAttr(default=None)

    model_config = {"extra": "forbid"}

    @property
    def has_retraction(self) -> bool:
        return self.atlas.retraction is not None

    @property
    def domain(self) -> str:
        """Top-level family folder name. Set by the loader; never serialized."""
        return self._domain

    @property
    def authors_last_names(self) -> set[str]:
        if self._authors_last_names is None:
            self._authors_last_names = extract_last_names(self.authors)
        return set(self._authors_last_names)

    def full_prompt(self) -> str:
        """Construct the full user prompt from preamble + operational request."""
        parts = []
        if self.probe.preamble.strip():
            parts.append(self.probe.preamble.strip())
        if self.probe.operational_request.strip():
            parts.append(self.probe.operational_request.strip())
        return "\n\n".join(parts)
