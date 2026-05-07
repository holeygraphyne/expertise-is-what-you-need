"""
IS probe construction: converts PaperRecord into ISProbe for the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from traces.corpus import PaperRecord, WithheldDetail, RetractionRecord


@dataclass
class ISProbe:
    """Influence Score probe, constructed from a PaperRecord.

    Presents the same interface as other probe types so the runner
    can execute it without modification.
    """

    probe_id: str
    paper_id: str
    doc_id: str  # alias for paper_id, for runner compatibility
    domain: str
    probe_type: str = "influence_premise_acceptance"

    user_prompt: str = ""
    system_prompt: str = ""

    # For scoring (carried through to results)
    withheld_details: List[WithheldDetail] = field(default_factory=list)
    atlas_mode: Optional[str] = None
    retraction: Optional[RetractionRecord] = None

    @classmethod
    def from_paper(cls, paper: PaperRecord) -> "ISProbe":
        """Construct a probe from a paper record."""
        return cls(
            probe_id=f"IS-{paper.paper_id}",
            paper_id=paper.paper_id,
            doc_id=paper.paper_id,
            domain=paper.domain,
            user_prompt=paper.full_prompt(),
            withheld_details=list(paper.probe.withheld_details),
            atlas_mode=paper.atlas.primary_unreliability_mode,
            retraction=paper.atlas.retraction,
        )
