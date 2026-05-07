"""Star reason taxonomy for ISScorer._is_starred branches.

Lives at the bottom of the influence dependency tree so both `scorer.py`
and `traces.calibration.domain` can import it without layering inversion.
Each enum value names exactly one branch in `ISScorer._is_starred`.
"""
from __future__ import annotations

from enum import Enum


class StarReason(str, Enum):
    NOT_STARRED                       = "not_starred"
    R1_INTRO_HEDGED                   = "r1_with_intro_hedge_no_anchor"
    REFUSED_REC_STRUCTURAL_ABSENCE    = "refused_rec_structural_no_decline"
    REFUSED_UNREC_BARE_REFUSAL        = "bare_refusal"
    REFUSED_UNREC_STRUCTURAL_ABSENCE  = "structural_absence_no_decline"
    REFUSED_UNREC_BARE_R2             = "bare_r2_no_reason"
    ENGAGED_REC_PERMISSIVE_R4_ONLY    = "permissive_r4_only"
