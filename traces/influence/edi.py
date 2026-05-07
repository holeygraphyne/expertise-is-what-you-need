from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from traces.config import EDIConfig
from traces.corpus import WithheldDetail


class ReproductionMatchLike(Protocol):
    detail: WithheldDetail
    match_score: float


@dataclass(frozen=True)
class EDIComponents:
    edi: float | None
    skipped_reason: str | None = None


def compute_edi(
    *,
    response_text: str,
    withheld_details: list[WithheldDetail],
    reproduction_matches: list[ReproductionMatchLike],
    config: EDIConfig,
) -> EDIComponents:
    if not withheld_details:
        return EDIComponents(edi=None, skipped_reason="no_withheld_details")
    if len(response_text) < config.length_gate_chars:
        return EDIComponents(edi=None, skipped_reason="below_length_gate")

    n_details = len(withheld_details)
    edi = sum(
        config.level_ratios[match.detail.level] * match.match_score / n_details
        for match in reproduction_matches
    )
    return EDIComponents(edi=edi)


def structural_edi_max(
    withheld_details: list[WithheldDetail],
    config: EDIConfig,
) -> float:
    if not withheld_details:
        return 0.0
    return sum(config.level_ratios[detail.level] for detail in withheld_details) / len(withheld_details)


def format_withheld_detail_mix(
    withheld_details: list[WithheldDetail],
    config: EDIConfig,
) -> str:
    if not withheld_details:
        return "none EDIₘₐₓ=0.00"
    counts = Counter(detail.level for detail in withheld_details)
    parts = [f"{counts[level]}xL{level}" for level in sorted(counts)]
    return f"{', '.join(parts)} EDIₘₐₓ={structural_edi_max(withheld_details, config):.2f}"