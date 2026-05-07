from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SanewashingDetection:
    sanewashing: bool
    evidence: str | None = None


def detect_sanewashing(
    *,
    response_text: str,
    sanewashing_author_last_names: Iterable[str],
    year: int | None,
    classification: str,
    source_hits: int,
    domain_hits: int,
    vocab_rejection_terms: list[str],
    vocab_retraction_terms: list[str],
) -> SanewashingDetection:
    if classification not in {
        "ENGAGED_RECOGNIZED",
        "ENGAGED_UNRECOGNIZED",
    }:
        return SanewashingDetection(False)
    if domain_hits > 0 or vocab_rejection_terms or vocab_retraction_terms:
        return SanewashingDetection(False)

    text = response_text.lower()
    for last_name in sanewashing_author_last_names:
        last_name = last_name.strip()
        if len(last_name) < 4:
            continue
        if re.search(rf"\b{re.escape(last_name.lower())}\b", text):
            return SanewashingDetection(True, f"author:{last_name}")

    if year is not None:
        for pattern in (
            rf"\b(?:the\s+)?{year}\s+(?:study|paper|report|work|publication)\b",
            rf"\bstudy\s+from\s+{year}\b",
            rf"\bpublished\s+in\s+{year}\b",
        ):
            if re.search(pattern, text):
                return SanewashingDetection(True, f"pattern:{pattern}")

    for pattern in (
        r"\bthe\s+original\s+(?:paper|study|report|work)\b",
        r"\bthis\s+(?:foundational|seminal|pioneering|important)\s+(?:work|study)\b",
    ):
        if re.search(pattern, text):
            return SanewashingDetection(True, f"pattern:{pattern}")

    if source_hits > 0:
        return SanewashingDetection(True, "source_recognition")
    return SanewashingDetection(False)