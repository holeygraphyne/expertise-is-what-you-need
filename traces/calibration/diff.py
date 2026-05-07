"""Diff scorer classifications against judge labels.

Builds the disagreements.json payload + clusters + rule_gap aggregates
that feed the judge report and the recommender.
"""
from __future__ import annotations

from collections import defaultdict

from traces.calibration.models import (
    Disagreement,
    JudgeLabel,
    PASS_CLASSIFICATIONS,
)
from traces.calibration.rescoring import ScoredResponse


def tag_direction(scorer_classification: str, judge_classification: str) -> str:
    """Return one of pass_to_fail / fail_to_pass / fail_to_fail / pass_to_pass."""
    scorer_pass = scorer_classification in PASS_CLASSIFICATIONS
    judge_pass = judge_classification in PASS_CLASSIFICATIONS
    if scorer_pass and not judge_pass:
        return "pass_to_fail"
    if not scorer_pass and judge_pass:
        return "fail_to_pass"
    if not scorer_pass and not judge_pass:
        return "fail_to_fail"
    return "pass_to_pass"


def build_disagreements(
    scored_by_key: dict[str, ScoredResponse],
    judged_by_key: dict[str, JudgeLabel],
    errored_keys: list[str],
) -> list[Disagreement]:
    """Emit a Disagreement row for each (scored, judged) pair that diverges.

    - Agreements (same classification enum) produce no row.
    - Error keys (judge failed) are excluded from the diff entirely;
      the caller surfaces them in a separate `errored` section.
    """
    errored_set = set(errored_keys)
    out: list[Disagreement] = []
    for key, sr in scored_by_key.items():
        if key in errored_set:
            continue
        judged = judged_by_key.get(key)
        if judged is None:
            continue
        scorer_class = sr.is_result.classification.value
        judge_class = judged.classification
        if scorer_class == judge_class:
            continue
        excerpt = sr.raw_result.response_text[:800]
        out.append(Disagreement(
            probe_id=sr.raw_result.probe_id,
            model=sr.raw_result.model,
            scorer_classification=scorer_class,
            scorer_pass=scorer_class in PASS_CLASSIFICATIONS,
            judge_classification=judge_class,
            judge_pass=judge_class in PASS_CLASSIFICATIONS,
            direction=tag_direction(scorer_class, judge_class),
            judge_reason=judged.primary_reason,
            judge_key_spans=judged.key_spans,
            judge_confidence=judged.confidence,
            judge_rule_gap=judged.rule_gap,
            response_excerpt_first_800=excerpt,
        ))
    return out


def cluster_disagreements(
    disagreements: list[Disagreement],
) -> dict[tuple[str, str], list[Disagreement]]:
    """Group by (scorer_classification, judge_classification) pair."""
    clusters: dict[tuple[str, str], list[Disagreement]] = defaultdict(list)
    for d in disagreements:
        clusters[(d.scorer_classification, d.judge_classification)].append(d)
    return dict(clusters)


def aggregate_rule_gaps(
    disagreements: list[Disagreement],
) -> dict[str, dict[str, int]]:
    """Return `{rule_gap: {direction: count}}`. Excludes null rule_gap."""
    agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in disagreements:
        if d.judge_rule_gap is None:
            continue
        agg[d.judge_rule_gap][d.direction] += 1
    return {k: dict(v) for k, v in agg.items()}
