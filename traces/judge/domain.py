"""Pure domain services for scorer-native judge verdicts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from traces.judge.aggregation import AggregatedResult, aggregate_panel_verdict
from traces.judge.consistency import check_consistency
from traces.judge.models import (
    JudgeVerdict,
    PanelVerdict,
    ReviewQueueRow,
    pick_primary_disagreement_kind,
)
from traces.judge.review_queue import collect_triggers


@dataclass(frozen=True)
class PanelEvaluation:
    """Domain result for one benchmark response after panel aggregation."""

    panel_verdict: PanelVerdict
    cache_entry: dict
    review_queue_row: Optional[dict]
    triggers: list[str]
    consistency_violations: list[str]


def per_judge_to_serializable(per_judge: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for member_id, value in per_judge.items():
        if isinstance(value, JudgeVerdict):
            out[member_id] = value.model_dump(by_alias=True)
        else:
            out[member_id] = value
    return out


def per_judge_from_serializable(raw: dict) -> dict[str, object]:
    out: dict[str, object] = {}
    for member_id, value in raw.items():
        if isinstance(value, dict) and "error" in value and "label" not in value:
            out[member_id] = value
        else:
            out[member_id] = JudgeVerdict.model_validate(value)
    return out


def evaluate_panel_row(
    *,
    probe_id: str,
    model: str,
    response_sha256: str,
    response_text: str,
    deterministic_label: str,
    per_judge: dict[str, object],
    judge_version: str,
    harm_aggregation: str,
    evidence_issues: list[str],
) -> PanelEvaluation:
    """Aggregate and route one row without doing IO or HTTP."""
    agg: AggregatedResult = aggregate_panel_verdict(
        per_judge=per_judge,
        harm_aggregation=harm_aggregation,
    )
    violations = check_consistency(agg) + evidence_issues
    triggers = collect_triggers(
        deterministic_label=deterministic_label,
        agg=agg,
        per_judge=per_judge,
        consistency_violations=violations,
    )

    panel_verdict = PanelVerdict.model_validate({
        "probe_id": probe_id,
        "model": model,
        "response_sha256": response_sha256,
        "per_judge": per_judge_to_serializable(per_judge),
        "aggregated_label": agg.aggregated_label,
        "aggregated_pass_ifr_a": agg.aggregated_pass_ifr_a,
        "aggregated_pass_ifr_i": agg.aggregated_pass_ifr_i,
        "aggregated_refused": agg.aggregated_refused,
        "aggregated_recognized": agg.aggregated_recognized,
        "aggregated_delivered_operational_content": (
            agg.aggregated_delivered_operational_content
        ),
        "aggregated_source_identified": agg.aggregated_source_identified,
        "aggregated_sanewashing": agg.aggregated_sanewashing,
        "evidence_validation_issues": evidence_issues,
        "intra_panel_kappa": agg.intra_panel_kappa,
        "refusal_count": agg.refusal_count,
        "judge_version": judge_version,
    })

    review_queue_row: dict | None = None
    if triggers:
        primary = pick_primary_disagreement_kind(triggers)
        consistency_text = "; ".join(violations) if violations else None
        review_row = ReviewQueueRow.model_validate({
            "probe_id": probe_id,
            "model": model,
            "response_sha256": response_sha256,
            "deterministic_label": deterministic_label,
            "panel_verdict": panel_verdict.model_dump(),
            "disagreement_kind": primary,
            "all_triggers": triggers,
            "consistency_violation": consistency_text,
            "response_excerpt_first_2000": response_text[:2000],
            "human_label": None,
            "human_notes": None,
        })
        review_queue_row = review_row.model_dump()

    return PanelEvaluation(
        panel_verdict=panel_verdict,
        cache_entry={
            "panel_verdict": panel_verdict.model_dump(),
            "_judge_version": judge_version,
        },
        review_queue_row=review_queue_row,
        triggers=triggers,
        consistency_violations=violations,
    )


def review_queue_row_from_cached_verdict(
    *,
    probe_id: str,
    model: str,
    response_sha256: str,
    response_text: str,
    deterministic_label: str,
    cache_entry: dict,
) -> Optional[dict]:
    """Rebuild review-queue state from a cached panel verdict."""
    pv_data = cache_entry.get("panel_verdict")
    if pv_data is None:
        return None
    per_judge = per_judge_from_serializable(pv_data.get("per_judge") or {})
    agg = AggregatedResult(
        aggregated_label=pv_data.get("aggregated_label"),
        aggregated_pass_ifr_a=pv_data.get("aggregated_pass_ifr_a"),
        aggregated_pass_ifr_i=pv_data.get("aggregated_pass_ifr_i"),
        aggregated_refused=pv_data.get("aggregated_refused"),
        aggregated_recognized=pv_data.get("aggregated_recognized"),
        aggregated_delivered_operational_content=(
            pv_data.get("aggregated_delivered_operational_content")
        ),
        aggregated_source_identified=pv_data.get("aggregated_source_identified"),
        aggregated_sanewashing=pv_data.get("aggregated_sanewashing"),
        intra_panel_kappa=pv_data.get("intra_panel_kappa"),
        refusal_count=pv_data.get("refusal_count", 0),
    )
    violations = check_consistency(agg) + list(
        pv_data.get("evidence_validation_issues") or []
    )
    triggers = collect_triggers(
        deterministic_label=deterministic_label,
        agg=agg,
        per_judge=per_judge,
        consistency_violations=violations,
    )
    if not triggers:
        return None
    primary = pick_primary_disagreement_kind(triggers)
    consistency_text = "; ".join(violations) if violations else None
    review_row = ReviewQueueRow.model_validate({
        "probe_id": probe_id,
        "model": model,
        "response_sha256": response_sha256,
        "deterministic_label": deterministic_label,
        "panel_verdict": pv_data,
        "disagreement_kind": primary,
        "all_triggers": triggers,
        "consistency_violation": consistency_text,
        "response_excerpt_first_2000": response_text[:2000],
        "human_label": None,
        "human_notes": None,
    })
    return review_row.model_dump()
