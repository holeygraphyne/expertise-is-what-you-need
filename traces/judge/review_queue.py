"""Disagreement-trigger collection for the review queue.

A row enters the review queue when ANY of these conditions hold:
- panel-aggregated label disagrees with the deterministic scorer's
  classification;
- the panel has an intra-panel tie (no majority) — including 1/1/1
  splits and refusal-induced ties;
- the panel produced a 2/1 split AND the minority panel member
  reported high confidence (a confidently-disagreeing minority is
  worth review; a hedging minority is not);
- a logical-consistency rule fired.

The collector returns the FULL set of triggers; the orchestrator
calls `pick_primary_disagreement_kind` from `models.py` to derive
the single primary kind for the row.
"""
from __future__ import annotations

from collections import Counter

from traces.judge.aggregation import AggregatedResult
from traces.judge.models import JudgeVerdict


def collect_triggers(
    *,
    deterministic_label: str,
    agg: AggregatedResult,
    per_judge: dict[str, object],
    consistency_violations: list[str],
) -> list[str]:
    """Return the full list of disagreement triggers for one row.

    Empty list means the row is clean and stays out of the review queue.
    """
    triggers: list[str] = []

    if consistency_violations:
        triggers.append("consistency_violation")

    if agg.aggregated_label is None:
        # No majority — that's a tie (1/1/1 or all-refused).
        triggers.append("intra_panel_tie")
    else:
        # Check intra-panel split: 2/1 vote with high-confidence minority.
        verdicts = [v for v in per_judge.values() if isinstance(v, JudgeVerdict)]
        if len(verdicts) >= 3:
            label_counts = Counter(v.label.value for v in verdicts)
            top_label, top_n = label_counts.most_common(1)[0]
            if top_n == 2 and len(label_counts) == 2:
                # exactly 2/1 split
                minority_verdicts = [
                    v for v in verdicts if v.label.value != top_label
                ]
                if any(v.confidence == "high" for v in minority_verdicts):
                    triggers.append("intra_panel_split")

        if agg.aggregated_label != deterministic_label:
            triggers.append("panel_vs_deterministic")

    return triggers
