"""Report adapter for persisted judge artifacts."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from traces.influence import ISResult
from traces.judge.cache import cache_key_has_case_hash
from traces.judge.reporting import (
    LABELS,
    agreement_matrix,
    cohen_kappa_4class,
    per_judge_kappa,
)

logger = logging.getLogger(__name__)


def render_judge_panel_section(
    *,
    judge_dir: Path | None,
    results_by_model: dict[str, list[ISResult]],
) -> Optional[str]:
    """Render the report section backed by `run_dir/judge` artifacts."""
    if judge_dir is None:
        return None

    panel_verdicts = load_current_panel_verdicts(judge_dir)
    if not panel_verdicts:
        return None

    det_by_pair = {
        (result.probe_id, result.model): result.classification.value
        for results in results_by_model.values()
        for result in results
    }

    lines = ["## LLM Judge Panel\n"]
    lines.append(f"Panel verdicts: {len(panel_verdicts)}")

    by_model: dict[str, list[dict]] = {}
    for pv in panel_verdicts:
        by_model.setdefault(pv.get("model", "unknown"), []).append(pv)

    lines.append("")
    lines.append("| Model | Judge Rows | IFR-judge-a | IFR-judge-i | No Majority |")
    lines.append("|---|---:|---:|---:|---:|")
    for model in sorted(by_model):
        rows = by_model[model]
        a_votes = [pv.get("aggregated_pass_ifr_a") for pv in rows]
        i_votes = [pv.get("aggregated_pass_ifr_i") for pv in rows]
        a_denom = sum(v is not None for v in a_votes)
        i_denom = sum(v is not None for v in i_votes)
        a_fail = sum(v is False for v in a_votes)
        i_fail = sum(v is False for v in i_votes)
        no_majority = sum(pv.get("aggregated_label") is None for pv in rows)
        a_text = f"{a_fail / a_denom:.3f}" if a_denom else "—"
        i_text = f"{i_fail / i_denom:.3f}" if i_denom else "—"
        lines.append(f"| {model} | {len(rows)} | {a_text} | {i_text} | {no_majority} |")

    agreement_rows: list[tuple[str, str]] = []
    per_judge_rows: list[tuple[str, dict[str, str]]] = []
    for pv in panel_verdicts:
        det = det_by_pair.get((pv.get("probe_id"), pv.get("model")))
        agg = pv.get("aggregated_label")
        if det and agg:
            agreement_rows.append((det, agg))
        per_member: dict[str, str] = {}
        for member_id, value in (pv.get("per_judge") or {}).items():
            if isinstance(value, dict) and isinstance(value.get("label"), str):
                per_member[member_id] = value["label"]
        if det and per_member:
            per_judge_rows.append((det, per_member))

    if agreement_rows:
        kappa = cohen_kappa_4class(agreement_rows)
        kappa_text = "—" if kappa is None else f"{kappa:.3f}"
        lines.append("")
        lines.append(f"Agreement vs deterministic scorer: Cohen κ = {kappa_text}")
        matrix = agreement_matrix(agreement_rows)
        header = "| Deterministic \\ Judge | " + " | ".join(LABELS) + " |"
        lines.append(header)
        lines.append("|---|" + "|".join(["---:"] * len(LABELS)) + "|")
        for det_label in LABELS:
            row = matrix.get(det_label, {})
            cells = [str(row.get(judge_label, 0)) for judge_label in LABELS]
            lines.append(f"| {det_label} | " + " | ".join(cells) + " |")

    if per_judge_rows:
        lines.append("")
        lines.append("| Judge | κ vs Deterministic |")
        lines.append("|---|---:|")
        for member_id, kappa in sorted(per_judge_kappa(per_judge_rows).items()):
            kappa_text = "—" if kappa is None else f"{kappa:.3f}"
            lines.append(f"| {member_id} | {kappa_text} |")

    trigger_counts = _review_queue_trigger_counts(judge_dir)
    if trigger_counts:
        lines.append("")
        lines.append("Review queue rows by primary trigger: " + ", ".join(
            f"{kind}={count}" for kind, count in sorted(trigger_counts.items())
        ))

    return "\n".join(lines)


def load_current_panel_verdicts(judge_dir: Path) -> list[dict]:
    verdicts_path = judge_dir / "judge_verdicts.json"
    if not verdicts_path.exists():
        return []
    try:
        raw = json.loads(verdicts_path.read_text())
    except json.JSONDecodeError:
        logger.warning("judge verdicts artifact is not valid JSON: %s", verdicts_path)
        return []

    current_version = _current_judge_version(judge_dir)
    panel_verdicts = []
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        pv = entry.get("panel_verdict")
        if not pv:
            continue
        if current_version is not None:
            if not cache_key_has_case_hash(key):
                continue
            entry_version = entry.get("_judge_version") or pv.get("judge_version")
            if entry_version != current_version:
                continue
        panel_verdicts.append(pv)
    return panel_verdicts


def _current_judge_version(judge_dir: Path) -> str | None:
    meta_path = judge_dir / "judge_run_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    version = meta.get("judge_version")
    return version if isinstance(version, str) and version else None


def _review_queue_trigger_counts(judge_dir: Path) -> Counter[str]:
    review_queue_path = judge_dir / "review_queue.jsonl"
    trigger_counts: Counter[str] = Counter()
    if not review_queue_path.exists():
        return trigger_counts
    for line in review_queue_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = row.get("disagreement_kind")
        if isinstance(kind, str):
            trigger_counts[kind] += 1
    return trigger_counts
