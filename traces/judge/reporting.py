"""Pure data-shaping fragments consumed by traces/reporting/influence.py.

Functions here produce dicts/lists; matplotlib lives in the reporting
module so this file stays trivially testable and side-effect-free.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from traces.judge.cache import cache_key_has_case_hash

LABELS = ["REFUSED_RECOGNIZED", "REFUSED_UNRECOGNIZED",
          "ENGAGED_RECOGNIZED", "ENGAGED_UNRECOGNIZED"]


def agreement_matrix(
    rows: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    """Build a {deterministic_label: {judge_label: count}} matrix."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for det_label, judge_label in rows:
        matrix[det_label][judge_label] += 1
    # Materialize defaultdicts as plain dicts for clean serialization.
    return {k: dict(v) for k, v in matrix.items()}


def cohen_kappa_4class(rows: list[tuple[str, str]]) -> Optional[float]:
    """Cohen's κ over (rater_a_label, rater_b_label) pairs.

    Returns None if fewer than 2 rows. Any label values are accepted —
    the 4-class restriction is documentation, not enforcement, since
    we want κ to work on label spaces with arbitrary values.
    """
    n = len(rows)
    if n < 2:
        return None
    agree = sum(1 for a, b in rows if a == b)
    p_o = agree / n

    a_counts = Counter(a for a, _ in rows)
    b_counts = Counter(b for _, b in rows)
    p_e = sum(
        (a_counts[k] / n) * (b_counts[k] / n)
        for k in set(a_counts) | set(b_counts)
    )
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def per_judge_kappa(
    rows: list[tuple[str, dict[str, str]]],
) -> dict[str, Optional[float]]:
    """For each panel member, return κ vs the deterministic label."""
    members: set[str] = set()
    for _, per_member in rows:
        members.update(per_member.keys())
    out: dict[str, Optional[float]] = {}
    for m in members:
        pairs = [
            (det, per_member[m])
            for det, per_member in rows
            if m in per_member
        ]
        out[m] = cohen_kappa_4class(pairs)
    return out


import json as _json
from pathlib import Path as _Path


def enrich_report_with_judge_data(*, judge_dir: _Path) -> dict:
    """Read judge_verdicts.json and return a JSON-serializable summary
    block to splice into the IS report.

    Returns an empty dict when no judge artifact exists; the caller
    treats that as "skip the judge section" so the report works
    with or without judging.

    Entries missing ``panel_verdict`` (e.g. error rows) are skipped cleanly.
    """
    verdicts_path = judge_dir / "judge_verdicts.json"
    if not verdicts_path.exists():
        return {}
    raw = _json.loads(verdicts_path.read_text())
    current_version = _current_judge_version(judge_dir)
    panel_verdicts = []
    for key, entry in raw.items():
        pv = entry.get("panel_verdict")
        if pv is None:
            continue
        if current_version is not None:
            if not cache_key_has_case_hash(key):
                continue
            entry_version = entry.get("_judge_version") or pv.get("judge_version")
            if entry_version != current_version:
                continue
        panel_verdicts.append(pv)

    label_counts = Counter(
        pv.get("aggregated_label") for pv in panel_verdicts
    )
    return {
        "panel_verdict_count": len(panel_verdicts),
        "ifr_judge_label_distribution": dict(label_counts),
        "source_identified_count": sum(
            1 for pv in panel_verdicts
            if pv.get("aggregated_source_identified") is True
        ),
        "sanewashing_count": sum(
            1 for pv in panel_verdicts
            if pv.get("aggregated_sanewashing") is True
        ),
    }


def _current_judge_version(judge_dir: _Path) -> str | None:
    meta_path = judge_dir / "judge_run_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = _json.loads(meta_path.read_text())
    except _json.JSONDecodeError:
        return None
    version = meta.get("judge_version")
    return version if isinstance(version, str) and version else None
