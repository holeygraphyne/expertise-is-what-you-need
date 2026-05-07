"""Promote human-labeled review-queue rows to a committed corpus artifact.

The review queue lives under a (gitignored-by-default) run directory
and grows organically as humans label disagreements. Once labeled,
rows promote to traces/judge/labeled_disagreements.jsonl — a
versioned corpus artifact that:

  - Feeds the next scorer iteration as test fixtures (rows where
    deterministic was wrong).
  - Drives rubric updates (rows where the panel was confidently
    wrong).
  - Forms an organic ground-truth corpus over time.

Promotion is manual (`traces score promote-labels --run-id NAME`)
because it commits source-of-truth data; it should never be a
side-effect of running the scorer.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _key(row: dict) -> tuple[str, str, str]:
    return (row["probe_id"], row["model"], row["response_sha256"])


def promote_labels(
    *,
    review_queue_path: Path,
    corpus_artifact_path: Path,
) -> int:
    """Append human-labeled rows from the review queue to the corpus
    artifact, deduping by (probe_id, model, response_sha256). Returns
    the number of rows actually appended.
    """
    if not review_queue_path.exists():
        return 0
    rows = []
    for line in review_queue_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    labeled = [r for r in rows if r.get("human_label")]
    if not labeled:
        return 0

    existing_keys: set[tuple[str, str, str]] = set()
    if corpus_artifact_path.exists():
        for line in corpus_artifact_path.read_text().splitlines():
            line = line.strip()
            if line:
                existing_keys.add(_key(json.loads(line)))

    corpus_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with corpus_artifact_path.open("a") as fh:
        for r in labeled:
            if _key(r) in existing_keys:
                continue
            fh.write(json.dumps(r) + "\n")
            existing_keys.add(_key(r))
            appended += 1
    return appended


CSV_FIELDS = [
    "probe_id",
    "model",
    "response_sha256",
    "deterministic_label",
    "aggregated_label",
    "disagreement_kind",
    "all_triggers",
    "consistency_violation",
    "refused",
    "recognized",
    "delivered_operational_content",
    "source_identified",
    "sanewashing",
    "intra_panel_kappa",
    "human_label",
    "human_notes",
    "response_excerpt_first_2000",
]


def export_review_queue_csv(
    *,
    review_queue_path: Path,
    out_path: Path,
) -> int:
    """Export review_queue.jsonl as CSV with a stable column ordering.

    Returns the number of rows written (excluding the header).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        if not review_queue_path.exists():
            return 0
        for line in review_queue_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pv = row.get("panel_verdict") or {}
            writer.writerow({
                "probe_id": row.get("probe_id", ""),
                "model": row.get("model", ""),
                "response_sha256": row.get("response_sha256", ""),
                "deterministic_label": row.get("deterministic_label", ""),
                "aggregated_label": pv.get("aggregated_label", ""),
                "disagreement_kind": row.get("disagreement_kind", ""),
                "all_triggers": ",".join(row.get("all_triggers") or []),
                "consistency_violation": row.get("consistency_violation") or "",
                "refused": pv.get("aggregated_refused", ""),
                "recognized": pv.get("aggregated_recognized", ""),
                "delivered_operational_content": pv.get("aggregated_delivered_operational_content", ""),
                "source_identified": pv.get("aggregated_source_identified", ""),
                "sanewashing": pv.get("aggregated_sanewashing", ""),
                "intra_panel_kappa": pv.get("intra_panel_kappa", "") or "",
                "human_label": row.get("human_label") or "",
                "human_notes": row.get("human_notes") or "",
                "response_excerpt_first_2000": row.get("response_excerpt_first_2000", ""),
            })
            written += 1
    return written
