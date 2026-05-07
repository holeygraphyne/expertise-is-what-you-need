"""Tests for traces.judge.promotion."""
import csv as _csv
import json
from io import StringIO

from traces.judge.promotion import promote_labels, export_review_queue_csv


def _row(probe_id, model, sha, **overrides):
    base = {
        "probe_id": probe_id,
        "model": model,
        "response_sha256": sha,
        "deterministic_label": "REFUSED_UNRECOGNIZED",
        "panel_verdict": {
            "probe_id": probe_id, "model": model, "response_sha256": sha,
            "per_judge": {},
            "aggregated_label": "REFUSED_RECOGNIZED",
            "aggregated_pass_ifr_a": True,
            "aggregated_pass_ifr_i": True,
            "aggregated_refused": True,
            "aggregated_recognized": True,
            "aggregated_delivered_operational_content": False,
            "aggregated_source_identified": False,
            "aggregated_sanewashing": False,
            "intra_panel_kappa": 1.0,
            "refusal_count": 0,
            "judge_version": "v",
        },
        "disagreement_kind": "panel_vs_deterministic",
        "all_triggers": ["panel_vs_deterministic"],
        "consistency_violation": None,
        "response_excerpt_first_2000": "abc",
        "human_label": None,
        "human_notes": None,
    }
    base.update(overrides)
    return base


class TestPromoteLabels:
    def test_promotes_only_labeled_rows(self, tmp_path):
        review_queue = tmp_path / "review_queue.jsonl"
        rows = [
            _row("p1", "m1", "a"),
            _row("p2", "m2", "b", human_label="REFUSED_RECOGNIZED",
                 human_notes="checked"),
        ]
        review_queue.write_text("\n".join(json.dumps(r) for r in rows))

        corpus_artifact = tmp_path / "labeled_disagreements.jsonl"
        promoted = promote_labels(
            review_queue_path=review_queue,
            corpus_artifact_path=corpus_artifact,
        )
        assert promoted == 1
        out_lines = corpus_artifact.read_text().strip().splitlines()
        assert len(out_lines) == 1
        promoted_row = json.loads(out_lines[0])
        assert promoted_row["probe_id"] == "p2"

    def test_dedupes_by_keys(self, tmp_path):
        review_queue = tmp_path / "review_queue.jsonl"
        review_queue.write_text(
            json.dumps(_row("p", "m", "x", human_label="REFUSED_RECOGNIZED")) + "\n"
        )

        corpus_artifact = tmp_path / "labeled.jsonl"
        # First promotion writes the row.
        promote_labels(review_queue_path=review_queue, corpus_artifact_path=corpus_artifact)
        # Second promotion of the SAME row must not duplicate.
        promote_labels(review_queue_path=review_queue, corpus_artifact_path=corpus_artifact)

        lines = corpus_artifact.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_empty_queue_returns_zero(self, tmp_path):
        review_queue = tmp_path / "review_queue.jsonl"
        review_queue.write_text("")
        corpus_artifact = tmp_path / "labeled.jsonl"
        assert promote_labels(
            review_queue_path=review_queue,
            corpus_artifact_path=corpus_artifact,
        ) == 0


class TestExportReviewQueueCSV:
    def test_emits_header_and_rows(self, tmp_path):
        review_queue = tmp_path / "review_queue.jsonl"
        review_queue.write_text(
            json.dumps(_row("p1", "m1", "a", disagreement_kind="panel_vs_deterministic")) + "\n"
        )
        out_path = tmp_path / "review_queue.csv"
        n = export_review_queue_csv(
            review_queue_path=review_queue, out_path=out_path,
        )
        assert n == 1
        rows = list(_csv.DictReader(StringIO(out_path.read_text())))
        assert rows[0]["probe_id"] == "p1"
        assert rows[0]["disagreement_kind"] == "panel_vs_deterministic"

    def test_empty_queue_writes_header_only(self, tmp_path):
        review_queue = tmp_path / "review_queue.jsonl"
        review_queue.write_text("")
        out_path = tmp_path / "review_queue.csv"
        n = export_review_queue_csv(
            review_queue_path=review_queue, out_path=out_path,
        )
        assert n == 0
        # Header still written.
        assert "probe_id" in out_path.read_text()
