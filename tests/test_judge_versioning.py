"""Tests for traces.judge.versioning."""
from traces.judge.versioning import (
    JUDGE_AGGREGATION_VERSION,
    JUDGE_EVIDENCE_VERSION,
    JUDGE_OUTPUT_SCHEMA_VERSION,
    compute_judge_version,
)


class TestComputeJudgeVersion:
    def test_deterministic(self):
        v1 = compute_judge_version(
            rubric_text="r",
            payload_template_text="p",
            panel_member_ids=["a/m1", "b/m2", "c/m3"],
            output_schema_version=JUDGE_OUTPUT_SCHEMA_VERSION,
        )
        v2 = compute_judge_version(
            rubric_text="r",
            payload_template_text="p",
            panel_member_ids=["a/m1", "b/m2", "c/m3"],
            output_schema_version=JUDGE_OUTPUT_SCHEMA_VERSION,
        )
        assert v1 == v2
        assert len(v1) == 12

    def test_panel_order_irrelevant(self):
        v1 = compute_judge_version("r", "p", ["a/m1", "b/m2"], "1")
        v2 = compute_judge_version("r", "p", ["b/m2", "a/m1"], "1")
        assert v1 == v2

    def test_rubric_change_invalidates(self):
        v1 = compute_judge_version("r", "p", ["a/m1"], "1")
        v2 = compute_judge_version("r-changed", "p", ["a/m1"], "1")
        assert v1 != v2

    def test_payload_change_invalidates(self):
        v1 = compute_judge_version("r", "p", ["a/m1"], "1")
        v2 = compute_judge_version("r", "p-changed", ["a/m1"], "1")
        assert v1 != v2

    def test_panel_change_invalidates(self):
        v1 = compute_judge_version("r", "p", ["a/m1"], "1")
        v2 = compute_judge_version("r", "p", ["a/m1", "b/m2"], "1")
        assert v1 != v2

    def test_schema_version_invalidates(self):
        v1 = compute_judge_version("r", "p", ["a/m1"], "1")
        v2 = compute_judge_version("r", "p", ["a/m1"], "2")
        assert v1 != v2

    def test_aggregation_policy_invalidates(self):
        v1 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            aggregation_version=JUDGE_AGGREGATION_VERSION,
            aggregation_policy="native_boolean_majority;sanewashing=derived",
        )
        v2 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            aggregation_version=JUDGE_AGGREGATION_VERSION,
            aggregation_policy="native_boolean_majority;sanewashing=disabled",
        )
        assert v1 != v2

    def test_aggregation_version_invalidates(self):
        v1 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            aggregation_version="1",
            aggregation_policy="native_boolean_majority;sanewashing=derived",
        )
        v2 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            aggregation_version="2",
            aggregation_policy="native_boolean_majority;sanewashing=derived",
        )
        assert v1 != v2

    def test_evidence_policy_invalidates(self):
        v1 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            evidence_version=JUDGE_EVIDENCE_VERSION,
            evidence_policy="verbatim;source_reference=metadata",
        )
        v2 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            evidence_version=JUDGE_EVIDENCE_VERSION,
            evidence_policy="verbatim;source_reference=metadata_or_paper_id_alias",
        )
        assert v1 != v2

    def test_evidence_version_invalidates(self):
        v1 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            evidence_version="1",
            evidence_policy="verbatim;source_reference=metadata_or_paper_id_alias",
        )
        v2 = compute_judge_version(
            "r", "p", ["a/m1"], "1",
            evidence_version="2",
            evidence_policy="verbatim;source_reference=metadata_or_paper_id_alias",
        )
        assert v1 != v2
