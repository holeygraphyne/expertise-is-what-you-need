"""Tests for judge cache/version policy."""

from tests.helpers import make_paper
from traces.judge.cache_policy import JudgeCachePolicy


class TestJudgeCachePolicy:
    def test_panel_order_does_not_change_judge_version(self):
        a = JudgeCachePolicy.from_inputs(
            rubric_text="rubric",
            payload_template_text="1",
            panel_member_ids=["b", "a"],
            output_schema_version="schema",
            aggregation_version="agg",
            aggregation_policy="policy",
            evidence_version="evidence",
            evidence_policy="evidence-policy",
        )
        b = JudgeCachePolicy.from_inputs(
            rubric_text="rubric",
            payload_template_text="1",
            panel_member_ids=["a", "b"],
            output_schema_version="schema",
            aggregation_version="agg",
            aggregation_policy="policy",
            evidence_version="evidence",
            evidence_policy="evidence-policy",
        )

        assert a.judge_version == b.judge_version

    def test_cache_key_includes_response_and_case_hashes(self):
        policy = JudgeCachePolicy.from_inputs(
            rubric_text="rubric",
            payload_template_text="1",
            panel_member_ids=["a", "b", "c"],
            output_schema_version="schema",
        )
        paper = make_paper("bem_psi_2011")

        key = policy.cache_key_for_row(
            probe_id="IS-bem_psi_2011",
            model="model-a",
            paper=paper,
            response_text="response text",
        )

        parts = key.split("|")
        assert parts[0] == "IS-bem_psi_2011"
        assert parts[1] == "model-a"
        assert len(parts[2]) == 64
        assert len(parts[3]) == 64
        assert parts[4] == policy.judge_version

    def test_current_cache_entries_keep_only_current_version_and_key_shape(self):
        policy = JudgeCachePolicy.from_inputs(
            rubric_text="rubric",
            payload_template_text="1",
            panel_member_ids=["a", "b", "c"],
            output_schema_version="schema",
        )
        current_key = "IS-x|m|" + ("0" * 64) + "|" + ("1" * 64) + f"|{policy.judge_version}"
        loaded = {
            current_key: {"_judge_version": policy.judge_version},
            "old-shape|m|abc|v": {"_judge_version": policy.judge_version},
            "IS-x|m|abc|case|old": {"_judge_version": "old"},
        }

        assert policy.current_cache_entries(loaded) == {
            current_key: {"_judge_version": policy.judge_version},
        }
