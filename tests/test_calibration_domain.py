"""Unit tests for traces.calibration.domain (pure data types)."""
import pytest

from traces.calibration.domain import (
    OptimizationFinding,
    PaperMetadata,
    StarredCase,
    StarReason,
)


def test_star_reason_re_exported():
    """StarReason from traces.influence.star_reason is re-exported here."""
    from traces.influence.star_reason import StarReason as InfluenceStarReason
    assert StarReason is InfluenceStarReason


def test_paper_metadata_is_frozen():
    pm = PaperMetadata(
        paper_id="p1",
        domain="pseudoscience",
        central_claim="X cures Y",
        operational_request="Design a follow-up study",
        atlas_unreliability_mode="atlas:Pseudoscience",
        claim_type="cargo_cult_experiment",
    )
    with pytest.raises((AttributeError, TypeError)):
        pm.paper_id = "other"  # frozen dataclass


def test_starred_case_carries_required_fields():
    case = StarredCase(
        cache_key="abc123",
        probe_id="IS-p1",
        model="openai/gpt-4o",
        response_text="...",
        paper_metadata=PaperMetadata(
            paper_id="p1", domain="pseudoscience",
            central_claim="X cures Y", operational_request="Design X",
            atlas_unreliability_mode="atlas:Pseudoscience",
            claim_type="cargo_cult_experiment",
        ),
        scorer_classification="REFUSED_UNRECOGNIZED",
        scorer_pass_ifr_a=True,
        scorer_pass_ifr_i=True,
        star_reason=StarReason.REFUSED_UNREC_BARE_REFUSAL,
    )
    assert case.cache_key == "abc123"
    assert case.star_reason == StarReason.REFUSED_UNREC_BARE_REFUSAL


def test_optimization_finding_target_kinds_are_closed():
    """Pydantic rejects target_kind values outside the closed set."""
    from pydantic import ValidationError
    valid = OptimizationFinding(
        target_kind="lexicon",
        target_locator="lexicon.yaml:domain_violation_lemmas",
        rule_gap="recognition_false_negative",
        evidence_case_keys=["abc123"],
        hypothesis="Missing 'fringe' lemma",
        suggestion="Add 'fringe' to domain_violation_lemmas",
    )
    assert valid.target_kind == "lexicon"
    with pytest.raises(ValidationError):
        OptimizationFinding(
            target_kind="regex_add",  # not in the closed set
            target_locator="x",
            rule_gap="other",
            evidence_case_keys=[],
            hypothesis="x",
            suggestion="x",
        )
