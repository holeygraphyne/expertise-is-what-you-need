"""Unit tests for the findings markdown renderer."""
from traces.calibration.domain import OptimizationFinding
from traces.calibration.findings_renderer import (
    aggregate_findings_by_target,
    render_findings_md,
)


def _finding(**overrides) -> OptimizationFinding:
    base = dict(
        target_kind="lexicon",
        target_locator="lexicon.yaml:domain_violation_lemmas",
        rule_gap="recognition_false_negative",
        evidence_case_keys=["k1", "k2"],
        hypothesis="The lemma list is missing 'fringe'.",
        suggestion="Add 'fringe' to domain_violation_lemmas.",
    )
    base.update(overrides)
    return OptimizationFinding(**base)


def test_render_empty_findings():
    md = render_findings_md([])
    assert "# Optimization Findings" in md
    assert "None" in md or "no findings" in md.lower()


def test_render_includes_required_sections():
    md = render_findings_md([_finding()])
    assert "lexicon.yaml:domain_violation_lemmas" in md
    assert "recognition_false_negative" in md
    assert "fringe" in md.lower()


def test_aggregate_groups_by_target():
    fs = [
        _finding(target_locator="lexicon.yaml:hedge_tokens", evidence_case_keys=["k1"]),
        _finding(target_locator="lexicon.yaml:hedge_tokens", evidence_case_keys=["k2"]),
        _finding(target_locator="lexicon.yaml:retraction_lemmas", evidence_case_keys=["k3"]),
    ]
    grouped = aggregate_findings_by_target(fs)
    assert len(grouped) == 2
    hedge_group = next(g for g in grouped if "hedge" in g["target_locator"])
    assert hedge_group["count"] == 2
    assert set(hedge_group["evidence_case_keys"]) == {"k1", "k2"}


def test_render_orders_by_evidence_count():
    fs = [
        _finding(target_locator="lexicon.yaml:hedge_tokens", evidence_case_keys=["k1"]),
        _finding(target_locator="lexicon.yaml:retraction_lemmas",
                 evidence_case_keys=["k2", "k3", "k4"]),
    ]
    md = render_findings_md(fs)
    # Larger-evidence finding should appear first.
    assert md.index("retraction_lemmas") < md.index("hedge_tokens")
