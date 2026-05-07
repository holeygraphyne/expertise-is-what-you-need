"""Tests for traces.judge.cost."""
import pytest

from traces.judge.cost import (
    CostBudgetError,
    estimate_cost,
    enforce_budget,
)


class TestEstimateCost:
    def test_simple_estimate(self):
        # 100 calls per judge × 3 judges × $0.04 each = $12
        cost = estimate_cost(
            n_responses=100,
            panel_member_ids=["a/m1", "b/m2", "c/m3"],
            cost_per_call_usd={"a/m1": 0.04, "b/m2": 0.04, "c/m3": 0.04},
        )
        assert cost == pytest.approx(12.0)

    def test_per_member_rates(self):
        cost = estimate_cost(
            n_responses=50,
            panel_member_ids=["a/m1", "b/m2"],
            cost_per_call_usd={"a/m1": 0.05, "b/m2": 0.03},
        )
        assert cost == pytest.approx(50 * 0.05 + 50 * 0.03)

    def test_missing_rate_uses_default(self):
        cost = estimate_cost(
            n_responses=10,
            panel_member_ids=["a/m1", "b/m2"],
            cost_per_call_usd={"a/m1": 0.05},
            default_per_call_usd=0.02,
        )
        assert cost == pytest.approx(10 * 0.05 + 10 * 0.02)


class TestEnforceBudget:
    def test_under_budget_returns_silently(self):
        enforce_budget(estimated_cost=5.0, max_cost=10.0)

    def test_over_budget_raises(self):
        with pytest.raises(CostBudgetError, match="exceeds"):
            enforce_budget(estimated_cost=20.0, max_cost=10.0)

    def test_max_cost_zero_disables_gate(self):
        # Convention: --max-cost 0 disables the gate entirely.
        enforce_budget(estimated_cost=999_999.0, max_cost=0.0)
