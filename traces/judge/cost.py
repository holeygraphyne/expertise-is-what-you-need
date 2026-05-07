"""Cost estimation + budget enforcement for `traces judge is`.

Crude flat-rate model: each judge call has a config-defined cost.
Total = sum_over_responses(sum_over_panel_members(rate)).

The gate is intentionally simple — fine-grained provisioning
(per-token cost, response-length-aware estimates) is the user's job
or a v1.1 feature. The gate exists to avoid surprise $200 invoices,
not to be a perfect oracle.
"""
from __future__ import annotations


class CostBudgetError(RuntimeError):
    """Raised when the estimated cost exceeds --max-cost."""


def estimate_cost(
    *,
    n_responses: int,
    panel_member_ids: list[str],
    cost_per_call_usd: dict[str, float],
    default_per_call_usd: float = 0.05,
) -> float:
    """Return the total estimated USD cost for a full panel run."""
    per_response = sum(
        cost_per_call_usd.get(mid, default_per_call_usd)
        for mid in panel_member_ids
    )
    return n_responses * per_response


def enforce_budget(*, estimated_cost: float, max_cost: float) -> None:
    """Raise CostBudgetError if estimated_cost > max_cost.

    Convention: max_cost == 0 disables the gate (useful for CI or
    one-off manual overrides). Negative max_cost is also treated as
    disabled to avoid a footgun.
    """
    if max_cost <= 0:
        return
    if estimated_cost > max_cost:
        raise CostBudgetError(
            f"Estimated cost ${estimated_cost:.2f} exceeds --max-cost "
            f"${max_cost:.2f}. Lower scope (--sample N), raise --max-cost, "
            f"or override audit.cost_per_call_usd estimates."
        )
