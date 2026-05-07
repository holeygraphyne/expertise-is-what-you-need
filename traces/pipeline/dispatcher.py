"""Per-model circuit breaker and bulkhead routing.

The dispatcher sits between ISRunner and ProviderClient. It owns:
  - per-physical-model `ModelHealth` (consecutive failures, sliding-rate
    window, wall-clock-no-success — see TripThresholds for the knobs)
  - per-physical-model `threading.Semaphore` for bulkhead isolation
  - `execute(probe, model_id)` acquires the bulkhead, calls the provider,
    updates health, and returns a `RawProbeResult`.

Trip is one-way for the run; on `--resume` the next process invocation
starts with fresh ModelHealth (only `_completed` carries forward through
`cp['results']`).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from traces.config import ModelConfig
    from traces.pipeline.provider_client import ProviderClient
    from traces.pipeline.runner import RawProbeResult
    from traces.prompts import ISProbe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripThresholds:
    """Knobs for ModelHealth's three trip signals.

    All fields validated at construction; invalid values raise ValueError.
    """
    consecutive_failures: int = 5
    rate_window_size: int = 10
    rate_min_samples: int = 5
    rate_threshold: float = 0.7
    wallclock_no_success_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.consecutive_failures < 1:
            raise ValueError(
                f"consecutive_failures must be >= 1, got {self.consecutive_failures}"
            )
        if self.rate_window_size < 1:
            raise ValueError(
                f"rate_window_size must be >= 1, got {self.rate_window_size}"
            )
        if not (1 <= self.rate_min_samples <= self.rate_window_size):
            raise ValueError(
                f"rate_min_samples must be in [1, rate_window_size={self.rate_window_size}], "
                f"got {self.rate_min_samples}"
            )
        if not (0.0 < self.rate_threshold <= 1.0):
            raise ValueError(
                f"rate_threshold must be in (0.0, 1.0], got {self.rate_threshold}"
            )
        if self.wallclock_no_success_seconds <= 0:
            raise ValueError(
                f"wallclock_no_success_seconds must be > 0, "
                f"got {self.wallclock_no_success_seconds}"
            )


class ModelHealth:
    """Health and trip state for one physical model.

    Thread-safe: all public methods (mark_attempt_start, record,
    should_attempt) hold self._lock for the entire body. Internal
    helpers (_evaluate_trip) must only be called by code that already
    holds the lock.
    """

    def __init__(self, model_id: str, thresholds: TripThresholds):
        self.model_id = model_id
        self._t = thresholds
        self._lock = threading.Lock()

        self.consecutive_failures: int = 0
        self.recent_outcomes: deque[bool] = deque(maxlen=thresholds.rate_window_size)
        self.last_success_at: float = 0.0
        self.last_attempt_at: float = 0.0
        self.first_attempt_at: float = 0.0
        self._has_attempt: bool = False
        self.tripped: bool = False
        self.trip_reason: Optional[str] = None
        self.trip_at: Optional[float] = None
        self.attempts: int = 0
        self.successes: int = 0

    def mark_attempt_start(self, now: float) -> None:
        """Called by the dispatcher BEFORE a network request goes out.

        Sets first_attempt_at on the very first attempt; always updates
        last_attempt_at. Holding this state lets the wall-clock signal
        detect calls that hang and never reach record().
        """
        with self._lock:
            if not self._has_attempt:
                self.first_attempt_at = now
                self._has_attempt = True
            self.last_attempt_at = now

    def record(
        self,
        success: bool,
        now: float,
        *,
        immediate_trip_reason: Optional[str] = None,
    ) -> None:
        """Called AFTER a network call returns or raises.

        Updates counters and outcome window. If immediate_trip_reason is
        set, trips immediately (used for auth/entitlement, where the
        threshold path would waste budget on a permanent failure).
        Otherwise consults _evaluate_trip(now).

        One-way semantics: a successful call after trip does NOT un-trip.
        """
        with self._lock:
            self.attempts += 1
            self.recent_outcomes.append(success)
            if success:
                self.successes += 1
                self.consecutive_failures = 0
                self.last_success_at = now
            else:
                self.consecutive_failures += 1

            if self.tripped:
                # Already tripped; do not change state. Preserve
                # one-way semantics — successes after trip do not
                # un-trip the model.
                return

            if immediate_trip_reason is not None:
                self.tripped = True
                self.trip_reason = immediate_trip_reason
                self.trip_at = now
                return

            reason = self._evaluate_trip(now)
            if reason is not None:
                self.tripped = True
                self.trip_reason = reason
                self.trip_at = now

    def should_attempt(self, now: float) -> bool:
        """True if the model is not tripped (and not about to trip on
        wall-clock). Re-evaluates wall-clock so a silent-but-stuck model
        trips at probe time even when no record() has fired.
        """
        with self._lock:
            if self.tripped:
                return False
            reason = self._evaluate_trip(now)
            if reason is not None:
                self.tripped = True
                self.trip_reason = reason
                self.trip_at = now
                return False
            return True

    def _evaluate_trip(self, now: float) -> Optional[str]:
        """Evaluate trip signals against current state. Returns the trip reason
        or None; does NOT mutate trip state — callers handle that. MUST be
        called holding self._lock."""
        if self.consecutive_failures >= self._t.consecutive_failures:
            return "consecutive"
        if (
            len(self.recent_outcomes) >= self._t.rate_min_samples
            and (1.0 - sum(self.recent_outcomes) / len(self.recent_outcomes))
                >= self._t.rate_threshold
        ):
            return "rate"
        # Wall-clock: no success since the last attempt for too long.
        # Two cases: (a) never had a success — measure from first_attempt_at;
        # (b) had a success but not since last_attempt_at — measure from last_success_at.
        # _has_attempt guards both (fresh model → skip).
        if self._has_attempt:
            no_success_yet = self.successes == 0
            stale_success = self.successes > 0 and self.last_success_at < self.last_attempt_at
            if no_success_yet or stale_success:
                origin = self.first_attempt_at if no_success_yet else max(self.last_success_at, self.first_attempt_at)
                if (now - origin) >= self._t.wallclock_no_success_seconds:
                    return "wallclock"
        return None

    def snapshot(self) -> dict:
        """JSON-serializable trip-state snapshot. Observability-only."""
        with self._lock:
            return {
                "tripped": self.tripped,
                "trip_reason": self.trip_reason,
                "trip_at_monotonic": self.trip_at,
                "attempts": self.attempts,
                "successes": self.successes,
            }


class ModelDispatcher:
    """Routes (probe, model_id) calls with health tracking and bulkhead isolation.

    See docs/superpowers/specs/2026-04-29-runner-resilience-design.md
    for full design discussion (note: fallback-chain sections describe
    code that no longer exists as of refactor/remove-slot-fallback-machinery).
    """

    def __init__(
        self,
        model_configs: dict[str, "ModelConfig"],
        provider_clients: dict[str, "ProviderClient"],
        trip_thresholds: "TripThresholds | None" = None,
    ):
        if trip_thresholds is None:
            trip_thresholds = TripThresholds()

        self._provider_clients = provider_clients
        self._model_configs = model_configs
        self._thresholds = trip_thresholds
        self._max_inflight_by_model: dict[str, int] = {}
        for model_id, cfg in model_configs.items():
            max_inflight = getattr(cfg, "max_inflight", 2)
            if max_inflight < 1:
                raise ValueError(
                    f"models[{model_id!r}].max_inflight must be >= 1, got {max_inflight}"
                )
            self._max_inflight_by_model[model_id] = max_inflight

        # Build per-physical-model state from model_configs keys directly.
        self._health: dict[str, ModelHealth] = {
            m: ModelHealth(model_id=m, thresholds=trip_thresholds)
            for m in model_configs
        }
        # Per-physical-model semaphore implementing the bulkhead pattern:
        # caps in-flight calls to one model so a slow/timing-out endpoint
        # can't monopolize the global ThreadPool's worker slots.
        self._semaphores: dict[str, threading.Semaphore] = {
            model_id: threading.Semaphore(self._max_inflight_by_model[model_id])
            for model_id in model_configs
        }

    def health_snapshot(self) -> dict[str, dict]:
        """JSON-serializable trip-state snapshot per model. Observability-only."""
        return {model_id: h.snapshot() for model_id, h in self._health.items()}

    def _log_trip_if_newly_tripped(
        self,
        model_id: str,
        was_tripped: bool,
    ) -> None:
        """Emit a WARNING when a model just tripped. Caller passes the
        was_tripped flag captured BEFORE the call to detect transitions.
        """
        h = self._health[model_id]
        if was_tripped or not h.tripped:
            return
        logger.warning(
            "Model %s tripped: reason=%s after %d attempts (%d successes).",
            model_id, h.trip_reason, h.attempts, h.successes,
        )

    def execute(self, probe: "ISProbe", model_id: str) -> "RawProbeResult":
        """Execute the probe against model_id with bulkhead + health tracking.

        If the model is tripped, returns an error result immediately.
        Otherwise acquires the bulkhead semaphore, runs the call, releases,
        then records the outcome.
        """
        health = self._health[model_id]
        sem = self._semaphores[model_id]

        now = time.monotonic()
        if not health.should_attempt(now):
            from traces.pipeline.runner import RawProbeResult  # avoid cycle
            return RawProbeResult(
                probe_id=probe.probe_id,
                paper_id=probe.paper_id,
                model=model_id,
                response_text="",
                latency_ms=0.0,
                error="model_tripped",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        was_tripped = health.tripped
        sem.acquire()
        try:
            health.mark_attempt_start(now=time.monotonic())
            result, success, immediate_reason = self._invoke(probe, model_id)
        finally:
            sem.release()

        health.record(
            success=success,
            now=time.monotonic(),
            immediate_trip_reason=immediate_reason,
        )
        self._log_trip_if_newly_tripped(model_id, was_tripped)
        return result

    def _invoke(
        self,
        probe: "ISProbe",
        model_id: str,
    ) -> "tuple[RawProbeResult, bool, Optional[str]]":
        """Make the network call and return (result, success, immediate_reason).

        Caller holds the bulkhead semaphore. health.record() is intentionally
        NOT called here — execute() calls it after releasing the semaphore so
        the bulkhead is freed before any record() side-effects.
        """
        from traces.pipeline.runner import RawProbeResult  # avoid cycle

        cfg = self._model_configs[model_id]
        client = self._provider_clients[cfg.provider]

        immediate_reason: Optional[str] = None
        success: bool = False
        response_text: str = ""
        latency_ms: float = 0.0
        prompt_tokens: int = 0
        completion_tokens: int = 0
        finish_reason: Optional[str] = None
        error: Optional[str] = None

        try:
            resp = client.complete(
                model=cfg.provider_model_id,
                user_prompt=probe.user_prompt,
                system_prompt=getattr(probe, "system_prompt", "") or "",
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                seed=cfg.seed,
            )
            success = True
            response_text = resp.content
            latency_ms = resp.latency_ms
            prompt_tokens = resp.prompt_tokens
            completion_tokens = resp.completion_tokens
            finish_reason = resp.finish_reason
        except Exception as e:
            error = str(e)
            immediate_reason = self._classify_immediate_trip(e)

        result = RawProbeResult(
            probe_id=probe.probe_id,
            paper_id=probe.paper_id,
            model=model_id,
            response_text=response_text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            error=error,
        )
        return result, success, immediate_reason

    @staticmethod
    def _classify_immediate_trip(exc: Exception) -> Optional[str]:
        """Map an exception to an immediate-trip reason, if any.

        Auth/entitlement errors are permanent for the key×model pair.
        Everything else goes through the threshold path.

        NVIDIA NIM 404 entitlement responses come in (at least) two
        observed forms — both of which mean "model exists in catalog
        but not entitled for this API key":
          - "Not found for account 'X'"  (older format)
          - "Specified function in account 'X' is not found"  (newer)
        Both contain "account '" plus a "not found" phrase, so we
        match the conjunction case-insensitively rather than pinning a
        specific wording.
        """
        from traces.pipeline.provider_client import ProviderHTTPError

        if isinstance(exc, ProviderHTTPError):
            if exc.status_code in (401, 403):
                return "auth"
            if exc.status_code == 404:
                detail_lower = (exc.detail or "").lower()
                if "account '" in detail_lower and "not found" in detail_lower:
                    return "entitlement"
        return None
