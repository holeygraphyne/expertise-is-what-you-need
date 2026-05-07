import pytest

from traces.pipeline.dispatcher import ModelHealth, TripThresholds


def test_trip_thresholds_default_values():
    t = TripThresholds()
    assert t.consecutive_failures == 5
    assert t.rate_window_size == 10
    assert t.rate_min_samples == 5
    assert t.rate_threshold == 0.7
    assert t.wallclock_no_success_seconds == 120.0


def test_trip_thresholds_rejects_zero_consecutive():
    with pytest.raises(ValueError, match="consecutive_failures"):
        TripThresholds(consecutive_failures=0)


def test_trip_thresholds_rejects_min_samples_above_window():
    with pytest.raises(ValueError, match="rate_min_samples"):
        TripThresholds(rate_window_size=5, rate_min_samples=10)


def test_trip_thresholds_rejects_zero_window():
    with pytest.raises(ValueError, match="rate_window_size"):
        TripThresholds(rate_window_size=0)


def test_trip_thresholds_rejects_rate_threshold_zero():
    with pytest.raises(ValueError, match="rate_threshold"):
        TripThresholds(rate_threshold=0.0)


def test_trip_thresholds_rejects_rate_threshold_above_one():
    with pytest.raises(ValueError, match="rate_threshold"):
        TripThresholds(rate_threshold=1.5)


def test_trip_thresholds_rejects_zero_wallclock():
    with pytest.raises(ValueError, match="wallclock"):
        TripThresholds(wallclock_no_success_seconds=0.0)


def test_trip_thresholds_accepts_boundary_values():
    """Pin the inclusive boundaries: rate_threshold=1.0 and
    rate_min_samples == rate_window_size are valid configs."""
    t = TripThresholds(
        consecutive_failures=1,
        rate_window_size=1,
        rate_min_samples=1,
        rate_threshold=1.0,
        wallclock_no_success_seconds=0.001,
    )
    assert t.rate_threshold == 1.0
    assert t.rate_min_samples == 1
    assert t.consecutive_failures == 1


def test_model_health_initial_state():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds())
    assert h.model_id == "m1"
    assert h.tripped is False
    assert h.attempts == 0
    assert h.successes == 0
    assert h.consecutive_failures == 0
    assert h.last_success_at == 0.0
    assert h.last_attempt_at == 0.0
    assert h.first_attempt_at == 0.0
    assert h.trip_reason is None
    assert h.trip_at is None
    assert len(h.recent_outcomes) == 0


def test_mark_attempt_start_sets_first_and_last():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds())
    h.mark_attempt_start(now=100.0)
    assert h.first_attempt_at == 100.0
    assert h.last_attempt_at == 100.0


def test_mark_attempt_start_does_not_overwrite_first():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds())
    h.mark_attempt_start(now=100.0)
    h.mark_attempt_start(now=200.0)
    assert h.first_attempt_at == 100.0
    assert h.last_attempt_at == 200.0


def test_should_attempt_initial_returns_true():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds())
    assert h.should_attempt(now=0.0) is True


def _failures(h: ModelHealth, n: int, *, now: float = 0.0) -> None:
    """Helper: record n consecutive failures."""
    for i in range(n):
        h.mark_attempt_start(now=now + i)
        h.record(success=False, now=now + i)


def test_consecutive_trip_at_threshold():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds(consecutive_failures=5))
    _failures(h, 4)
    assert h.tripped is False
    h.mark_attempt_start(now=4)
    h.record(success=False, now=4)
    assert h.tripped is True
    assert h.trip_reason == "consecutive"


def test_consecutive_resets_on_success():
    # Disable rate path by requiring 100 samples; only consecutive matters here.
    h = ModelHealth(model_id="m1", thresholds=TripThresholds(
        consecutive_failures=5,
        rate_window_size=100, rate_min_samples=100,
    ))
    _failures(h, 4)
    h.mark_attempt_start(now=10)
    h.record(success=True, now=10)
    _failures(h, 4, now=20)
    assert h.tripped is False


def test_rate_trip_needs_min_samples():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds(
        consecutive_failures=100,  # disable consecutive path
        rate_window_size=10, rate_min_samples=5, rate_threshold=0.7,
    ))
    _failures(h, 3)
    assert h.tripped is False  # below min_samples


def test_rate_trip_at_threshold():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds(
        consecutive_failures=100,
        rate_window_size=10, rate_min_samples=5, rate_threshold=0.7,
    ))
    # 5 attempts: 1 success then 4 failures = 80% failure rate
    h.mark_attempt_start(now=0); h.record(success=True, now=0)
    _failures(h, 4, now=1)
    assert h.tripped is True
    assert h.trip_reason == "rate"


def test_wallclock_trip():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds(
        consecutive_failures=100,
        rate_window_size=100, rate_min_samples=100,  # disable rate
        wallclock_no_success_seconds=300.0,
    ))
    h.mark_attempt_start(now=0.0)
    h.record(success=False, now=0.0)
    assert h.tripped is False
    assert h.should_attempt(now=301.0) is False
    assert h.trip_reason == "wallclock"


def test_wallclock_does_not_trip_fresh_model():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds())
    assert h.should_attempt(now=10_000.0) is True


def test_immediate_trip_auth():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds())
    h.mark_attempt_start(now=0)
    h.record(success=False, now=0, immediate_trip_reason="auth")
    assert h.tripped is True
    assert h.trip_reason == "auth"


def test_immediate_trip_entitlement():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds())
    h.mark_attempt_start(now=0)
    h.record(success=False, now=0, immediate_trip_reason="entitlement")
    assert h.tripped is True
    assert h.trip_reason == "entitlement"


def test_classify_immediate_trip_handles_both_nim_entitlement_formats():
    """NIM uses two observed 404 wordings for entitlement errors; the
    classifier must catch both."""
    from traces.pipeline.dispatcher import ModelDispatcher
    from traces.pipeline.provider_client import ProviderHTTPError

    older = ProviderHTTPError(404, "Function 'abc': Not found for account 'wzx'")
    newer = ProviderHTTPError(404, "Function id 'abc' version 'null': Specified function in account 'wzx' is not found")
    other_404 = ProviderHTTPError(404, "Some completely different not-found message")

    assert ModelDispatcher._classify_immediate_trip(older) == "entitlement"
    assert ModelDispatcher._classify_immediate_trip(newer) == "entitlement"
    assert ModelDispatcher._classify_immediate_trip(other_404) is None


def test_classify_immediate_trip_handles_auth():
    from traces.pipeline.dispatcher import ModelDispatcher
    from traces.pipeline.provider_client import ProviderHTTPError

    assert ModelDispatcher._classify_immediate_trip(ProviderHTTPError(401, "")) == "auth"
    assert ModelDispatcher._classify_immediate_trip(ProviderHTTPError(403, "")) == "auth"
    assert ModelDispatcher._classify_immediate_trip(ProviderHTTPError(500, "")) is None
    assert ModelDispatcher._classify_immediate_trip(RuntimeError("boom")) is None


def test_successful_call_after_trip_does_not_untrip():
    h = ModelHealth(model_id="m1", thresholds=TripThresholds(consecutive_failures=2))
    _failures(h, 2)
    assert h.tripped is True
    h.mark_attempt_start(now=10)
    h.record(success=True, now=10)
    assert h.tripped is True  # one-way semantics


def test_concurrent_record_thread_safe():
    """100 record() calls across 8 threads; counters end up consistent."""
    import threading
    h = ModelHealth(model_id="m1", thresholds=TripThresholds(consecutive_failures=10_000))
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        barrier.wait()
        for i in range(n):
            h.mark_attempt_start(now=i * 0.001)
            h.record(success=(i % 2 == 0), now=i * 0.001)

    threads = [threading.Thread(target=worker, args=(100,)) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert h.attempts == 800
    assert h.successes == 400


# ---------------------------------------------------------------------------
# ModelDispatcher — per-model dispatch
# ---------------------------------------------------------------------------

import time
import threading
from typing import Any
from unittest.mock import MagicMock

from traces.pipeline.dispatcher import ModelDispatcher
from traces.pipeline.runner import RawProbeResult
from traces.config import ModelConfig


class _FakeProviderClient:
    """Returns scripted CompletionResponse-shaped objects, optional sleep + raise."""
    def __init__(self):
        self.script: list[Any] = []  # each item: callable(model)->resp OR Exception
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(self, model: str, **kwargs):
        self.calls.append({"model": model, **kwargs})
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if not self.script:
                return _ok_response(model)
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item(model)
        finally:
            with self._lock:
                self.in_flight -= 1


def _ok_response(model: str):
    r = MagicMock()
    r.model = model
    r.content = "hello"
    r.finish_reason = "stop"
    r.prompt_tokens = 1
    r.completion_tokens = 1
    r.latency_ms = 5.0
    return r


def _make_probe(probe_id: str = "P1", paper_id: str = "paperX"):
    p = MagicMock()
    p.probe_id = probe_id
    p.paper_id = paper_id
    p.user_prompt = "u"
    p.system_prompt = ""
    return p


def _solo_dispatcher(model_id: str = "m1", *, max_inflight: int = 2):
    """Dispatcher with one model."""
    client = _FakeProviderClient()
    cfg = ModelConfig(
        id=model_id, provider="fake",
        provider_model_id=model_id,
        max_inflight=max_inflight,
    )
    return ModelDispatcher(
        model_configs={model_id: cfg},
        provider_clients={"fake": client},
    ), client


def test_dispatcher_rejects_invalid_model_max_inflight():
    cfg = ModelConfig.model_construct(
        id="m1",
        provider="fake",
        provider_model_id="m1",
        max_inflight=0,
    )
    with pytest.raises(ValueError, match="max_inflight"):
        ModelDispatcher(
            model_configs={"m1": cfg},
            provider_clients={"fake": _FakeProviderClient()},
        )


def test_dispatcher_uses_per_model_max_inflight():
    dispatcher = ModelDispatcher(
        model_configs={
            "slow": ModelConfig(
                id="slow",
                provider="fake",
                provider_model_id="slow",
                max_inflight=1,
            ),
            "fast": ModelConfig(
                id="fast",
                provider="fake",
                provider_model_id="fast",
                max_inflight=3,
            ),
        },
        provider_clients={"fake": _FakeProviderClient()},
    )

    assert dispatcher._max_inflight_by_model["slow"] == 1
    assert dispatcher._max_inflight_by_model["fast"] == 3


def test_dispatcher_uses_model_specific_max_tokens():
    client = _FakeProviderClient()
    dispatcher = ModelDispatcher(
        model_configs={
            "m1": ModelConfig(
                id="m1",
                provider="fake",
                provider_model_id="provider-m1",
                max_tokens=1234,
            )
        },
        provider_clients={"fake": client},
    )

    result = dispatcher.execute(_make_probe(), model_id="m1")

    assert result.error is None
    assert client.calls[0]["max_tokens"] == 1234


def test_dispatcher_uses_default_model_max_tokens_when_omitted():
    client = _FakeProviderClient()
    dispatcher = ModelDispatcher(
        model_configs={
            "m1": ModelConfig(
                id="m1",
                provider="fake",
                provider_model_id="provider-m1",
            )
        },
        provider_clients={"fake": client},
    )

    dispatcher.execute(_make_probe(), model_id="m1")

    assert client.calls[0]["max_tokens"] == 4096


def test_solo_model_success():
    d, _ = _solo_dispatcher()
    r = d.execute(_make_probe(), model_id="m1")
    assert isinstance(r, RawProbeResult)
    assert r.model == "m1"
    assert r.error is None


def test_solo_model_failure_recorded_on_health():
    d, client = _solo_dispatcher()
    client.script = [RuntimeError("boom")]
    r = d.execute(_make_probe(), model_id="m1")
    assert r.error is not None
    assert "boom" in r.error
    # Check that ModelHealth recorded the failure
    health = d._health["m1"]  # internal access for test
    assert health.attempts == 1
    assert health.successes == 0
    assert health.consecutive_failures == 1


def test_tripped_model_returns_model_tripped_error():
    """Once a model is tripped, execute() returns model_tripped immediately."""
    d, client = _solo_dispatcher()
    thresholds = TripThresholds(consecutive_failures=1)
    d._health["m1"] = ModelHealth(model_id="m1", thresholds=thresholds)
    d._health["m1"].record(success=False, now=0.0, immediate_trip_reason="auth")
    assert d._health["m1"].tripped is True

    r = d.execute(_make_probe(), model_id="m1")
    assert r.error == "model_tripped"


def test_bulkhead_caps_concurrent_calls():
    d, client = _solo_dispatcher(max_inflight=2)
    # Make calls take ~50ms each so we observe concurrency
    def slow(model: str):
        time.sleep(0.05)
        return _ok_response(model)
    client.script = [slow] * 20

    threads = [
        threading.Thread(target=lambda: d.execute(_make_probe(), model_id="m1"))
        for _ in range(10)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    assert client.max_in_flight <= 2  # the bulkhead worked


def test_in_flight_call_finishes_after_trip():
    """A slow call mid-flight when the model trips still records its
    outcome on completion (we do not cancel in-flight calls)."""
    d, client = _solo_dispatcher()
    # Replace health with one that trips on first failure
    d._health["m1"] = ModelHealth(
        model_id="m1",
        thresholds=TripThresholds(consecutive_failures=1),
    )

    # First call hangs for 100ms, then succeeds
    def slow_ok(model: str):
        time.sleep(0.1)
        return _ok_response(model)
    client.script = [slow_ok]

    # Trip the model from another thread while the call is mid-flight
    def tripper():
        time.sleep(0.02)
        d._health["m1"].record(success=False, now=time.monotonic(),
                              immediate_trip_reason="auth")
    threading.Thread(target=tripper).start()

    r = d.execute(_make_probe(probe_id="P-slow"), model_id="m1")
    # The call finishes successfully even though m1 tripped during it
    assert r.error is None
    assert r.model == "m1"
    # Subsequent attempt sees m1 tripped → model_tripped error
    r2 = d.execute(_make_probe(probe_id="P-after"), model_id="m1")
    assert r2.error == "model_tripped"


# ---------------------------------------------------------------------------
# health_snapshot + structured trip log emission
# ---------------------------------------------------------------------------

def test_health_snapshot_initial():
    d, _ = _solo_dispatcher()
    snap = d.health_snapshot()
    assert "m1" in snap
    entry = snap["m1"]
    assert entry["tripped"] is False
    assert entry["attempts"] == 0
    assert entry["trip_reason"] is None


def test_health_snapshot_after_trip():
    d, client = _solo_dispatcher()
    d._health["m1"] = ModelHealth(
        model_id="m1",
        thresholds=TripThresholds(consecutive_failures=1),
    )
    client.script = [RuntimeError("a-down")]
    d.execute(_make_probe(), model_id="m1")
    snap = d.health_snapshot()
    assert snap["m1"]["tripped"] is True
    assert snap["m1"]["trip_reason"] == "consecutive"
    assert snap["m1"]["attempts"] == 1
    assert snap["m1"]["successes"] == 0


def test_trip_emits_warning_log_line(caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="traces.pipeline.dispatcher")
    d, client = _solo_dispatcher()
    d._health["m1"] = ModelHealth(
        model_id="m1",
        thresholds=TripThresholds(consecutive_failures=1),
    )
    client.script = [RuntimeError("a-down")]
    d.execute(_make_probe(), model_id="m1")
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Model m1 tripped" in m for m in msgs), msgs
    assert any("reason=consecutive" in m for m in msgs), msgs
