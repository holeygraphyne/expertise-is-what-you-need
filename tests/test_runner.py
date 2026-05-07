"""Tests for ISRunner checkpoint behavior."""
from __future__ import annotations

from collections import deque
import io
import json
import logging
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

from traces.config import (
    AtlasConfig,
    ModelConfig,
    PipelineConfig,
    ProviderConfig,
    TracesConfig,
)
from traces.pipeline.runner import (
    ISRunner,
    ProgressConsole,
    RawProbeResult,
    _build_work_items,
    _progress_logging,
    _pop_next_runnable_work,
)
from traces.prompts import ISProbe


def _make_runner() -> ISRunner:
    cfg = TracesConfig(
        atlas=AtlasConfig(ontology_path="x", vocabularies_path="y"),
        pipeline=PipelineConfig(concurrency=1),
        providers={
            "local": ProviderConfig(
                base_url="http://localhost:11434/v1",
                rpm_limit=0,
            ),
        },
        audit={"provider": "local"},
        models=[ModelConfig(id="m1", provider="local", provider_model_id="m1")],
    )
    return ISRunner(cfg)


def _result(probe_id: str, model: str, error: str = "") -> RawProbeResult:
    return RawProbeResult(
        probe_id=probe_id,
        paper_id=probe_id.replace("IS-", ""),
        model=model,
        response_text="" if error else "ok",
        latency_ms=0,
        error=error or None,
    )


def _probe(probe_id: str) -> ISProbe:
    probe = MagicMock(spec=ISProbe)
    probe.probe_id = probe_id
    probe.paper_id = probe_id.replace("IS-", "")
    probe.user_prompt = "u"
    probe.system_prompt = ""
    return probe


def test_load_checkpoint_filters_failed_results(tmp_path: Path):
    """Failed (probe, model) pairs in a stale checkpoint must be retried,
    not skipped. _load_checkpoint should drop them from both _completed
    and _raw_results so the next run reattempts them.
    """
    cp = tmp_path / "checkpoint.json"
    success = _result("IS-a", "m1")
    failure = _result("IS-b", "m1", error="timed out")
    cp.write_text(json.dumps({
        "completed": [["IS-a", "m1"], ["IS-b", "m1"]],
        "results": [asdict(success), asdict(failure)],
    }))

    runner = _make_runner()
    runner._load_checkpoint(str(cp))

    assert runner._completed == {("IS-a", "m1")}
    assert len(runner._raw_results) == 1
    assert runner._raw_results[0].probe_id == "IS-a"
    assert runner._raw_results[0].error is None


def test_build_work_items_is_probe_major():
    probes = [_probe("IS-p1"), _probe("IS-p2")]
    model_ids = ["m1", "m2", "m3"]

    work = _build_work_items(probes, model_ids, completed=set())

    assert [(probe.probe_id, model_id) for probe, model_id in work] == [
        ("IS-p1", "m1"),
        ("IS-p1", "m2"),
        ("IS-p1", "m3"),
        ("IS-p2", "m1"),
        ("IS-p2", "m2"),
        ("IS-p2", "m3"),
    ]


def test_build_work_items_skips_completed_pairs():
    probes = [_probe("IS-p1"), _probe("IS-p2")]
    model_ids = ["m1", "m2"]

    work = _build_work_items(
        probes,
        model_ids,
        completed={("IS-p1", "m2")},
    )

    assert [(probe.probe_id, model_id) for probe, model_id in work] == [
        ("IS-p1", "m1"),
        ("IS-p2", "m1"),
        ("IS-p2", "m2"),
    ]


def test_pop_next_runnable_work_skips_saturated_model():
    pending = deque([
        (_probe("IS-p1"), "slow"),
        (_probe("IS-p1"), "fast"),
    ])

    item = _pop_next_runnable_work(
        pending,
        inflight_by_model={"slow": 2},
        max_inflight_by_model={"slow": 2, "fast": 2},
    )

    assert item is not None
    assert item[0].probe_id == "IS-p1"
    assert item[1] == "fast"
    assert [(probe.probe_id, model_id) for probe, model_id in pending] == [
        ("IS-p1", "slow"),
    ]


def test_pop_next_runnable_work_returns_none_when_all_models_saturated():
    pending = deque([
        (_probe("IS-p1"), "m1"),
        (_probe("IS-p2"), "m2"),
    ])

    item = _pop_next_runnable_work(
        pending,
        inflight_by_model={"m1": 1, "m2": 1},
        max_inflight_by_model={"m1": 1, "m2": 1},
    )

    assert item is None
    assert [(probe.probe_id, model_id) for probe, model_id in pending] == [
        ("IS-p1", "m1"),
        ("IS-p2", "m2"),
    ]


def test_provider_config_default_timeout_is_120():
    cfg = ProviderConfig(base_url="http://localhost:11434/v1")
    assert cfg.timeout == 120


def test_provider_config_accepts_custom_timeout():
    cfg = ProviderConfig(base_url="http://localhost:11434/v1", timeout=600)
    assert cfg.timeout == 600


def test_model_config_seed_defaults_to_none():
    m = ModelConfig(id="m1", provider="p", provider_model_id="m1")
    assert m.seed is None


def test_model_config_accepts_seed():
    m = ModelConfig(id="m1", provider="p", provider_model_id="m1", seed=42)
    assert m.seed == 42


def test_scheduler_respects_per_model_max_inflight():
    pending = deque([
        (_probe("IS-p1"), "slow"),
        (_probe("IS-p1"), "fast"),
        (_probe("IS-p2"), "fast"),
        (_probe("IS-p3"), "fast"),
    ])

    item = _pop_next_runnable_work(
        pending,
        inflight_by_model={"slow": 1},
        max_inflight_by_model={"slow": 1, "fast": 3},
    )

    assert item is not None
    assert item[1] == "fast"


def test_provider_client_omits_seed_when_unset(tmp_path):
    """Default: seed is not in the outgoing payload (keeps the pre-seed
    wire format unchanged for providers that reject unknown fields)."""
    import json
    from traces.pipeline.provider_client import ProviderClient

    cfg = ProviderConfig(base_url="http://localhost:11434/v1")
    client = ProviderClient(cfg)
    captured = {}

    class _FakeResp:
        status_code = 200
        text = '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}'

        def json(self):
            return json.loads(self.text)

    def fake(_url, *, json=None, **_kwargs):
        captured["data"] = json
        return _FakeResp()

    from unittest.mock import patch
    with patch("traces.calibration.llm_client.requests.post", side_effect=fake):
        client.complete(model="m", user_prompt="hi")

    assert "seed" not in captured["data"]


def test_provider_client_includes_seed_when_set():
    import json
    from traces.pipeline.provider_client import ProviderClient

    cfg = ProviderConfig(base_url="http://localhost:11434/v1")
    client = ProviderClient(cfg)
    captured = {}

    class _FakeResp:
        status_code = 200
        text = '{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}'

        def json(self):
            return json.loads(self.text)

    def fake(_url, *, json=None, **_kwargs):
        captured["data"] = json
        return _FakeResp()

    from unittest.mock import patch
    with patch("traces.calibration.llm_client.requests.post", side_effect=fake):
        client.complete(model="m", user_prompt="hi", seed=42)

    assert captured["data"]["seed"] == 42


def test_provider_client_retries_empty_completion_then_succeeds(monkeypatch):
    from traces.pipeline.provider_client import ProviderClient

    cfg = ProviderConfig(base_url="http://localhost:11434/v1", max_retries=3, retry_delay=0.01)
    client = ProviderClient(cfg)
    payloads = [
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}], "usage": {"completion_tokens": 0}},
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 4}},
    ]

    class _FakeResp:
        status_code = 200
        text = "{}"

        def __init__(self, body: dict):
            self._body = body

        def json(self):
            return self._body

    calls: list[int] = []

    def fake(*_args, **_kwargs):
        calls.append(1)
        return _FakeResp(payloads[len(calls) - 1])

    monkeypatch.setattr("traces.calibration.llm_client.requests.post", fake)
    response = client.complete(model="m", user_prompt="hi")

    assert response.content == "ok"
    assert len(calls) == 2


def test_provider_client_raises_empty_completion_after_retry_exhaustion(monkeypatch):
    from traces.pipeline.provider_client import EmptyCompletionError, ProviderClient

    cfg = ProviderConfig(base_url="http://localhost:11434/v1", max_retries=2, retry_delay=0.01)
    client = ProviderClient(cfg)
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}], "usage": {"completion_tokens": 0}}

    class _FakeResp:
        status_code = 200
        text = "{}"

        def json(self):
            return body

    calls: list[int] = []

    def fake(*_args, **_kwargs):
        calls.append(1)
        return _FakeResp()

    monkeypatch.setattr("traces.calibration.llm_client.requests.post", fake)

    try:
        client.complete(model="m", user_prompt="hi")
    except EmptyCompletionError as exc:
        assert "empty_completion" in str(exc)
    else:
        raise AssertionError("Expected EmptyCompletionError")

    assert len(calls) == 2


def test_runner_persists_empty_completion_failure(monkeypatch):
    runner = _make_runner()
    probe = ISProbe(
        probe_id="IS-a",
        paper_id="a",
        doc_id="doc-a",
        domain="pseudoscience",
        system_prompt="system",
        user_prompt="user",
    )

    monkeypatch.setattr(
        runner._provider_clients["local"],
        "complete",
        lambda **kwargs: (_ for _ in ()).throw(Exception("empty_completion: 0-token completion")),
    )

    result = runner._dispatcher.execute(probe, "m1")

    assert result.response_text == ""
    assert result.error == "empty_completion: 0-token completion"


def test_runner_progress_console_keeps_warnings_separate(capsys):
    console = ProgressConsole(total=4, enabled=True)
    console.started("IS-a", "m1")
    console.update(done_count=1, fail_count=0, elapsed=5.0, session_done=1)
    console.warn("warning: IS-a @ m1 failed: empty_completion")

    status_render = console._build_renderable().renderables[0].plain

    console.close()

    err = capsys.readouterr().err
    assert "warning: IS-a @ m1 failed: empty_completion" in err
    assert "latest event: warning: IS-a @ m1 failed: empty_completion" in status_render
    assert "1/4" in status_render
    assert "elapsed 5s" in status_render
    assert "current IS-a @ m1" in status_render
    assert console._build_renderable().renderables[1] is console._progress


def test_progress_logging_restores_root_handlers():
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    console = ProgressConsole(total=1, enabled=True, stream=io.StringIO())

    try:
        with _progress_logging(console.console, enabled=True):
            assert root.handlers
            assert root.handlers != old_handlers
    finally:
        console.close()

    assert root.handlers == old_handlers
    assert root.level == old_level


def test_progress_logging_disabled_does_not_replace_handlers():
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    console = ProgressConsole(total=1, enabled=False, stream=io.StringIO())

    with _progress_logging(console.console, enabled=False):
        assert root.handlers == old_handlers
        assert root.level == old_level

    assert root.handlers == old_handlers
    assert root.level == old_level


def test_progress_console_warn_enabled_prints_once_without_logger_warning(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr("traces.pipeline.runner.logger.warning", lambda msg: calls.append(msg))

    console = ProgressConsole(total=1, enabled=True, stream=io.StringIO())
    try:
        console.warn("hello")
    finally:
        console.close()

    assert calls == []


def test_progress_console_warn_disabled_uses_logger(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr("traces.pipeline.runner.logger.warning", lambda msg: calls.append(msg))

    console = ProgressConsole(total=1, enabled=False, stream=io.StringIO())
    console.warn("hello")

    assert calls == ["hello"]


def test_provider_client_uses_constant_retry_delay_for_non_429(monkeypatch):
    from traces.pipeline.provider_client import ProviderClient

    cfg = ProviderConfig(base_url="http://localhost:11434/v1", max_retries=3, retry_delay=2.5)
    client = ProviderClient(cfg)
    waits: list[float] = []

    monkeypatch.setattr("time.sleep", lambda value: waits.append(value))

    class _FakeResp:
        status_code = 403
        text = "forbidden"

        def json(self):
            raise json.JSONDecodeError("err", "doc", 0)

    monkeypatch.setattr(
        "traces.calibration.llm_client.requests.post",
        lambda *_args, **_kwargs: _FakeResp(),
    )

    try:
        client.complete(model="m", user_prompt="hi")
    except Exception:
        pass

    assert waits == [2.5, 2.5]


def test_raw_probe_result_domain_is_optional():
    r = RawProbeResult(
        probe_id="IS-x",
        paper_id="x",
        model="m1",
        response_text="ok",
        latency_ms=0,
    )
    assert r.domain is None


def test_raw_probe_result_accepts_domain_for_backcompat():
    """Existing raw_results.json files have a string domain — must still
    deserialize."""
    r = RawProbeResult(
        probe_id="IS-x",
        paper_id="x",
        model="m1",
        domain="pseudoscience",
        response_text="ok",
        latency_ms=0,
    )
    assert r.domain == "pseudoscience"


def test_raw_probe_result_round_trip_via_asdict():
    """asdict(...) -> dict -> RawProbeResult(**dict) preserves all fields."""
    r = RawProbeResult(
        probe_id="P1", paper_id="paperX", model="m1",
        response_text="hi", latency_ms=10.0,
    )
    d = asdict(r)
    r2 = RawProbeResult(**d)
    assert r2 == r


def test_runner_caches_one_client_per_provider_and_shares_limiter():
    """Two models on provider_a + one on provider_b → exactly two
    ProviderClients in the cache, and the two provider_a models share
    the same ThreadSafeRpmLimiter (so RPM quotas are per-endpoint, not
    duplicated per-model)."""
    cfg = TracesConfig(
        atlas=AtlasConfig(ontology_path="x", vocabularies_path="y"),
        pipeline=PipelineConfig(concurrency=1),
        providers={
            "provider_a": ProviderConfig(
                base_url="http://localhost:1111/v1", rpm_limit=10,
            ),
            "provider_b": ProviderConfig(
                base_url="http://localhost:2222/v1", rpm_limit=20,
            ),
        },
        audit={"provider": "provider_a"},
        models=[
            ModelConfig(id="m1", provider="provider_a", provider_model_id="m1"),
            ModelConfig(id="m2", provider="provider_a", provider_model_id="m2"),
            ModelConfig(id="m3", provider="provider_b", provider_model_id="m3"),
        ],
    )
    runner = ISRunner(cfg)
    assert set(runner._provider_clients.keys()) == {"provider_a", "provider_b"}
    # Two models on the same provider use the SAME client (and therefore
    # the same rate limiter — that's the contract that quotas are per-
    # endpoint, not per-model).
    client_for_provider_a = runner._provider_clients["provider_a"]
    client_for_provider_b = runner._provider_clients["provider_b"]
    assert client_for_provider_a is not client_for_provider_b
    # Limiters live on the client; identity check confirms sharing.
    assert client_for_provider_a._rpm_limiter is not None
    assert client_for_provider_b._rpm_limiter is not None
    assert client_for_provider_a._rpm_limiter is not client_for_provider_b._rpm_limiter
    # Provider's rpm_limit propagated to the limiter.
    assert client_for_provider_a._rpm_limiter.rpm == 10
    assert client_for_provider_b._rpm_limiter.rpm == 20


def test_runner_uses_dispatcher_for_model(tmp_path):
    """End-to-end: ISRunner with one model calls dispatcher,
    gets a result, writes a checkpoint."""
    from unittest.mock import MagicMock, patch
    from traces.config import (
        AtlasConfig, AuditConfig, ModelConfig, PipelineConfig, ProviderConfig,
        TracesConfig,
    )
    from traces.pipeline.runner import ISRunner
    from traces.prompts import ISProbe

    config = TracesConfig(
        atlas=AtlasConfig(ontology_path="x", vocabularies_path="y"),
        providers={"p": ProviderConfig(base_url="http://x", rpm_limit=0)},
        pipeline=PipelineConfig(concurrency=1, checkpoint_interval=10),
        audit=AuditConfig(provider="p", judge_model="m1", proposer_model="m1"),
        models=[ModelConfig(id="m1", provider="p", provider_model_id="m1")],
    )
    runner = ISRunner(config)

    probe = MagicMock(spec=ISProbe)
    probe.probe_id = "P1"
    probe.paper_id = "paper1"
    probe.user_prompt = "u"
    probe.system_prompt = ""

    # Patch ProviderClient.complete to return a canned response
    with patch.object(
        runner._provider_clients["p"], "complete",
        return_value=MagicMock(
            content="ok", latency_ms=1.0, prompt_tokens=1, completion_tokens=1,
            finish_reason="stop", model="m1",
        ),
    ):
        results = runner.run(probes=[probe], checkpoint_path=str(tmp_path / "ckpt.json"))
    assert len(results) == 1
    r = results[0]
    assert r.model == "m1"
    assert r.error is None


def test_runner_concurrency_with_failing_model_marks_tripped(tmp_path):
    """Under concurrency=4 with one model always failing, the model trips
    and remaining probes get model_tripped errors without thread-pool starvation.
    """
    from unittest.mock import MagicMock
    from traces.config import (
        AtlasConfig, AuditConfig, ModelConfig, PipelineConfig, ProviderConfig,
        TracesConfig, TripThresholdsConfig,
    )
    from traces.pipeline.runner import ISRunner
    from traces.pipeline.provider_client import ProviderHTTPError

    config = TracesConfig(
        atlas=AtlasConfig(ontology_path="x", vocabularies_path="y"),
        providers={"p": ProviderConfig(base_url="http://x", rpm_limit=0)},
        pipeline=PipelineConfig(
            concurrency=4,
            checkpoint_interval=10,
            trip_thresholds=TripThresholdsConfig(
                consecutive_failures=2,  # trip fast for the test
                rate_window_size=10,
                rate_min_samples=10,  # disable rate path
                rate_threshold=1.0,
                wallclock_no_success_seconds=60.0,
            ),
        ),
        models=[
            ModelConfig(
                id="primary",
                provider="p",
                provider_model_id="primary",
                max_inflight=2,
            ),
        ],
        audit=AuditConfig(provider="p", judge_model="j", proposer_model="r"),
    )
    runner = ISRunner(config)

    def fake_complete(model: str, **kwargs):
        raise ProviderHTTPError(503, "primary down")

    runner._provider_clients["p"].complete = fake_complete

    probes = []
    for i in range(8):
        p = MagicMock()
        p.probe_id = f"P{i}"
        p.paper_id = f"paper{i}"
        p.user_prompt = "u"
        p.system_prompt = ""
        probes.append(p)

    results = runner.run(
        probes=probes,
        models=["primary"],
        checkpoint_path=str(tmp_path / "ckpt.json"),
    )

    # All 8 probes must produce results (no thread-pool starvation).
    assert len(results) == 8

    # Primary should be tripped after 2 consecutive failures.
    assert runner._dispatcher._health["primary"].tripped is True

    # All results should be errors (either the real error or model_tripped).
    assert all(r.error is not None for r in results)


def test_runner_resume_keeps_successful_pairs_skipped_and_retries_failures(tmp_path):
    from traces.config import AuditConfig

    config = TracesConfig(
        atlas=AtlasConfig(ontology_path="x", vocabularies_path="y"),
        providers={"p": ProviderConfig(base_url="http://x", rpm_limit=0)},
        pipeline=PipelineConfig(concurrency=1, checkpoint_interval=10),
        audit=AuditConfig(provider="p", judge_model="m1", proposer_model="m1"),
        models=[ModelConfig(id="m1", provider="p", provider_model_id="m1")],
    )
    runner = ISRunner(config)
    checkpoint = tmp_path / "ckpt.json"
    _write = {
        "completed": [["P1", "m1"]],
        "results": [
            asdict(_result("P1", "m1")),
            asdict(_result("P2", "m1", error="HTTP 502")),
        ],
    }
    checkpoint.write_text(json.dumps(_write))

    executed: list[str] = []

    def fake_execute(probe, model_id):
        executed.append(f"{probe.probe_id}:{model_id}")
        return _result(probe.probe_id, model_id)

    runner._dispatcher.execute = fake_execute

    results = runner.run(
        probes=[_probe("P1"), _probe("P2")],
        models=["m1"],
        checkpoint_path=str(checkpoint),
    )

    assert executed == ["P2:m1"]
    assert {(row.probe_id, row.model) for row in results if not row.error} == {
        ("P1", "m1"),
        ("P2", "m1"),
    }


def test_runner_emits_panel_summary_at_end(caplog, tmp_path):
    import logging
    from unittest.mock import MagicMock, patch
    from traces.config import (
        AtlasConfig, AuditConfig, ModelConfig, PipelineConfig, ProviderConfig, TracesConfig,
    )
    from traces.pipeline.runner import ISRunner

    config = TracesConfig(
        atlas=AtlasConfig(ontology_path="x", vocabularies_path="y"),
        providers={"p": ProviderConfig(base_url="http://x", rpm_limit=0)},
        pipeline=PipelineConfig(concurrency=1),
        models=[ModelConfig(id="m1", provider="p", provider_model_id="m1")],
        audit=AuditConfig(provider="p", judge_model="j", proposer_model="r"),
    )
    runner = ISRunner(config)
    probe = MagicMock()
    probe.probe_id = "P1"; probe.paper_id = "paper1"
    probe.user_prompt = "u"; probe.system_prompt = ""

    caplog.set_level(logging.INFO, logger="traces.pipeline.runner")
    with patch.object(
        runner._provider_clients["p"], "complete",
        return_value=MagicMock(
            content="ok", latency_ms=1.0, prompt_tokens=1, completion_tokens=1,
            finish_reason="stop", model="m1",
        ),
    ):
        runner.run(probes=[probe], checkpoint_path=str(tmp_path / "ckpt.json"))
    msgs = [r.message for r in caplog.records]
    assert any("model availability" in m.lower() for m in msgs), msgs
