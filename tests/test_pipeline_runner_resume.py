import json
from pathlib import Path

from traces.config import (
    AtlasConfig, AuditConfig, ModelConfig, PipelineConfig, ProviderConfig, TracesConfig,
)
from traces.pipeline.runner import ISRunner, RawProbeResult


def _runner_with_models(models):
    config = TracesConfig(
        atlas=AtlasConfig(ontology_path="x", vocabularies_path="y"),
        providers={"p": ProviderConfig(base_url="http://x", rpm_limit=0)},
        pipeline=PipelineConfig(concurrency=1),
        models=models,
        audit=AuditConfig(provider="p", judge_model="j", proposer_model="r"),
    )
    return ISRunner(config), config


def _write_ckpt(path: Path, results: list[RawProbeResult]) -> None:
    from dataclasses import asdict
    data = {
        "completed": [
            [r.probe_id, r.model] for r in results if not r.error
        ],
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(data))


def test_resume_with_solo_slot_skips_completed(tmp_path):
    runner, _ = _runner_with_models([
        ModelConfig(id="m1", provider="p", provider_model_id="m1"),
    ])
    ckpt = tmp_path / "ckpt.json"
    _write_ckpt(ckpt, [
        RawProbeResult(probe_id="P1", paper_id="paperX", model="m1",
                       response_text="hi", latency_ms=1.0),
    ])
    runner._load_checkpoint(str(ckpt))
    assert ("P1", "m1") in runner._completed


def test_resume_non_terminal_error_is_retried(tmp_path):
    runner, _ = _runner_with_models([
        ModelConfig(id="m1", provider="p", provider_model_id="m1"),
    ])
    ckpt = tmp_path / "ckpt.json"
    _write_ckpt(ckpt, [
        # Error result — gets retried on resume
        RawProbeResult(probe_id="P1", paper_id="paperX", model="m1",
                       response_text="", latency_ms=0.0,
                       error="HTTP 502"),
    ])
    runner._load_checkpoint(str(ckpt))
    assert ("P1", "m1") not in runner._completed


def test_resume_legacy_checkpoint_loads(tmp_path):
    """Legacy checkpoint with slot_id / is_terminal_failure fields loads cleanly
    (those fields are stripped by _load_checkpoint)."""
    runner, _ = _runner_with_models([
        ModelConfig(id="m1", provider="p", provider_model_id="m1"),
    ])
    ckpt = tmp_path / "ckpt.json"
    # Hand-write a legacy checkpoint with the old extra fields
    legacy = {
        "completed": [["P1", "m1"]],
        "results": [{
            "probe_id": "P1", "paper_id": "paperX", "model": "m1",
            "response_text": "hi", "latency_ms": 1.0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "finish_reason": None, "timestamp": "", "error": None,
            "domain": None,
            "slot_id": "m1",
            "is_terminal_failure": False,
        }],
    }
    ckpt.write_text(json.dumps(legacy))
    runner._load_checkpoint(str(ckpt))
    assert ("P1", "m1") in runner._completed


def test_resume_does_not_restore_trip_state(tmp_path):
    """Trip state is per-process; ModelHealth starts fresh on resume."""
    runner, _ = _runner_with_models([
        ModelConfig(id="m1", provider="p", provider_model_id="m1"),
    ])
    ckpt = tmp_path / "ckpt.json"
    _write_ckpt(ckpt, [
        RawProbeResult(probe_id="P1", paper_id="paperX", model="m1",
                       response_text="hi", latency_ms=1.0),
    ])
    runner._load_checkpoint(str(ckpt))
    health = runner._dispatcher._health["m1"]
    assert health.tripped is False
    assert health.attempts == 0
    assert health.consecutive_failures == 0
