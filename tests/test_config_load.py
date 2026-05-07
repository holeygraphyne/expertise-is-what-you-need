"""Tests for TracesConfig.load — env-var precedence + .env loading.

The project follows a 12-factor-style precedence model for sensitive
config fields:

    explicit env var > .env file > config-file value > pydantic default

This is implemented in `TracesConfig.load`. Each provider entry under
`providers:` derives its env-var name from
`<PROVIDER_NAME_UPPER>_API_KEY` automatically (see
`TestProviderEnvPrecedence` below).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from traces.config import TracesConfig


# --- Multi-provider env-var precedence ---


_MULTI_PROVIDER_CONFIG = """\
corpus:
  root: "traces/corpus"
atlas:
  ontology_path: "../atlas-ontology/src/ontology/atlas.ttl"
  vocabularies_path: "../atlas-ontology/vocabularies/"
providers:
  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    api_key: "{or_key}"
  nvidia:
    base_url: "https://integrate.api.nvidia.com/v1"
    api_key: "{nvidia_key}"
  custom:
    base_url: "https://example.local/v1"
    api_key: "{custom_key}"
"""


def _write_multi_config(tmp_path, *, or_key="", nvidia_key="", custom_key=""):
    p = tmp_path / "traces_config.yaml"
    p.write_text(_MULTI_PROVIDER_CONFIG.format(
        or_key=or_key, nvidia_key=nvidia_key, custom_key=custom_key,
    ))
    return p


# --- .env file loading ---


class TestDotenvLoading:
    """`.env` file in the load CWD populates env vars (12-factor)."""

    def test_dotenv_value_applied(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=dotenv-value\n")
        cfg_path = _write_multi_config(tmp_path, or_key="literal")
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["openrouter"].api_key == "dotenv-value"

    def test_shell_env_wins_over_dotenv(self, tmp_path, monkeypatch):
        """Shell-exported env var beats .env file (load_dotenv default)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "shell-value")
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=dotenv-value\n")
        cfg_path = _write_multi_config(tmp_path, or_key="")
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["openrouter"].api_key == "shell-value"

    def test_no_dotenv_no_error(self, tmp_path, monkeypatch):
        """Missing .env is fine — load() succeeds and falls through to literal."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        cfg_path = _write_multi_config(tmp_path, or_key="literal")
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["openrouter"].api_key == "literal"


class TestProviderEnvPrecedence:
    """Each provider's api_key picks up `<PROVIDER_NAME_UPPER>_API_KEY`."""

    def test_openrouter_env_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-env")
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        cfg_path = _write_multi_config(tmp_path, or_key="literal-or")
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["openrouter"].api_key == "or-env"

    def test_nvidia_env_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-env")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        cfg_path = _write_multi_config(tmp_path, nvidia_key="literal-nvidia")
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["nvidia"].api_key == "nvidia-env"

    def test_arbitrary_provider_name_picks_up_env(self, tmp_path, monkeypatch):
        """Adding a new provider just requires setting <NAME>_API_KEY — no code change."""
        monkeypatch.setenv("CUSTOM_API_KEY", "custom-env")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        cfg_path = _write_multi_config(tmp_path, custom_key="literal-custom")
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["custom"].api_key == "custom-env"

    def test_no_env_uses_literal_per_provider(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        cfg_path = _write_multi_config(
            tmp_path, or_key="lit-or", nvidia_key="lit-nv", custom_key="lit-cu",
        )
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["openrouter"].api_key == "lit-or"
        assert config.providers["nvidia"].api_key == "lit-nv"
        assert config.providers["custom"].api_key == "lit-cu"

    def test_provider_with_hyphenated_name_uppercases_correctly(self, tmp_path, monkeypatch):
        """`my-provider` → MY_PROVIDER_API_KEY (hyphens become underscores in env name)."""
        monkeypatch.setenv("MY_PROVIDER_API_KEY", "hyphen-env")
        cfg_path = tmp_path / "traces_config.yaml"
        cfg_path.write_text(
            'corpus:\n  root: "traces/corpus"\n'
            'atlas:\n  ontology_path: "x"\n  vocabularies_path: "y"\n'
            'providers:\n'
            '  my-provider:\n'
            '    base_url: "https://example/v1"\n'
            '    api_key: "literal"\n'
            # audit.provider defaults to "nvidia"; override it to a known key
            'audit:\n'
            '  provider: "my-provider"\n'
        )
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.providers["my-provider"].api_key == "hyphen-env"


class TestModelProviderReference:
    def test_model_provider_must_reference_known_provider(self, tmp_path, monkeypatch):
        """A `provider:` field on a model must match a key in `providers:`."""
        from pydantic import ValidationError
        cfg_path = tmp_path / "traces_config.yaml"
        cfg_path.write_text(
            'corpus:\n  root: "traces/corpus"\n'
            'atlas:\n  ontology_path: "x"\n  vocabularies_path: "y"\n'
            'providers:\n'
            '  nvidia:\n'
            '    base_url: "https://integrate.api.nvidia.com/v1"\n'
            'models:\n'
            '  - id: bad-model\n'
            '    provider: openrouter      # not in providers dict\n'
            '    provider_model_id: "anything"\n'
        )
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValidationError, match=r"provider.*openrouter"):
            TracesConfig.load(cfg_path)

    def test_audit_provider_must_reference_known_provider(self, tmp_path, monkeypatch):
        """`audit.provider` must match a key in `providers:` — same rule
        as `models[].provider`. Mirrors the `_validate_provider_references`
        validator's second branch."""
        from pydantic import ValidationError
        cfg_path = tmp_path / "traces_config.yaml"
        cfg_path.write_text(
            'corpus:\n  root: "traces/corpus"\n'
            'atlas:\n  ontology_path: "x"\n  vocabularies_path: "y"\n'
            'providers:\n'
            '  nvidia:\n'
            '    base_url: "https://integrate.api.nvidia.com/v1"\n'
            'audit:\n'
            '  provider: ghost-provider      # not in providers dict\n'
        )
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValidationError, match=r"audit.provider.*ghost-provider"):
            TracesConfig.load(cfg_path)

    def test_model_with_known_provider_loads(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "traces_config.yaml"
        cfg_path.write_text(
            'corpus:\n  root: "traces/corpus"\n'
            'atlas:\n  ontology_path: "x"\n  vocabularies_path: "y"\n'
            'providers:\n'
            '  nvidia:\n'
            '    base_url: "https://integrate.api.nvidia.com/v1"\n'
            'models:\n'
            '  - id: deepseek-v4\n'
            '    provider: nvidia\n'
            '    provider_model_id: "deepseek-ai/deepseek-v4-pro"\n'
        )
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        assert config.models[0].provider == "nvidia"
        assert config.models[0].provider_model_id == "deepseek-ai/deepseek-v4-pro"

    def test_model_without_provider_rejected(self, tmp_path, monkeypatch):
        """`provider` and `provider_model_id` are required on every model."""
        from pydantic import ValidationError
        cfg_path = tmp_path / "traces_config.yaml"
        cfg_path.write_text(
            'corpus:\n  root: "traces/corpus"\n'
            'atlas:\n  ontology_path: "x"\n  vocabularies_path: "y"\n'
            'models:\n'
            '  - id: legacy\n'
        )
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValidationError):
            TracesConfig.load(cfg_path)

    def test_model_per_call_defaults_have_expected_values(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "traces_config.yaml"
        cfg_path.write_text(
            'corpus:\n  root: "traces/corpus"\n'
            'atlas:\n  ontology_path: "x"\n  vocabularies_path: "y"\n'
            'providers:\n'
            '  nvidia:\n'
            '    base_url: "https://x"\n'
            'models:\n'
            '  - id: foo\n'
            '    provider: nvidia\n'
            '    provider_model_id: "v"\n'
        )
        monkeypatch.chdir(tmp_path)
        config = TracesConfig.load(cfg_path)
        m = config.models[0]
        assert m.temperature == 1.0
        assert m.max_tokens == 4096
        assert m.seed is None


def test_pipeline_trip_threshold_defaults():
    from traces.config import PipelineConfig
    p = PipelineConfig()
    assert p.trip_thresholds.consecutive_failures == 5
    assert p.trip_thresholds.rate_window_size == 10
    assert p.trip_thresholds.rate_min_samples == 5
    assert p.trip_thresholds.rate_threshold == 0.7
    assert p.trip_thresholds.wallclock_no_success_seconds == 120.0


def test_pipeline_trip_threshold_overrides():
    from traces.config import PipelineConfig
    p = PipelineConfig(trip_thresholds={"consecutive_failures": 10})
    assert p.trip_thresholds.consecutive_failures == 10
    assert p.trip_thresholds.rate_window_size == 10


def test_trip_thresholds_to_thresholds_round_trip():
    """TripThresholdsConfig.to_thresholds() preserves every threshold field."""
    from traces.config import TripThresholdsConfig
    rc = TripThresholdsConfig(
        consecutive_failures=7,
        rate_window_size=20,
        rate_min_samples=8,
        rate_threshold=0.9,
        wallclock_no_success_seconds=120.0,
    )
    t = rc.to_thresholds()
    assert t.consecutive_failures == 7
    assert t.rate_window_size == 20
    assert t.rate_min_samples == 8
    assert t.rate_threshold == 0.9
    assert t.wallclock_no_success_seconds == 120.0


def test_model_max_inflight_defaults_to_two():
    from traces.config import ModelConfig
    m = ModelConfig(id="m", provider="p", provider_model_id="m")
    assert m.max_inflight == 2


def test_model_max_tokens_defaults_to_current_value():
    from traces.config import ModelConfig
    m = ModelConfig(id="m", provider="p", provider_model_id="m")
    assert m.max_tokens == 4096


def test_model_max_tokens_override():
    from traces.config import ModelConfig
    m = ModelConfig(id="m", provider="p", provider_model_id="m", max_tokens=2048)
    assert m.max_tokens == 2048


def test_model_max_tokens_rejects_zero():
    from traces.config import ModelConfig
    with pytest.raises(ValueError, match="max_tokens"):
        ModelConfig(id="m", provider="p", provider_model_id="m", max_tokens=0)


def test_model_max_tokens_rejects_negative():
    from traces.config import ModelConfig
    with pytest.raises(ValueError, match="max_tokens"):
        ModelConfig(id="m", provider="p", provider_model_id="m", max_tokens=-1)


def test_model_max_inflight_override():
    from traces.config import ModelConfig
    m = ModelConfig(id="m", provider="p", provider_model_id="m", max_inflight=5)
    assert m.max_inflight == 5


def test_model_max_inflight_rejects_zero():
    from traces.config import ModelConfig
    with pytest.raises(ValueError, match="max_inflight"):
        ModelConfig(id="m", provider="p", provider_model_id="m", max_inflight=0)
