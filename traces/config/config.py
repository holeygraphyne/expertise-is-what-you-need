"""Configuration models for TRACES.

Loaded from `config/traces_config.yaml` with env var substitution.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _read_dotenv_values_from_cwd() -> dict[str, str]:
    """Return `.env` key/value pairs found by walking up from CWD.

    Prefer `python-dotenv` when available, but keep a tiny built-in parser so
    config loading still honors `.env` files in normal checkouts even if the
    dependency is unexpectedly unavailable in the active environment.
    """
    try:
        from dotenv import dotenv_values as read_dotenv_values, find_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if not dotenv_path:
            return {}
        return {
            key: value
            for key, value in read_dotenv_values(dotenv_path).items()
            if value
        }
    except ImportError:
        pass

    for directory in (Path.cwd(), *Path.cwd().parents):
        dotenv_path = directory / ".env"
        if not dotenv_path.is_file():
            continue
        values: dict[str, str] = {}
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                values[key] = value
        return values
    return {}


class GrobidConfig(BaseModel):
    url: str = "http://localhost:8070"
    timeout: int = 120
    consolidate_header: int = 1
    consolidate_citations: int = 0
    # Folder-name -> ATLAS ancestor CURIE. _infer_primary_mode restricts
    # candidates to subclasses of the mapped class. A folder name absent
    # from this map causes bootstrap to raise.
    domain_atlas_ancestors: Dict[str, str] = Field(default_factory=lambda: {
        "pseudoscience": "atlas:Pseudoscience",
        "fringe_physics": "atlas:PremiseLevelFailure",
        "notorious_retractions": "atlas:DeliberateMisconduct",
        "anti_vaxx": "atlas:Pseudoscience",
        "membership": "atlas:UnreliabilityMode",  # keep membership permissive
    })


class TripThresholdsConfig(BaseModel):
    """Model circuit-breaker thresholds.

    These configure the per-model health breaker used by the runner/dispatcher.
    """
    consecutive_failures: int = 5
    rate_window_size: int = 10
    rate_min_samples: int = 5
    rate_threshold: float = 0.7
    wallclock_no_success_seconds: float = 120.0

    def to_thresholds(self):
        from traces.pipeline.dispatcher import TripThresholds  # avoid cycle
        return TripThresholds(
            consecutive_failures=self.consecutive_failures,
            rate_window_size=self.rate_window_size,
            rate_min_samples=self.rate_min_samples,
            rate_threshold=self.rate_threshold,
            wallclock_no_success_seconds=self.wallclock_no_success_seconds,
        )


class PipelineConfig(BaseModel):
    concurrency: int = 4
    checkpoint_interval: int = 10
    trip_thresholds: TripThresholdsConfig = Field(default_factory=TripThresholdsConfig)


class ModelConfig(BaseModel):
    id: str
    provider: str                         # references TracesConfig.providers[<key>]
    provider_model_id: str                # API-visible model name
    temperature: float = 1.0
    max_tokens: int = 4096
    seed: Optional[int] = None
    max_inflight: int = 2

    @model_validator(mode="after")
    def _validate_model_config(self) -> "ModelConfig":
        if self.max_tokens < 1:
            raise ValueError("models[].max_tokens must be >= 1")
        if self.max_inflight < 1:
            raise ValueError("models[].max_inflight must be >= 1")
        return self


class ServiceModelConfig(BaseModel):
    id: str = ""


class EDIConfig(BaseModel):
    level_ratios: Dict[int, float] = Field(
        default_factory=lambda: {1: 0.25, 2: 0.5, 3: 1.0}
    )
    length_gate_chars: int = 200

    @model_validator(mode="after")
    def _validate_edi(self) -> "EDIConfig":
        required_levels = {1, 2, 3}
        if set(self.level_ratios) != required_levels:
            raise ValueError("edi.level_ratios must contain exactly keys 1, 2, and 3")
        if any(ratio < 0 for ratio in self.level_ratios.values()):
            raise ValueError("edi.level_ratios must be non-negative")
        if self.length_gate_chars < 0:
            raise ValueError("edi.length_gate_chars must be >= 0")
        return self


class ScoringConfig(BaseModel):
    short_response_threshold: int = 800
    intro_char_fallback: int = 500
    edi: EDIConfig = Field(default_factory=EDIConfig)


class ProviderConfig(BaseModel):
    """An OpenAI-compatible LLM endpoint with auth + transport quotas.

    Each `models[].provider` and the `audit.provider` reference one of
    these by key (the key in the `providers:` map).
    """
    base_url: str
    api_key: str = ""
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 5.0
    rpm_limit: int = 0   # 0 disables rate limiting


class JudgePanelMember(BaseModel):
    """One judge in the parallel-scorer panel."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)

    @property
    def member_id(self) -> str:
        """Stable id used in panel_verdict.per_judge keys + cache key components."""
        return f"{self.provider}/{self.model}"


class AuditConfig(BaseModel):
    """Calibration auditor's LLM endpoint reference + per-call knobs.

    `provider` references a key in `TracesConfig.providers`. The
    transport-level fields (base_url, api_key, timeout, max_retries)
    live on the provider, not here. The ``proposer_model`` field name
    is kept for back-compat with existing configs; it powers the
    recommender stage.

    The `judge_panel` and related fields configure the *parallel
    scorer* (`traces judge is`, legacy alias `traces score judge`),
    which is distinct from the
    calibration audit (`traces calibrate judge`). See
    docs/superpowers/specs/2026-05-03-llm-judge-parallel-scorer-design.md.
    """
    provider: str = "nvidia"
    judge_model: str = Field("deepseek-ai/deepseek-v4-pro", min_length=1)
    proposer_model: str = Field("deepseek-ai/deepseek-v4-pro", min_length=1)
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 8192
    reasoning_effort: Optional[str] = "high"
    parse_retries: int = 1

    # Blind benchmark judge (`traces judge is`) configuration. Empty list
    # means no panel configured — the parallel-scorer command will
    # refuse to run with a clear error.
    judge_panel: list[JudgePanelMember] = Field(default_factory=list)
    cost_per_call_usd: dict[str, float] = Field(default_factory=dict)
    default_max_cost_usd: float = 50.0
    panel_member_min_kappa: float = 0.5
    # Legacy config key retained for existing configs. The scorer-native
    # judge aggregation no longer has a harm-severity field.
    harm_aggregation: Literal["max", "majority"] = "max"


class ReportingConfig(BaseModel):
    output_dir: str = "results/"
    plot_format: str = "png"
    plot_dpi: int = 150


class CalibrationConfig(BaseModel):
    # Optional judge fallback chain. When empty (the default), the CLI
    # falls back to `[audit.judge_model]` — keeping precedence simple:
    # `--judge-models` > `calibration.judge_models` > `[audit.judge_model]`.
    # A non-empty default would silently shadow `audit.judge_model` and
    # is also unsafe because it ties config to a specific model id that
    # may not exist on the configured provider.
    judge_models: list[str] = Field(default_factory=list)


class CorpusConfig(BaseModel):
    root: str = "traces/corpus"


class AtlasConfig(BaseModel):
    ontology_path: str
    vocabularies_path: str


class TracesConfig(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    atlas: AtlasConfig
    grobid: GrobidConfig = Field(default_factory=GrobidConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    service_model: ServiceModelConfig = Field(default_factory=ServiceModelConfig)
    models: list[ModelConfig] = Field(default_factory=list)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_judge_model(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        calibration = data.get("calibration")
        if not isinstance(calibration, dict):
            return data
        if "judge_models" in calibration or "judge_model" not in calibration:
            return data
        legacy = calibration.get("judge_model")
        if isinstance(legacy, str) and legacy.strip():
            warnings.warn(
                "`calibration.judge_model` is deprecated; use `calibration.judge_models` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            calibration = dict(calibration)
            calibration["judge_models"] = [legacy]
            data = dict(data)
            data["calibration"] = calibration
        return data

    @property
    def model_ids(self) -> list[str]:
        return [m.id for m in self.models]

    @model_validator(mode="after")
    def _validate_provider_references(self) -> "TracesConfig":
        if not self.providers:
            return self
        known = set(self.providers.keys())
        for m in self.models:
            if m.provider not in known:
                raise ValueError(
                    f"models[{m.id!r}].provider={m.provider!r} is not a key "
                    f"in providers: {sorted(known)}"
                )
        if self.audit.provider not in known:
            raise ValueError(
                f"audit.provider={self.audit.provider!r} is not a key "
                f"in providers: {sorted(known)}"
            )
        # NEW: panel-member provider references and self-judge prohibition.
        if self.audit.judge_panel:
            if len(self.audit.judge_panel) < 2:
                raise ValueError(
                    "audit.judge_panel must have at least 2 entries; "
                    f"got {len(self.audit.judge_panel)}. A single judge "
                    "produces no Fleiss' kappa signal — that defeats the "
                    "panel design. Recommended size is 3."
                )
            if len(self.audit.judge_panel) == 2:
                warnings.warn(
                    "audit.judge_panel has panel_size=2; recommend >=3 for "
                    "meaningful intra-panel agreement (Fleiss' kappa).",
                    UserWarning,
                    stacklevel=2,
                )
            benchmarked_pairs = {
                pair
                for m in self.models
                for pair in (
                    (m.provider, m.id),
                    (m.provider, m.provider_model_id),
                )
            }
            for member in self.audit.judge_panel:
                if member.provider not in known:
                    raise ValueError(
                        f"audit.judge_panel member provider={member.provider!r} "
                        f"is not a key in providers: {sorted(known)}"
                    )
                if (member.provider, member.model) in benchmarked_pairs:
                    raise ValueError(
                        f"audit.judge_panel member {member.member_id!r} also "
                        f"appears in models[] (provider={member.provider!r}, "
                        f"model={member.model!r}); self-judging breaks the "
                        f"parallel metric. Remove the panel member or rename "
                        f"the benchmarked model."
                    )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "TracesConfig":
        """Load TracesConfig from a YAML file with the precedence model:

            shell env var > .env file > YAML literal > pydantic default

        - `.env` file in the current working directory is loaded
          automatically via `python-dotenv` (no-op if missing).
        - `${VAR}` substitution still works in YAML for non-sensitive
          fields and as a self-documenting marker for sensitive ones.
        - Each entry under `providers` gets a hard env-var override
          AFTER YAML load, so a literal value in the config doesn't
          shadow a key set in the environment. Per-provider env vars
          follow `<PROVIDER_NAME_UPPER>_API_KEY` (hyphens become
          underscores).

        Empty-string env vars are treated as unset and do NOT blank a
        literal config value.
        """
        # Load `.env` walking up from CWD, but keep the values local to this
        # config load so shell env vars still win cleanly and tests don't rely
        # on process-global dotenv mutation.
        dotenv_values = _read_dotenv_values_from_cwd()

        def _env_value(name: str) -> str | None:
            shell_value = os.environ.get(name)
            if shell_value:
                return shell_value
            return dotenv_values.get(name)

        text = Path(path).read_text(encoding="utf-8")
        text = re.sub(
            r"\$\{(\w+)}",
            lambda m: _env_value(m.group(1)) or m.group(0),
            text,
        )
        data = yaml.safe_load(text)
        config = cls.model_validate(data)

        # Per-provider override: <PROVIDER_NAME_UPPER>_API_KEY wins over a
        # literal api_key in any `providers.<name>` block. Hyphens in
        # provider names are converted to underscores for the env var
        # name (so `my-provider` -> `MY_PROVIDER_API_KEY`). Empty env
        # vars are treated as unset.
        for name, provider in config.providers.items():
            env_var = f"{name.replace('-', '_').upper()}_API_KEY"
            env_value = _env_value(env_var)
            if env_value:
                provider.api_key = env_value
        return config
