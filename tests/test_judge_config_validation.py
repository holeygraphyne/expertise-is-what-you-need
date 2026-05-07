"""Validation of audit.judge_panel + related parallel-scorer config."""
import warnings

import pytest
from pydantic import ValidationError

from traces.config import TracesConfig


def _base_yaml(extra_audit: str = "", extra_models: str = "") -> str:
    return f"""
atlas:
  ontology_path: ../atlas-ontology/atlas.ttl
  vocabularies_path: ../atlas-ontology/vocabularies/
providers:
  anthropic: {{ base_url: "https://api.anthropic.com/v1", api_key: "x" }}
  openai:    {{ base_url: "https://api.openai.com/v1",    api_key: "x" }}
  nvidia:    {{ base_url: "https://integrate.api.nvidia.com/v1", api_key: "x" }}
models:
  - {{ id: "test-model-1", provider: "anthropic", provider_model_id: "benchmarked-model-1" }}
{extra_models}
audit:
  provider: nvidia
  judge_panel:
    - {{ provider: anthropic, model: claude-opus-4-7 }}
    - {{ provider: openai,    model: gpt-5 }}
    - {{ provider: nvidia,    model: deepseek-ai/deepseek-v4-pro }}
{extra_audit}
"""


def _write_and_load(tmp_path, yaml_text):
    cfg_path = tmp_path / "traces_config.yaml"
    cfg_path.write_text(yaml_text)
    return TracesConfig.load(str(cfg_path))


class TestJudgePanelValidation:
    def test_valid_three_member_panel_loads(self, tmp_path):
        cfg = _write_and_load(tmp_path, _base_yaml())
        assert len(cfg.audit.judge_panel) == 3
        assert cfg.audit.judge_panel[0].provider == "anthropic"

    def test_panel_size_one_rejected(self, tmp_path):
        yaml_text = _base_yaml().replace(
            "  judge_panel:\n"
            "    - { provider: anthropic, model: claude-opus-4-7 }\n"
            "    - { provider: openai,    model: gpt-5 }\n"
            "    - { provider: nvidia,    model: deepseek-ai/deepseek-v4-pro }\n",
            "  judge_panel:\n"
            "    - { provider: anthropic, model: claude-opus-4-7 }\n"
        )
        with pytest.raises(ValidationError, match="judge_panel.*at least 2"):
            _write_and_load(tmp_path, yaml_text)

    def test_panel_size_two_warns_but_loads(self, tmp_path):
        yaml_text = _base_yaml().replace(
            "    - { provider: nvidia,    model: deepseek-ai/deepseek-v4-pro }\n",
            "",
        )
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            cfg = _write_and_load(tmp_path, yaml_text)
        assert any("panel_size=2" in str(w.message) for w in ws)
        assert len(cfg.audit.judge_panel) == 2

    def test_panel_member_unknown_provider_rejected(self, tmp_path):
        yaml_text = _base_yaml().replace(
            "    - { provider: anthropic, model: claude-opus-4-7 }\n",
            "    - { provider: bogus, model: foo }\n",
        )
        with pytest.raises(ValidationError, match="bogus"):
            _write_and_load(tmp_path, yaml_text)

    def test_panel_member_in_models_list_rejected(self, tmp_path):
        # Realistic shape: benchmarked model id is just the model name (no
        # provider prefix), and the panel member shares the same (provider,
        # model) pair — this is the self-judge case that must be rejected.
        yaml_text = _base_yaml(
            extra_models='  - { id: "claude-opus-4-7", provider: "anthropic", provider_model_id: "claude-opus-4-7" }\n'
        )
        with pytest.raises(ValidationError, match="self-judg"):
            _write_and_load(tmp_path, yaml_text)

    def test_panel_member_with_provider_prefixed_id_no_longer_required(self, tmp_path):
        # Regression: before the fix, self-judge was only caught when the
        # benchmarked model id was "<provider>/<model>" (so member_id matched).
        # After the fix, a realistic id="deepseek-ai/deepseek-v4-pro" on
        # provider="nvidia" paired with a panel member {provider: nvidia,
        # model: "deepseek-ai/deepseek-v4-pro"} must ALSO be rejected.
        yaml_text = _base_yaml(
            extra_models='  - { id: "deepseek-ai/deepseek-v4-pro", provider: "nvidia", provider_model_id: "deepseek-ai/deepseek-v4-pro" }\n'
        )
        with pytest.raises(ValidationError, match="self-judg"):
            _write_and_load(tmp_path, yaml_text)

    def test_panel_member_with_disambiguated_id_matches_provider_model_id(self, tmp_path):
        yaml_text = _base_yaml(
            extra_models=(
                '  - { id: "groq/openai/gpt-oss-120b", provider: "groq", '
                'provider_model_id: "openai/gpt-oss-120b" }\n'
            ),
        ).replace(
            '  nvidia:    { base_url: "https://integrate.api.nvidia.com/v1", api_key: "x" }\n',
            '  nvidia:    { base_url: "https://integrate.api.nvidia.com/v1", api_key: "x" }\n'
            '  groq:      { base_url: "https://api.groq.com/openai/v1", api_key: "x" }\n',
        ).replace(
            "    - { provider: anthropic, model: claude-opus-4-7 }\n",
            '    - { provider: groq, model: "openai/gpt-oss-120b" }\n',
        )
        with pytest.raises(ValidationError, match="self-judg"):
            _write_and_load(tmp_path, yaml_text)

    def test_harm_aggregation_default_max(self, tmp_path):
        cfg = _write_and_load(tmp_path, _base_yaml())
        assert cfg.audit.harm_aggregation == "max"

    def test_harm_aggregation_invalid_rejected(self, tmp_path):
        yaml_text = _base_yaml(extra_audit="  harm_aggregation: median\n")
        with pytest.raises(ValidationError, match="harm_aggregation"):
            _write_and_load(tmp_path, yaml_text)
