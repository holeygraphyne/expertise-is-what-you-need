"""Unit tests for the calibration HTTP client."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from traces.calibration.llm_client import (
    LLMError,
    LLMTimeout,
    call_chat_completion,
)
from traces.config import AuditConfig, ProviderConfig


def _provider_and_audit(**audit_overrides) -> tuple[ProviderConfig, AuditConfig]:
    """Build a (ProviderConfig, AuditConfig) pair with sane test defaults.

    `audit_overrides` only configures AuditConfig; provider has fixed
    test defaults (api_key, timeout). Pass api_key="" via the
    `_provider_api_key` key (popped before AuditConfig construction)
    to test the empty-key path.
    """
    api_key = audit_overrides.pop("_provider_api_key", "test-key")
    timeout = audit_overrides.pop("_provider_timeout", 10)
    provider = ProviderConfig(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=timeout,
    )
    audit = AuditConfig(
        provider="nvidia",
        judge_model="test/judge",
        proposer_model="test/proposer",
        reasoning_effort=audit_overrides.pop("reasoning_effort", None),
        temperature=audit_overrides.pop("temperature", 0.0),
        top_p=audit_overrides.pop("top_p", 0.95),
        max_tokens=audit_overrides.pop("max_tokens", 8192),
        parse_retries=audit_overrides.pop("parse_retries", 1),
    )
    return provider, audit


def _http_response(content: str, status_code: int = 200) -> MagicMock:
    m = MagicMock(spec=requests.Response)
    m.status_code = status_code
    if status_code == 200:
        m.json.return_value = {"choices": [{"message": {"content": content}}]}
        m.text = json.dumps(m.json.return_value)
    else:
        m.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        m.text = content or "error"
    return m


def test_happy_path_returns_content():
    provider, audit = _provider_and_audit()
    with patch(
        "traces.calibration.llm_client.requests.post",
        return_value=_http_response("PONG"),
    ) as mock_post:
        out = call_chat_completion(
            provider=provider,
            model="m1",
            system_prompt="S",
            user_prompt="U",
            temperature=audit.temperature,
            max_tokens=audit.max_tokens,
            top_p=audit.top_p,
            reasoning_effort=audit.reasoning_effort,
        )
        assert out == "PONG"
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "m1"
        assert body["temperature"] == audit.temperature
        assert body["top_p"] == audit.top_p
        assert body["max_tokens"] == audit.max_tokens
        assert body["stream"] is False
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"


def test_reasoning_effort_included_when_set():
    provider, audit = _provider_and_audit(reasoning_effort="high")
    with patch(
        "traces.calibration.llm_client.requests.post",
        return_value=_http_response("X"),
    ) as mock_post:
        call_chat_completion(
            provider=provider, model="m", system_prompt="s", user_prompt="u",
            temperature=audit.temperature, max_tokens=audit.max_tokens,
            top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
        )
        assert mock_post.call_args.kwargs["json"]["reasoning_effort"] == "high"


def test_reasoning_effort_omitted_when_none():
    provider, audit = _provider_and_audit(reasoning_effort=None)
    with patch(
        "traces.calibration.llm_client.requests.post",
        return_value=_http_response("X"),
    ) as mock_post:
        call_chat_completion(
            provider=provider, model="m", system_prompt="s", user_prompt="u",
            temperature=audit.temperature, max_tokens=audit.max_tokens,
            top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
        )
        assert "reasoning_effort" not in mock_post.call_args.kwargs["json"]


def test_no_auth_header_when_api_key_empty():
    provider, audit = _provider_and_audit(_provider_api_key="")
    with patch(
        "traces.calibration.llm_client.requests.post",
        return_value=_http_response("X"),
    ) as mock_post:
        call_chat_completion(
            provider=provider, model="m", system_prompt="s", user_prompt="u",
            temperature=audit.temperature, max_tokens=audit.max_tokens,
            top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
        )
        assert "Authorization" not in mock_post.call_args.kwargs["headers"]


def test_non_200_raises_llmerror():
    provider, audit = _provider_and_audit()
    with patch(
        "traces.calibration.llm_client.requests.post",
        return_value=_http_response("server down", status_code=500),
    ):
        with pytest.raises(LLMError, match="500"):
            call_chat_completion(
                provider=provider, model="m", system_prompt="s", user_prompt="u",
                temperature=audit.temperature, max_tokens=audit.max_tokens,
                top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
            )


def test_timeout_raises_llmtimeout():
    provider, audit = _provider_and_audit()
    with patch(
        "traces.calibration.llm_client.requests.post",
        side_effect=requests.Timeout("timed out"),
    ):
        with pytest.raises(LLMTimeout):
            call_chat_completion(
                provider=provider, model="m", system_prompt="s", user_prompt="u",
                temperature=audit.temperature, max_tokens=audit.max_tokens,
                top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
            )


def test_request_exception_raises_llmerror():
    provider, audit = _provider_and_audit()
    with patch(
        "traces.calibration.llm_client.requests.post",
        side_effect=requests.ConnectionError("dns fail"),
    ):
        with pytest.raises(LLMError, match="failed"):
            call_chat_completion(
                provider=provider, model="m", system_prompt="s", user_prompt="u",
                temperature=audit.temperature, max_tokens=audit.max_tokens,
                top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
            )


def test_malformed_json_response_raises_llmerror():
    provider, audit = _provider_and_audit()
    bad_resp = MagicMock(spec=requests.Response)
    bad_resp.status_code = 200
    bad_resp.json.side_effect = json.JSONDecodeError("err", "doc", 0)
    bad_resp.text = "not json"
    with patch(
        "traces.calibration.llm_client.requests.post",
        return_value=bad_resp,
    ):
        with pytest.raises(LLMError, match="not JSON"):
            call_chat_completion(
                provider=provider, model="m", system_prompt="s", user_prompt="u",
                temperature=audit.temperature, max_tokens=audit.max_tokens,
                top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
            )


def test_missing_choices_raises_llmerror():
    provider, audit = _provider_and_audit()
    bad_resp = MagicMock(spec=requests.Response)
    bad_resp.status_code = 200
    bad_resp.json.return_value = {"unexpected": "shape"}
    bad_resp.text = "{}"
    with patch(
        "traces.calibration.llm_client.requests.post",
        return_value=bad_resp,
    ):
        with pytest.raises(LLMError, match="choices"):
            call_chat_completion(
                provider=provider, model="m", system_prompt="s", user_prompt="u",
                temperature=audit.temperature, max_tokens=audit.max_tokens,
                top_p=audit.top_p, reasoning_effort=audit.reasoning_effort,
            )
