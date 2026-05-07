"""Tests for traces.calibration.judge."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from traces.calibration.judge import (
    JudgeEmptyResponseError,
    JudgeError,
    JudgeRefusedError,
    _parse_judge_json,
    call_judge,
    is_empty_judge_response,
    is_refusal_shaped_judge_response,
)
from traces.calibration.models import JudgeLabel
from traces.config import AuditConfig, ProviderConfig


def _provider_and_audit(**audit_overrides) -> tuple[ProviderConfig, AuditConfig]:
    """Build a (ProviderConfig, AuditConfig) pair with tight retry budgets."""
    api_key = audit_overrides.pop("_provider_api_key", "test-key")
    timeout = audit_overrides.pop("_provider_timeout", 10)
    max_retries = audit_overrides.pop("max_retries", 2)
    provider = ProviderConfig(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    audit = AuditConfig(
        provider="nvidia",
        judge_model=audit_overrides.pop("judge_model", "test/judge-model"),
        proposer_model="test/proposer-model",
        parse_retries=audit_overrides.pop("parse_retries", 1),
        reasoning_effort=audit_overrides.pop("reasoning_effort", None),
        temperature=audit_overrides.pop("temperature", 0.0),
        top_p=audit_overrides.pop("top_p", 0.95),
        max_tokens=audit_overrides.pop("max_tokens", 8192),
    )
    return provider, audit


def _ok_label() -> dict:
    return {
        "classification": "ENGAGED_UNRECOGNIZED",
        "pass": False,
        "primary_reason": "delivered the asked task",
        "key_spans": ["..."],
        "confidence": "high",
        "rule_gap": None,
        "rule_gap_detail": None,
    }


def _ok_response_text() -> str:
    return json.dumps(_ok_label())


def _mock_http_response(content: str, status_code: int = 200) -> MagicMock:
    m = MagicMock(spec=requests.Response)
    m.status_code = status_code
    if status_code == 200:
        m.json.return_value = {"choices": [{"message": {"content": content}}]}
        m.text = json.dumps(m.json.return_value)
    else:
        m.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        m.text = content or "error"
    return m


class TestParseJudgeJson:
    def test_parses_clean_json(self):
        lbl = _parse_judge_json(_ok_response_text())
        assert isinstance(lbl, JudgeLabel)
        assert lbl.classification == "ENGAGED_UNRECOGNIZED"
        assert lbl.pass_ is False

    def test_parses_json_wrapped_in_text(self):
        wrapped = "Here you go: " + _ok_response_text() + "\n\n"
        lbl = _parse_judge_json(wrapped)
        assert lbl.classification == "ENGAGED_UNRECOGNIZED"

    def test_malformed_raises(self):
        with pytest.raises(JudgeError, match="parse"):
            _parse_judge_json("not json at all")

    def test_schema_violation_raises(self):
        bad = json.dumps({
            "classification": "NOT_A_REAL_LABEL",
            "pass": False,
            "primary_reason": "x",
            "key_spans": [],
            "confidence": "high",
            "rule_gap": None,
            "rule_gap_detail": None,
        })
        with pytest.raises(JudgeError, match="classification"):
            _parse_judge_json(bad)


class TestCallJudge:
    def test_happy_path(self):
        provider, audit = _provider_and_audit()
        with patch(
            "traces.calibration.llm_client.requests.post",
            return_value=_mock_http_response(_ok_response_text()),
        ) as mock_post:
            lbl = call_judge(
                payload="P", rubric="R", provider=provider, audit=audit,
            )
            assert lbl.classification == "ENGAGED_UNRECOGNIZED"
            assert mock_post.call_count == 1

    def test_http_failure_triggers_retries(self):
        provider, audit = _provider_and_audit()
        results = [
            _mock_http_response("oops", status_code=500),
            _mock_http_response("oops", status_code=500),
            _mock_http_response(_ok_response_text()),
        ]
        with patch(
            "traces.calibration.llm_client.requests.post",
            side_effect=results,
        ) as mock_post:
            lbl = call_judge(
                payload="P", rubric="R", provider=provider, audit=audit,
            )
            assert lbl.classification == "ENGAGED_UNRECOGNIZED"
            assert mock_post.call_count == 3

    def test_http_failure_exhausted_raises(self):
        provider, audit = _provider_and_audit()
        with patch(
            "traces.calibration.llm_client.requests.post",
            return_value=_mock_http_response("server died", status_code=500),
        ) as mock_post:
            with pytest.raises(JudgeError, match="failed"):
                call_judge(
                    payload="P", rubric="R", provider=provider, audit=audit,
                )
            assert mock_post.call_count == 3

    def test_malformed_json_triggers_one_retry(self):
        provider, audit = _provider_and_audit()
        results = [
            _mock_http_response("not json"),
            _mock_http_response(_ok_response_text()),
        ]
        with patch(
            "traces.calibration.llm_client.requests.post",
            side_effect=results,
        ) as mock_post:
            lbl = call_judge(
                payload="P", rubric="R", provider=provider, audit=audit,
            )
            assert lbl.classification == "ENGAGED_UNRECOGNIZED"
            assert mock_post.call_count == 2

    def test_malformed_json_retry_exhausted_raises(self):
        provider, audit = _provider_and_audit()
        with patch(
            "traces.calibration.llm_client.requests.post",
            return_value=_mock_http_response("not json"),
        ) as mock_post:
            with pytest.raises(JudgeError, match="parse"):
                call_judge(
                    payload="P", rubric="R", provider=provider, audit=audit,
                )
            assert mock_post.call_count == 2

    def test_request_body_includes_expected_fields(self):
        provider, audit = _provider_and_audit(reasoning_effort="high")
        with patch(
            "traces.calibration.llm_client.requests.post",
            return_value=_mock_http_response(_ok_response_text()),
        ) as mock_post:
            call_judge(
                payload="PAYLOAD", rubric="RUBRIC",
                provider=provider, audit=audit, model="OVERRIDE",
            )
            call = mock_post.call_args
            assert call.args[0] == provider.base_url + "/chat/completions"
            body = call.kwargs["json"]
            assert body["model"] == "OVERRIDE"
            assert body["temperature"] == audit.temperature
            assert body["top_p"] == audit.top_p
            assert body["max_tokens"] == audit.max_tokens
            assert body["stream"] is False
            assert body["reasoning_effort"] == "high"
            roles = [m["role"] for m in body["messages"]]
            assert roles == ["system", "user"]
            assert body["messages"][0]["content"] == "RUBRIC"
            assert body["messages"][1]["content"] == "PAYLOAD"
            headers = call.kwargs["headers"]
            assert headers["Authorization"] == f"Bearer {provider.api_key}"

    def test_uses_config_judge_model_when_no_override(self):
        provider, audit = _provider_and_audit(judge_model="cfg/judge")
        with patch(
            "traces.calibration.llm_client.requests.post",
            return_value=_mock_http_response(_ok_response_text()),
        ) as mock_post:
            call_judge(
                payload="P", rubric="R", provider=provider, audit=audit,
            )
            assert mock_post.call_args.kwargs["json"]["model"] == "cfg/judge"

    def test_all_timeouts_raise_judge_error_with_timeout_message(self):
        provider, audit = _provider_and_audit()
        with patch(
            "traces.calibration.llm_client.requests.post",
            side_effect=requests.Timeout("timed out"),
        ) as mock_post:
            with pytest.raises(JudgeError, match="timed out"):
                call_judge(
                    payload="P", rubric="R", provider=provider, audit=audit,
                )
            assert mock_post.call_count == 3


class TestEmptyAndRefusalDetection:
    """Cover the deterministic-output classifiers that decide whether the
    judge response is parseable, refusal-shaped, or empty. They gate the
    fallback-chain logic in judge_orchestrator._dispatch_with_fallback."""

    def test_empty_helper_treats_whitespace_as_empty(self):
        assert is_empty_judge_response("")
        assert is_empty_judge_response("   \n\t  ")
        assert not is_empty_judge_response("{}")

    @pytest.mark.parametrize("text", [
        "I can't help with that.",
        "I'm not able to evaluate this content.",
        "I cannot comply.",
        "This content violates my policy.",
        "I decline to respond.",
    ])
    def test_refusal_helper_matches_known_markers(self, text):
        assert is_refusal_shaped_judge_response(text)

    def test_refusal_helper_skips_long_responses(self):
        # The 100-char gate prevents false positives on legit verdicts that
        # happen to mention "I cannot" inside their primary_reason.
        long_with_marker = (
            '{"classification": "ENGAGED_RECOGNIZED", "pass": false, '
            '"primary_reason": "model says i cannot rule out residuals", '
            '"key_spans": ["..."], "confidence": "high", '
            '"rule_gap": null, "rule_gap_detail": null}'
        )
        assert len(long_with_marker) >= 100
        assert not is_refusal_shaped_judge_response(long_with_marker)

    def test_refusal_helper_returns_false_on_empty(self):
        # is_empty_judge_response is the dedicated empty-checker; refusal
        # should not also fire on whitespace.
        assert not is_refusal_shaped_judge_response("")
        assert not is_refusal_shaped_judge_response("    ")

    def test_call_judge_raises_empty_on_blank_response(self):
        provider, audit = _provider_and_audit(max_retries=0)
        body = {"choices": [{"message": {"content": "   "}}]}
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = body
        with patch(
            "traces.calibration.llm_client.requests.post", return_value=resp,
        ):
            with pytest.raises(JudgeEmptyResponseError):
                call_judge(
                    payload="P", rubric="R", provider=provider, audit=audit,
                )

    def test_call_judge_raises_refused_on_refusal_shape(self):
        provider, audit = _provider_and_audit(max_retries=0)
        body = {"choices": [{"message": {"content": "I can't help."}}]}
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = body
        with patch(
            "traces.calibration.llm_client.requests.post", return_value=resp,
        ):
            with pytest.raises(JudgeRefusedError) as ei:
                call_judge(
                    payload="P", rubric="R", provider=provider, audit=audit,
                )
        # The exception carries the model + raw response for the dispatcher
        # to log when falling through to the next chain entry.
        assert ei.value.model == audit.judge_model
        assert "can't help" in ei.value.response_text
