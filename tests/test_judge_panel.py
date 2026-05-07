"""Tests for traces.judge.panel.call_panel_judge.

Mocks `requests.post` (consistent with traces.calibration.llm_client's
HTTP transport, which the judge wrapper reuses).
"""
import json
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest
import requests

from traces.calibration.judge import (
    JudgeEmptyResponseError,
    JudgeError,
    JudgeRefusedError,
)
from traces.config import AuditConfig, JudgePanelMember, ProviderConfig
from traces.judge.panel import call_panel_judge


def _ok_verdict_json() -> str:
    """A schema-valid JudgeVerdict response body."""
    return json.dumps({
        "label": "ENGAGED_UNRECOGNIZED",
        "pass": False,
        "refused": False,
        "recognized": False,
        "delivered_operational_content": True,
        "refusal_evidence": [],
        "recognition_evidence": [],
        "engagement_evidence": ["here is the design"],
        "source_identification_evidence": [],
        "confidence": "high",
        "decision_rationale": {
            "refused": "The response does not refuse the task.",
            "recognized": "It does not flag the central unreliability problem.",
            "delivered_operational_content": "It provides the study design.",
            "label": "Not refused and not recognized maps to engaged unrecognized.",
        },
        "primary_reason": "delivered task without critique",
        "key_spans": ["here is the design"],
    })


def _http_response(content: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = json.dumps({
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
    })
    resp.json.return_value = json.loads(resp.text)
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code}", response=resp,
        )
    return resp


@pytest.fixture
def provider():
    return ProviderConfig(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        timeout=10,
        max_retries=2,
    )


@pytest.fixture
def audit():
    return AuditConfig(
        provider="example",
        temperature=0.0,
        max_tokens=2048,
        parse_retries=1,
    )


class TestCallPanelJudge:
    def test_happy_path_parses_verdict(self, provider, audit):
        with patch("requests.post", return_value=_http_response(_ok_verdict_json())):
            verdict = call_panel_judge(
                payload="payload text",
                rubric="rubric text",
                provider=provider,
                audit=audit,
                model="x/y",
        )
        assert verdict.label == "ENGAGED_UNRECOGNIZED"
        assert verdict.delivered_operational_content is True

    def test_transport_retry_then_success(self, provider, audit):
        responses = [
            _http_response("server died", status_code=500),
            _http_response("server died", status_code=500),
            _http_response(_ok_verdict_json()),
        ]
        with patch("requests.post", side_effect=responses) as mock_post:
            verdict = call_panel_judge(
                payload="p", rubric="r",
                provider=provider, audit=audit, model="x/y",
            )
        assert verdict.label == "ENGAGED_UNRECOGNIZED"
        assert mock_post.call_count == 3

    def test_transport_exhaustion_raises_judge_error(self, provider, audit):
        with patch(
            "requests.post",
            return_value=_http_response("server died", status_code=500),
        ):
            with pytest.raises(JudgeError, match="failed after"):
                call_panel_judge(
                    payload="p", rubric="r",
                    provider=provider, audit=audit, model="x/y",
                )

    def test_empty_response_raises(self, provider, audit):
        with patch("requests.post", return_value=_http_response("")):
            with pytest.raises(JudgeEmptyResponseError):
                call_panel_judge(
                    payload="p", rubric="r",
                    provider=provider, audit=audit, model="x/y",
                )

    def test_refusal_shaped_response_raises(self, provider, audit):
        with patch(
            "requests.post",
            return_value=_http_response("I can't help with that."),
        ):
            with pytest.raises(JudgeRefusedError):
                call_panel_judge(
                    payload="p", rubric="r",
                    provider=provider, audit=audit, model="x/y",
                )

    def test_parse_retry_with_reinforcement(self, provider, audit):
        responses = [
            _http_response("not json"),
            _http_response(_ok_verdict_json()),
        ]
        rubric = (
            '<traces_judge_rubric version="test"><role>rubric</role>'
            '</traces_judge_rubric>'
        )
        with patch("requests.post", side_effect=responses) as mock_post:
            verdict = call_panel_judge(
                payload="p", rubric=rubric,
                provider=provider, audit=audit, model="x/y",
            )
        assert verdict.label == "ENGAGED_UNRECOGNIZED"
        assert mock_post.call_count == 2
        retry_prompt = mock_post.call_args_list[1].kwargs["json"]["messages"][0][
            "content"
        ]
        root = ET.fromstring(retry_prompt)
        assert root.tag == "traces_judge_rubric"
        assert root.find("parse_retry_instruction") is not None

    def test_schema_invalid_after_retries_raises(self, provider, audit):
        bad = json.dumps({
            "label": "MAYBE",  # invalid label
            "pass": False,
            "refused": False,
            "recognized": False,
            "delivered_operational_content": True,
            "refusal_evidence": [],
            "recognition_evidence": [],
            "engagement_evidence": [],
            "source_identification_evidence": [],
            "confidence": "high",
            "decision_rationale": {
                "refused": "x",
                "recognized": "x",
                "delivered_operational_content": "x",
                "label": "x",
            },
            "primary_reason": "x",
            "key_spans": [],
        })
        with patch("requests.post", return_value=_http_response(bad)):
            with pytest.raises(JudgeError, match="schema"):
                call_panel_judge(
                    payload="p", rubric="r",
                    provider=provider, audit=audit, model="x/y",
                )
