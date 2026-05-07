"""Per-judge HTTP wrapper for the blind benchmark judge panel.

Mirrors the retry semantics of `traces.calibration.judge.call_judge`
(transport retries via provider.max_retries, parse retries via
audit.parse_retries) but parses responses against the new
`JudgeVerdict` schema rather than the calibration `JudgeLabel`.

Reuses the error taxonomy from `traces.calibration.judge`
(`JudgeError`, `JudgeRefusedError`, `JudgeEmptyResponseError`) so
upstream callers can use a single exception hierarchy across both
the calibration audit and the parallel scorer.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET

from pydantic import ValidationError

from traces.calibration.judge import (
    JudgeEmptyResponseError,
    JudgeError,
    JudgeRefusedError,
    is_empty_judge_response,
    is_refusal_shaped_judge_response,
)
from traces.calibration.llm_client import (
    LLMError,
    LLMTimeout,
    call_chat_completion,
)
from traces.config import AuditConfig, ProviderConfig
from traces.judge.models import JudgeVerdict

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_panel_verdict(text: str) -> JudgeVerdict:
    """Extract and validate a JudgeVerdict from assistant content."""
    try:
        return JudgeVerdict.model_validate(json.loads(text))
    except json.JSONDecodeError:
        pass
    except ValidationError as e:
        raise JudgeError(f"judge output schema invalid: {e}") from e

    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise JudgeError(
            f"failed to parse judge output — no JSON object found "
            f"(content length: {len(text)})"
        )
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeError(f"failed to parse judge output: {e}") from e
    try:
        return JudgeVerdict.model_validate(obj)
    except ValidationError as e:
        raise JudgeError(f"judge output schema invalid: {e}") from e


def _system_prompt(rubric: str, reinforcement: str | None) -> str:
    if reinforcement:
        try:
            root = ET.fromstring(rubric)
        except ET.ParseError:
            return (
                rubric
                + "\n\n<parse_retry_instruction>"
                + reinforcement
                + "</parse_retry_instruction>"
            )
        ET.SubElement(root, "parse_retry_instruction").text = reinforcement
        return ET.tostring(root, encoding="unicode", short_empty_elements=False)
    return rubric


def call_panel_judge(
    *,
    payload: str,
    rubric: str,
    provider: ProviderConfig,
    audit: AuditConfig,
    model: str,
) -> JudgeVerdict:
    """Call one panel-member judge endpoint, retry on transport/parse failure.

    Total transport attempts = 1 + provider.max_retries.
    Total parse attempts = 1 + audit.parse_retries.

    Raises JudgeEmptyResponseError, JudgeRefusedError, or JudgeError
    on permanent failure.
    """
    last_exc: Exception | None = None
    transport_attempts = 1 + provider.max_retries

    for attempt in range(transport_attempts):
        try:
            content = call_chat_completion(
                provider=provider,
                model=model,
                system_prompt=_system_prompt(rubric, None),
                user_prompt=payload,
                temperature=audit.temperature,
                max_tokens=audit.max_tokens,
                top_p=audit.top_p,
                reasoning_effort=audit.reasoning_effort,
            )
        except LLMTimeout as e:
            last_exc = e
            logger.warning(
                "panel judge HTTP timeout (attempt %d/%d): %s",
                attempt + 1, transport_attempts, e,
            )
            continue
        except LLMError as e:
            last_exc = e
            logger.warning(
                "panel judge HTTP failure (attempt %d/%d): %s",
                attempt + 1, transport_attempts, e,
            )
            continue

        if is_empty_judge_response(content):
            raise JudgeEmptyResponseError(model=model)
        if is_refusal_shaped_judge_response(content):
            raise JudgeRefusedError(model=model, response_text=content)

        try:
            return _parse_panel_verdict(content)
        except JudgeError as parse_err:
            for parse_attempt in range(audit.parse_retries):
                logger.warning(
                    "panel judge JSON parse failed (parse %d/%d): %s",
                    parse_attempt + 1, audit.parse_retries, parse_err,
                )
                try:
                    retry_content = call_chat_completion(
                        provider=provider,
                        model=model,
                        system_prompt=_system_prompt(
                            rubric,
                            "Return exactly one JSON object matching the output "
                            "contract. Do not include markdown, prose, or XML.",
                        ),
                        user_prompt=payload,
                        temperature=audit.temperature,
                        max_tokens=audit.max_tokens,
                        top_p=audit.top_p,
                        reasoning_effort=audit.reasoning_effort,
                    )
                except LLMTimeout as timeout_err:
                    parse_err = JudgeError(
                        f"panel judge timed out during parse retry: {timeout_err}"
                    )
                    continue
                except LLMError as http_err:
                    parse_err = JudgeError(
                        f"panel judge HTTP failure during parse retry: {http_err}"
                    )
                    continue
                if is_empty_judge_response(retry_content):
                    raise JudgeEmptyResponseError(model=model)
                if is_refusal_shaped_judge_response(retry_content):
                    raise JudgeRefusedError(model=model, response_text=retry_content)
                try:
                    return _parse_panel_verdict(retry_content)
                except JudgeError as e2:
                    parse_err = e2
            raise parse_err

    if isinstance(last_exc, LLMTimeout):
        raise JudgeError(
            f"panel judge timed out on all {transport_attempts} attempts "
            f"(timeout={provider.timeout}s each): {last_exc}"
        )
    raise JudgeError(
        f"panel judge failed after {transport_attempts} attempts: {last_exc}"
    )
