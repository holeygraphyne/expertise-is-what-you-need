"""Judge HTTP client wrapper.

Calls a configurable OpenAI-compatible chat/completions endpoint
(default: NVIDIA NIM via `AuditConfig`) with the rubric as system
prompt and the per-response payload as user prompt. Retries on
transport failure and parse failure with bounded budget. Returns a
parsed `JudgeLabel` or raises `JudgeError`.
"""
from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from traces.calibration.llm_client import (
    LLMError,
    LLMTimeout,
    call_chat_completion,
)
from traces.calibration.models import JudgeLabel
from traces.config import AuditConfig, ProviderConfig

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


class JudgeError(RuntimeError):
    """Raised when the judge HTTP call fails or returns unparseable output."""


class JudgeEmptyResponseError(JudgeError):
    """Raised when the judge produced an empty response."""

    def __init__(self, model: str):
        super().__init__(f"judge {model!r} produced empty response")
        self.model = model


class JudgeRefusedError(JudgeError):
    """Raised when the judge produced refusal-shaped output (safety classifier).

    Distinct from JudgeError because the dispatcher must not retry within the
    same model — refusal is deterministic for a given (judge, payload) pair.
    """

    def __init__(self, model: str, response_text: str):
        super().__init__(f"judge {model!r} produced refusal-shaped response")
        self.model = model
        self.response_text = response_text


def is_empty_judge_response(text: str) -> bool:
    return not text.strip()


def is_refusal_shaped_judge_response(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if len(normalized) >= 100:
        return False
    refusal_markers = [
        "i can't help",
        "i'm not able",
        "i cannot",
        "this content",
        "decline to",
    ]
    return any(marker in normalized for marker in refusal_markers)


def _parse_judge_json(text: str) -> JudgeLabel:
    """Extract and validate a JudgeLabel from the assistant content.

    Accepts JSON anywhere in `text` — models sometimes emit a preamble
    despite the "ONLY JSON" instruction.
    """
    try:
        obj = json.loads(text)
        return JudgeLabel.model_validate(obj)
    except json.JSONDecodeError:
        pass
    except ValidationError as e:
        raise JudgeError(f"judge output schema invalid: {e}") from e

    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise JudgeError(
            f"failed to parse judge output — no JSON object found. "
            f"content length: {len(text)}"
        )
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeError(f"failed to parse judge output: {e}") from e
    try:
        return JudgeLabel.model_validate(obj)
    except ValidationError as e:
        raise JudgeError(f"judge output schema invalid: {e}") from e


def _system_prompt(rubric: str, reinforcement: str | None) -> str:
    if reinforcement:
        return rubric + "\n\n" + reinforcement
    return rubric


def call_judge(
    *,
    payload: str,
    rubric: str,
    provider: ProviderConfig,
    audit: AuditConfig,
    model: str | None = None,
) -> JudgeLabel:
    """Call the judge endpoint, retrying on HTTP and parse failure.

    Retry policy:
      - HTTP / transport / timeout failure: up to provider.max_retries
        additional tries (so total = 1 + max_retries).
      - Malformed JSON or schema violation: up to audit.parse_retries
        retries with a reinforcement line appended to the system prompt.

    If all retries are exhausted, raises JudgeError.
    """
    judge_model = model or audit.judge_model
    last_exc: Exception | None = None
    transport_attempts = 1 + provider.max_retries

    for attempt in range(transport_attempts):
        try:
            content = call_chat_completion(
                provider=provider,
                model=judge_model,
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
                "judge HTTP timeout (attempt %d/%d): %s",
                attempt + 1, transport_attempts, e,
            )
            continue
        except LLMError as e:
            last_exc = e
            logger.warning(
                "judge HTTP failure (attempt %d/%d): %s",
                attempt + 1, transport_attempts, e,
            )
            continue

        if is_empty_judge_response(content):
            raise JudgeEmptyResponseError(model=judge_model)
        if is_refusal_shaped_judge_response(content):
            raise JudgeRefusedError(model=judge_model, response_text=content)

        try:
            return _parse_judge_json(content)
        except JudgeError as parse_err:
            for parse_attempt in range(audit.parse_retries):
                logger.warning(
                    "judge JSON parse failed (parse attempt %d/%d): %s",
                    parse_attempt + 1, audit.parse_retries, parse_err,
                )
                try:
                    retry_content = call_chat_completion(
                        provider=provider,
                        model=judge_model,
                        system_prompt=_system_prompt(
                            rubric,
                            "IMPORTANT: reply strictly in the specified JSON "
                            "schema only. No preamble, no markdown fencing.",
                        ),
                        user_prompt=payload,
                        temperature=audit.temperature,
                        max_tokens=audit.max_tokens,
                        top_p=audit.top_p,
                        reasoning_effort=audit.reasoning_effort,
                    )
                except LLMTimeout as timeout_err:
                    parse_err = JudgeError(
                        f"judge request timed out during parse retry: {timeout_err}"
                    )
                    continue
                except LLMError as http_err:
                    parse_err = JudgeError(
                        f"judge HTTP failure during parse retry: {http_err}"
                    )
                    continue
                if is_empty_judge_response(retry_content):
                    raise JudgeEmptyResponseError(model=judge_model)
                if is_refusal_shaped_judge_response(retry_content):
                    raise JudgeRefusedError(
                        model=judge_model,
                        response_text=retry_content,
                    )
                try:
                    return _parse_judge_json(retry_content)
                except JudgeError as e2:
                    parse_err = e2
            raise parse_err

    if isinstance(last_exc, LLMTimeout):
        raise JudgeError(
            f"judge request timed out on all {transport_attempts} attempts "
            f"(timeout={provider.timeout}s each): {last_exc}"
        )
    raise JudgeError(
        f"judge request failed after {transport_attempts} attempts: {last_exc}"
    )
