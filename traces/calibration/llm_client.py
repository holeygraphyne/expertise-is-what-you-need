"""Shared OpenAI-compatible chat/completions HTTP transport.

Wraps an OpenAI-compatible /v1/chat/completions endpoint. Transport fields
(base_url, api_key, timeout) come from a `ProviderConfig`; per-call body
fields are explicit kwargs or a prebuilt request body.

`post_chat_completion` is the single HTTP implementation used by both the
benchmark `ProviderClient` and calibration/judge tooling. Retry semantics
stay in the callers because they differ by workflow.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import requests

from traces.config import ProviderConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Transport, HTTP, or response-shape failure from the LLM endpoint."""


class LLMTimeout(LLMError):
    """The request exceeded provider.timeout seconds."""


class LLMHTTPError(LLMError):
    """Non-200 HTTP response from the provider."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"chat-completion HTTP {status_code}: {detail!r}")


def _headers(provider: ProviderConfig) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://traces-benchmark.org",
        "X-Title": "TRACES Benchmark",
        # Groq's Cloudflare blocks the stdlib default UA; keep an explicit
        # identity on every provider request.
        "User-Agent": "TRACES-Benchmark/1.0 (https://traces-benchmark.org)",
    }
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    return headers


def extract_message_content(content: Any) -> str:
    """Normalize OpenAI-compatible message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def post_chat_completion(
    *,
    provider: ProviderConfig,
    body: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """POST one chat-completion request and return `(json, latency_ms)`."""
    url = provider.base_url.rstrip("/") + "/chat/completions"
    try:
        t0 = time.time()
        resp = requests.post(
            url, json=body, headers=_headers(provider), timeout=provider.timeout
        )
        latency_ms = (time.time() - t0) * 1000
    except requests.Timeout as e:
        raise LLMTimeout(
            f"chat-completion request timed out after {provider.timeout}s"
        ) from e
    except requests.RequestException as e:
        raise LLMError(f"chat-completion request failed: {e}") from e

    if resp.status_code != 200:
        raise LLMHTTPError(resp.status_code, resp.text[:500])

    try:
        return resp.json(), latency_ms
    except json.JSONDecodeError as e:
        raise LLMError(f"chat-completion response was not JSON: {e}") from e


def call_chat_completion(
    *,
    provider: ProviderConfig,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """POST one chat-completion request and return the assistant content.

    Transport (url, auth, timeout) comes from the provider; per-call body
    fields are explicit kwargs.

    Raises LLMTimeout on socket timeout, LLMError on any other transport
    error, non-200 response, or response missing choices[0].message.content.
    """
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    data, _latency_ms = post_chat_completion(provider=provider, body=body)

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(
            f"chat-completion response missing choices[0].message.content; "
            f"got: {str(data)[:500]!r}"
        ) from e
    return extract_message_content(content)
