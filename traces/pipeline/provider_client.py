"""OpenAI-compatible provider HTTP client (chat completions).

One ProviderClient per provider entry in `TracesConfig.providers`.
The runner builds a `dict[str, ProviderClient]` keyed by provider name
and routes per-model calls through the right client.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from traces.calibration.llm_client import (
    LLMError,
    LLMHTTPError,
    extract_message_content,
    post_chat_completion,
)
from traces.config import ProviderConfig

logger = logging.getLogger(__name__)


class EmptyCompletionError(RuntimeError):
    """Raised when the provider returns a syntactically valid but unusable empty completion."""

    def __init__(self, message: str = "empty_completion"):
        super().__init__(message)


class ProviderHTTPError(RuntimeError):
    """Raised when the provider returns a non-429 HTTP error after retries."""

    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Provider HTTP {status_code}: {detail}")


class ThreadSafeRpmLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            now = time.monotonic()
            cutoff = now - 60.0
            with self._lock:
                self._timestamps = [t for t in self._timestamps if t >= cutoff]
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                wait_s = (self._timestamps[0] + 60.0) - now
            time.sleep(max(0.01, wait_s))


@dataclass
class CompletionResponse:
    model: str
    content: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class ProviderClient:
    """OpenAI-compatible chat-completions HTTP client for a single provider.

    Constructor takes a `ProviderConfig` (transport-level settings: base_url,
    api_key, timeout, retries) plus an optional rate limiter. Per-call knobs
    (model id, prompts, temperature, max_tokens, seed) are passed to
    `complete()`.
    """

    def __init__(
        self,
        provider: ProviderConfig,
        rpm_limiter: Optional[ThreadSafeRpmLimiter] = None,
    ):
        if not provider.base_url:
            raise ValueError("provider.base_url is required")
        self._provider = provider
        self._max_retries = provider.max_retries
        self._retry_delay = provider.retry_delay
        self._rpm_limiter = rpm_limiter

    def _retryable_wait(self, attempt: int, *, rate_limited: bool) -> None:
        wait = self._retry_delay * (attempt + 1) if rate_limited else self._retry_delay
        time.sleep(wait)

    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> CompletionResponse:
        if self._rpm_limiter is not None:
            self._rpm_limiter.acquire()

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        for attempt in range(self._max_retries):
            try:
                parsed, latency = post_chat_completion(
                    provider=self._provider,
                    body=payload,
                )
                response = self._parse(parsed, model, latency)
                self._raise_if_empty_completion(response)
                return response

            except EmptyCompletionError:
                logger.warning(
                    "Empty completion for %s (attempt %s/%s)",
                    model,
                    attempt + 1,
                    self._max_retries,
                )
                if attempt < self._max_retries - 1:
                    self._retryable_wait(attempt, rate_limited=False)
                    continue
                raise

            except LLMHTTPError as e:
                code = e.status_code
                detail = e.detail
                if code == 429:
                    wait = self._retry_delay * (attempt + 1)
                    logger.warning("Rate limited (429) for %s, waiting %ss", model, wait)
                    self._retryable_wait(attempt, rate_limited=True)
                    continue
                logger.error("Provider HTTP %s for %s: %s", code, model, detail)
                if attempt < self._max_retries - 1:
                    self._retryable_wait(attempt, rate_limited=False)
                    continue
                raise ProviderHTTPError(code, detail) from e

            except LLMError as e:
                logger.warning(
                    "Request failed for %s (attempt %s/%s): %s",
                    model,
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries - 1:
                    self._retryable_wait(attempt, rate_limited=False)
                    continue
                raise

        raise RuntimeError(f"Failed after {self._max_retries} attempts")

    def _parse(self, data: dict, model: str, latency: float) -> CompletionResponse:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        return CompletionResponse(
            model=data.get("model", model),
            content=extract_message_content(message.get("content", "")),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency,
        )

    @staticmethod
    def _raise_if_empty_completion(response: CompletionResponse) -> None:
        if response.completion_tokens == 0 and not response.content.strip():
            raise EmptyCompletionError("empty_completion: 0-token completion")
        if not response.content.strip():
            raise EmptyCompletionError("empty_completion: missing assistant content")
