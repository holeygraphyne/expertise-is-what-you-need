"""Gated smoke test: real NVIDIA NIM round-trip (skipped without env var).

Confirms the auditor's HTTP wiring works against the live endpoint
when NVIDIA_API_KEY is set. Intentionally minimal — one tiny request,
short max_tokens, no reasoning_effort — so it costs cents on CI when
the secret is configured, and is a no-op locally without the key.
"""
import os

import pytest

from traces.calibration.llm_client import call_chat_completion
from traces.config import ProviderConfig

API_KEY = os.environ.get("NVIDIA_API_KEY")


@pytest.mark.skipif(not API_KEY, reason="NVIDIA_API_KEY not set; skipping live test")
def test_nvidia_chat_completion_returns_content():
    """Sends ONE tiny request to NVIDIA NIM. Confirms auth + body shape."""
    provider = ProviderConfig(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=API_KEY,
        timeout=120,
    )
    content = call_chat_completion(
        provider=provider,
        model="deepseek-ai/deepseek-v4-pro",
        system_prompt="Reply with exactly: PONG",
        user_prompt="ping",
        temperature=0.0,
        max_tokens=64,
        top_p=0.95,
        reasoning_effort=None,   # smoke test — minimize latency
    )
    assert "PONG" in content.upper()
