"""Cache key for judge labels.

Key is `sha256(probe_id + '|' + model + '|' + response_text)[:16]`.
The separator prevents ambiguity across probe/model/response
boundaries.

Rationale: we cache judge labels so crash-recovery and incremental
expansion of the response set don't re-bill Opus for already-judged
responses. The key intentionally excludes the rubric version, judge
model, and payload template; see spec §Error handling for manual
invalidation rules.
"""
import hashlib


def cache_key(probe_id: str, model: str, response_text: str) -> str:
    """Return a 16-hex-char stable key for a (probe, model, response) tuple."""
    raw = f"{probe_id}|{model}|{response_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
