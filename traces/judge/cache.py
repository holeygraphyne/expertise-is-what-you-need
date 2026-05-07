"""On-disk cache for parallel-scorer judge verdicts.

Cache key = "{probe_id}|{model}|{response_sha256}|{case_sha256}|{judge_version}".
Errored entries (transient HTTP failures, parse failures we want to
retry) are filtered on load so a network blip in a previous run
does not become a permanent gap.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_key_for(
    *,
    probe_id: str,
    model: str,
    response_sha256: str,
    judge_version: str,
    case_sha256: str | None = None,
) -> str:
    if case_sha256:
        return f"{probe_id}|{model}|{response_sha256}|{case_sha256}|{judge_version}"
    return f"{probe_id}|{model}|{response_sha256}|{judge_version}"


def cache_key_has_case_hash(key: str) -> bool:
    """Return true for the current 5-component cache key shape."""
    return len(key.split("|")) == 5


def load_judge_cache(path: Path) -> dict[str, dict]:
    """Load the verdict cache, dropping errored rows.

    Errored rows match the shape `{"error": "<str>"}` and represent
    transient failures from a previous run. Successful verdicts are
    retained as-is.
    """
    if not path.exists():
        return {}
    try:
        all_entries: dict[str, dict] = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "judge_verdicts.json was malformed; starting fresh: %s", path,
        )
        return {}
    fresh: dict[str, dict] = {}
    dropped = 0
    for key, raw in all_entries.items():
        if isinstance(raw, dict) and "error" in raw and "panel_verdict" not in raw:
            dropped += 1
            continue
        fresh[key] = raw
    if dropped:
        logger.info(
            "judge_verdicts.json: dropped %d errored entr%s on load (will retry)",
            dropped, "y" if dropped == 1 else "ies",
        )
    return fresh


def save_judge_cache(path: Path, data: dict[str, dict]) -> None:
    """Atomically replace the cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)
