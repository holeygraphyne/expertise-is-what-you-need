"""Small corpus helpers shared across loaders and bootstrap stages."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(path: Path) -> str:
    """Compute a stable SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()