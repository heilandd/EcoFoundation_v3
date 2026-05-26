"""Deterministic hashing for run-IDs and input fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_config(obj: Any, *, length: int = 10) -> str:
    """SHA-256 hash of a JSON-serializable object, truncated to ``length`` chars.

    Used to make run-IDs reproducible from configs.
    """
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def hash_file_head(path: Path | str, *, n_bytes: int = 1_048_576, length: int = 10) -> str:
    """Hash the first ``n_bytes`` of a file. Fast fingerprint for huge AnnData files."""
    p = Path(path)
    h = hashlib.sha256()
    h.update(str(p.resolve()).encode())
    h.update(str(p.stat().st_size).encode())
    with p.open("rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:length]
