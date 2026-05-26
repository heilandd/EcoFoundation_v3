"""YAML config loader — wraps Pydantic validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ecofoundation.config.schemas import RunConfig


def load_config(path: Path | str) -> RunConfig:
    """Load and validate a YAML config into a ``RunConfig``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return RunConfig.model_validate(raw)


def dump_config(cfg: RunConfig, path: Path | str) -> None:
    """Persist a ``RunConfig`` back to YAML (used in run-folder manifest)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json")
    with p.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
