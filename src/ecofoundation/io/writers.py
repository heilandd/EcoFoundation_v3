"""Run-folder management.

Every analysis run is persisted under ``runs/<timestamp>__<short_hash>/`` with:
  - config.yaml      — exact config used
  - manifest.json    — versions, seeds, input hash
  - report.html      — interactive report
  - log.txt          — full run log
  - artifacts/       — niche assignments, embeddings, model checkpoints
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ecofoundation.config.loader import dump_config
from ecofoundation.config.schemas import RunConfig
from ecofoundation.utils.hashing import hash_config, hash_file_head
from ecofoundation.utils.versioning import collect_versions


@dataclass(frozen=True)
class RunFolder:
    """Lightweight handle to a persisted run."""

    root: Path
    run_id: str

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def report_path(self) -> Path:
        return self.root / "report.html"

    @property
    def log_path(self) -> Path:
        return self.root / "log.txt"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def artifact(self, name: str) -> Path:
        """Return a path inside artifacts/ (parent dirs are pre-created)."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir / name


def create_run_folder(cfg: RunConfig) -> RunFolder:
    """Create a fresh run folder, persist config + manifest, return a handle."""
    cfg_hash = hash_config(cfg.model_dump(mode="json"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}__{cfg.run_name}__{cfg_hash}"
    root = Path(cfg.run_dir) / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "artifacts").mkdir(exist_ok=True)

    folder = RunFolder(root=root, run_id=run_id)

    # Persist config
    dump_config(cfg, folder.config_path)

    # Persist manifest
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": cfg_hash,
        "input_hash": _safe_input_hash(cfg.data.path),
        "versions": collect_versions(),
        "seed": cfg.seed,
        "device_preference": cfg.device,
    }
    with folder.manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return folder


def _safe_input_hash(path: Path) -> str | None:
    try:
        return hash_file_head(path)
    except FileNotFoundError:
        return None
