"""Niche-construction demo (Step 2).

Builds Delaunay-3hop niches on the full Xenium dataset, applies the overlap
controller, and writes a self-contained interactive report.

Usage::

    .venv/bin/python examples/02_niches_demo.py
"""

from __future__ import annotations

from pathlib import Path

from ecofoundation.config.loader import load_config
from ecofoundation.pipelines.niches import run_niche_pipeline

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "niches_demo.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    folder = run_niche_pipeline(cfg)
    print()
    print("=" * 60)
    print(f"Run ID    : {folder.run_id}")
    print(f"Folder    : {folder.root}")
    print(f"Report    : {folder.report_path}")
    print(f"Manifest  : {folder.manifest_path}")
    print(f"Niches    : {folder.artifacts_dir / 'niche_assignment.parquet'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
