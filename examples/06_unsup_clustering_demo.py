"""Unsupervised niche clustering demo (Step 6).

Trains a custom GAE on per-cell niches (no overlap filtering), pools to
per-niche embeddings, clusters with Leiden, and exports the result back
to a self-contained AnnData (``ecof_annotated.h5ad``).

To use DGI as an alternative architecture, change
``unsup.model.architecture: dgi`` in the config.

Usage::

    .venv/bin/python examples/06_unsup_clustering_demo.py
"""

from __future__ import annotations

from pathlib import Path

from ecofoundation.config.loader import load_config
from ecofoundation.pipelines.unsup_cluster import run_unsup_clustering_pipeline

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "unsup_clustering.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    folder = run_unsup_clustering_pipeline(cfg)
    print()
    print("=" * 60)
    print(f"Run ID    : {folder.run_id}")
    print(f"Folder    : {folder.root}")
    print(f"Report    : {folder.report_path}")
    print(f"AnnData   : {folder.artifacts_dir / 'ecof_annotated.h5ad'}")
    print(f"Embeddings: {folder.artifacts_dir / 'niche_embeddings.npz'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
