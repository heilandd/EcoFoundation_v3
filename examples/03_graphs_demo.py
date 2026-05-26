"""Niche + graph construction demo (Step 3).

Builds overlap-filtered niches plus per-niche PyG graphs (expression node
features, distance + tier-1 LR edge features) and writes an interactive report.

Usage::

    .venv/bin/python examples/03_graphs_demo.py
"""

from __future__ import annotations

from pathlib import Path

from ecofoundation.config.loader import load_config
from ecofoundation.pipelines.graphs import run_graph_construction_pipeline

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "graphs_demo.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    folder = run_graph_construction_pipeline(cfg)
    print()
    print("=" * 60)
    print(f"Run ID    : {folder.run_id}")
    print(f"Folder    : {folder.root}")
    print(f"Report    : {folder.report_path}")
    print(f"Graphs    : {folder.artifacts_dir / 'niche_graphs.pt'}")
    print(f"Metadata  : {folder.artifacts_dir / 'niche_graphs_metadata.parquet'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
