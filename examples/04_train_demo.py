"""Supervised GNN training demo (Step 4).

Trains a GINE classifier to predict the binary ``sonicated`` status from
niche graphs, with patient-level 5-fold CV.

Usage::

    .venv/bin/python examples/04_train_demo.py
"""

from __future__ import annotations

from pathlib import Path

from ecofoundation.config.loader import load_config
from ecofoundation.pipelines.train import run_training_pipeline

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "train_sonicated.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    folder = run_training_pipeline(cfg)
    print()
    print("=" * 60)
    print(f"Run ID    : {folder.run_id}")
    print(f"Folder    : {folder.root}")
    print(f"Report    : {folder.report_path}")
    print(f"Model     : {folder.artifacts_dir / 'model_fold0.pt'}")
    print(f"Test pred : {folder.artifacts_dir / 'test_predictions.parquet'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
