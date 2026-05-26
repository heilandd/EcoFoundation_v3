"""Explainability demo (Step 5).

Loads the trained sonicated-classifier from the Step-4 run and produces a
report with per-class top-gene importances, edge-feature attributions,
and spatial niche-graph overlays.

The Step-4 run must exist on disk — adjust SOURCE_RUN below if you have a
different timestamp.
"""

from __future__ import annotations

from pathlib import Path

from ecofoundation.config.loader import load_config
from ecofoundation.pipelines.explain import (
    ExplainPipelineInputs,
    run_explainability_pipeline,
)

# Adjust if you trained a different run
SOURCE_RUN = Path(
    "/Users/henrikheiland/Desktop/Coding/EcoFoundation/runs/"
    "20260525_225157__train_sonicated__b0149e3453"
)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "train_sonicated.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    cfg.run_name = "explain_sonicated"
    inputs = ExplainPipelineInputs(
        source_model_path=SOURCE_RUN / "artifacts" / "model_fold0.pt",
        source_predictions_path=SOURCE_RUN / "artifacts" / "test_predictions.parquet",
        target_name="sonicated",
        top_k_per_outcome=8,
        gnn_explainer_epochs=80,
        ig_steps=16,
    )
    folder = run_explainability_pipeline(cfg, inputs)
    print()
    print("=" * 60)
    print(f"Run ID    : {folder.run_id}")
    print(f"Folder    : {folder.root}")
    print(f"Report    : {folder.report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
