"""Smoke test: training pipeline on the tiny fixture (4 samples, 2 patients).

This trips the subgraph-split fallback (n_patients < 5). Goal: confirm the
end-to-end wiring compiles and produces metrics; not predictive accuracy.
"""

from __future__ import annotations

import pytest

from ecofoundation.config.schemas import (
    GraphConfig,
    LRScoringConfig,
    ModelConfig,
    NicheConfig,
    TargetSpec,
    TrainingConfig,
)
from ecofoundation.pipelines.train import run_training_pipeline


@pytest.mark.slow
def test_training_pipeline_runs(tiny_run_config):
    tiny_run_config.niches = NicheConfig(
        strategy="knn", knn_k=10, min_cells_per_niche=5,
        overlap_filter_enabled=True, max_overlap_fraction=0.5,
    )
    tiny_run_config.graph = GraphConfig(
        node_feature_source="embedding",
        node_embedding_key="X_scVI",
        edge_topology="knn_intra_niche",
        edge_knn_k=4,
        lr_scoring=LRScoringConfig(enabled=False),
    )
    tiny_run_config.model = ModelConfig(
        architecture="gine", hidden_dim=16, n_layers=2, dropout=0.0, pooling="mean",
        batch_norm=False, n_heads=2,
    )
    tiny_run_config.training = TrainingConfig(
        targets=[
            TargetSpec(
                name="sonicated", obs_column="sonicated",
                type="categorical", loss="cross_entropy", label_aggregation="ego",
            )
        ],
        batch_size=16,
        epochs=3,
        learning_rate=1e-3,
        early_stopping_patience=5,
        cv_strategy="subgraph_split",
        train_test_ratio=0.7,
        num_workers=0,
    )

    folder = run_training_pipeline(tiny_run_config)
    assert folder.report_path.exists()
    assert (folder.artifacts_dir / "test_predictions.parquet").exists()
    assert (folder.artifacts_dir / "model_fold0.pt").exists()

    html = folder.report_path.read_text()
    assert "Aggregated test metrics" in html
    assert "sonicated" in html
