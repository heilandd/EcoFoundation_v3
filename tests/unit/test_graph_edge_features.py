"""Edge-feature channel stacking."""

from __future__ import annotations

import numpy as np

from ecofoundation.config.schemas import GraphConfig, LRScoringConfig
from ecofoundation.graph.edge_features import build_edge_features


def test_distance_plus_lr_channels():
    cfg = GraphConfig(
        edge_feature_distance=True,
        edge_feature_normalize_distance=False,
        lr_scoring=LRScoringConfig(enabled=True),
    )
    distances = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    lr_scores = np.array([0.5, 0.1, 0.9], dtype=np.float32)
    ef = build_edge_features(distances=distances, lr_scores=lr_scores, cfg=cfg)
    assert ef.matrix.shape == (3, 2)
    assert ef.channel_names == ["distance", "lr_score_tier1"]


def test_distance_only():
    cfg = GraphConfig(
        edge_feature_distance=True, lr_scoring=LRScoringConfig(enabled=False)
    )
    distances = np.array([1.0, 2.0])
    ef = build_edge_features(distances=distances, lr_scores=None, cfg=cfg)
    assert ef.matrix.shape == (2, 1)


def test_distance_normalization():
    cfg = GraphConfig(
        edge_feature_distance=True,
        edge_feature_normalize_distance=True,
        lr_scoring=LRScoringConfig(enabled=False),
    )
    distances = np.array([10.0, 20.0, 40.0])
    ef = build_edge_features(
        distances=distances, lr_scores=None, cfg=cfg, distance_normalizer=10.0
    )
    assert np.allclose(ef.matrix[:, 0], [1.0, 2.0, 4.0])


def test_fallback_constant_channel():
    cfg = GraphConfig(edge_feature_distance=False, lr_scoring=LRScoringConfig(enabled=False))
    distances = np.array([1.0, 2.0])
    ef = build_edge_features(distances=distances, lr_scores=None, cfg=cfg)
    assert ef.channel_names == ["constant"]
    assert np.allclose(ef.matrix, 1.0)
