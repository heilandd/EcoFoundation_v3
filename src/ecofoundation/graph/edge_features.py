"""Edge features: distance + LR score (+ extensible tier-2 channels later)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecofoundation.config.schemas import GraphConfig


@dataclass(frozen=True)
class EdgeFeatures:
    """Stacked edge features for one niche."""

    matrix: np.ndarray  # (E, n_channels) float32
    channel_names: list[str]

    @property
    def n_channels(self) -> int:
        return self.matrix.shape[1]


def build_edge_features(
    *,
    distances: np.ndarray,
    lr_scores: np.ndarray | None,
    cfg: GraphConfig,
    distance_normalizer: float | None = None,
) -> EdgeFeatures:
    """Stack the configured edge-feature channels into a single matrix.

    Parameters
    ----------
    distances
        ``(E,)`` raw euclidean lengths in coordinate units (typically µm).
    lr_scores
        ``(E,)`` tier-1 LR score, or None to skip.
    distance_normalizer
        If set, divide distances by this value (e.g. patient-level median edge length).
    """
    channels: list[np.ndarray] = []
    names: list[str] = []

    if cfg.edge_feature_distance:
        d = distances.astype(np.float32)
        if cfg.edge_feature_normalize_distance and distance_normalizer:
            d = d / float(distance_normalizer)
        channels.append(d)
        names.append("distance_norm" if cfg.edge_feature_normalize_distance else "distance")

    if lr_scores is not None and cfg.lr_scoring.enabled:
        channels.append(lr_scores.astype(np.float32))
        names.append("lr_score_tier1")

    if not channels:
        # Always have at least one channel — fall back to a constant of 1.0
        channels.append(np.ones_like(distances, dtype=np.float32))
        names.append("constant")

    matrix = np.stack(channels, axis=1)
    return EdgeFeatures(matrix=matrix, channel_names=names)
