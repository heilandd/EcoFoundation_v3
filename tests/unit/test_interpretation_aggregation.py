"""ClassAttribution aggregation logic."""

from __future__ import annotations

import numpy as np

from ecofoundation.interpretation.aggregation import aggregate_explanations
from ecofoundation.interpretation.gnn_explainer import NicheExplanation
from ecofoundation.interpretation.integrated_gradients import IGResult


def test_aggregate_empty():
    attr = aggregate_explanations(
        [], [], feature_names=["A", "B"], edge_channel_names=["d"], target_class=0
    )
    assert attr.n_niches_explained == 0
    assert attr.gene_importance.empty
    assert attr.edge_channel_importance.empty


def test_aggregate_orders_genes_by_abs_attr():
    n_features = 4
    feature_names = [f"G{i}" for i in range(n_features)]
    # niche 0: feature 2 dominates
    ig1 = IGResult(
        niche_id=0,
        target_class=0,
        node_attrs=np.array([[0.01, 0.01, 1.0, 0.01], [0.01, 0.0, 0.9, 0.0]]),
        edge_attr_attrs=None,
    )
    # niche 1: same
    ig2 = IGResult(
        niche_id=1,
        target_class=0,
        node_attrs=np.array([[0.0, 0.0, 0.8, 0.0]]),
        edge_attr_attrs=None,
    )
    explanations = [
        NicheExplanation(niche_id=0, target_class=0,
                         node_mask=np.zeros((1, n_features)),
                         edge_mask=np.zeros(0), target_prob=0.5),
    ]
    attr = aggregate_explanations(
        explanations, [ig1, ig2],
        feature_names=feature_names, edge_channel_names=["d"], target_class=0,
    )
    assert attr.gene_importance.iloc[0]["gene"] == "G2"


def test_aggregate_edge_channel_importance():
    feature_names = ["A"]
    channels = ["distance", "lr_score"]
    igs = [
        IGResult(
            niche_id=i, target_class=0,
            node_attrs=np.zeros((2, 1)),
            edge_attr_attrs=np.array([[0.1, 1.0], [0.05, 0.9]]),
        )
        for i in range(3)
    ]
    attr = aggregate_explanations(
        [], igs, feature_names=feature_names, edge_channel_names=channels, target_class=0
    )
    assert attr.edge_channel_importance.iloc[0]["channel"] == "lr_score"
