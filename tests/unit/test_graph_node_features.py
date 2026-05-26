"""Node feature resolver: expression / embedding / concat + gene subsetting."""

from __future__ import annotations

import numpy as np
import pytest

from ecofoundation.config.schemas import GraphConfig
from ecofoundation.graph.node_features import build_node_features


def test_expression_default(tiny_adata):
    cfg = GraphConfig(node_feature_source="expression", node_expression_layer="X_exp")
    r = build_node_features(tiny_adata, cfg)
    assert r.source == "expression"
    assert r.n_features == tiny_adata.shape[1]
    sub = r.slice(np.array([0, 1, 2]))
    assert sub.shape == (3, tiny_adata.shape[1])
    assert sub.dtype == np.float32


def test_embedding_source(tiny_adata):
    cfg = GraphConfig(node_feature_source="embedding", node_embedding_key="X_scVI")
    r = build_node_features(tiny_adata, cfg)
    assert r.source == "embedding"
    assert r.n_features == tiny_adata.obsm["X_scVI"].shape[1]


def test_concat_source(tiny_adata):
    cfg = GraphConfig(node_feature_source="concat")
    r = build_node_features(tiny_adata, cfg)
    expected = tiny_adata.shape[1] + tiny_adata.obsm["X_scVI"].shape[1]
    assert r.n_features == expected


def test_hvg_subset(tiny_adata):
    cfg = GraphConfig(
        node_feature_source="expression", gene_subset="hvg", n_hvg=20
    )
    r = build_node_features(tiny_adata, cfg)
    assert r.n_features == 20


def test_custom_gene_subset(tiny_adata):
    pick = ["GENE_0", "GENE_5", "GENE_10"]
    cfg = GraphConfig(
        node_feature_source="expression", gene_subset="custom", custom_genes=pick
    )
    r = build_node_features(tiny_adata, cfg)
    assert r.n_features == len(pick)


def test_missing_embedding_raises(tiny_adata):
    cfg = GraphConfig(node_feature_source="embedding", node_embedding_key="not_a_key")
    with pytest.raises(KeyError):
        build_node_features(tiny_adata, cfg)
