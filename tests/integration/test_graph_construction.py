"""Integration: build_niche_graphs end-to-end on the tiny dataset."""

from __future__ import annotations

import numpy as np
import torch

from ecofoundation.config.schemas import GraphConfig, LRScoringConfig, NicheConfig
from ecofoundation.graph.construction import build_niche_graphs
from ecofoundation.niches.assembly import assign_niches


def _make_niches(tiny_adata, tiny_run_config):
    niche_cfg = NicheConfig(strategy="delaunay", k_hop=2, min_cells_per_niche=4)
    return assign_niches(tiny_adata, tiny_run_config.data, niche_cfg)[0]


def test_build_niche_graphs_smoke(tiny_adata, tiny_run_config):
    niches = _make_niches(tiny_adata, tiny_run_config)
    cfg = GraphConfig(
        node_feature_source="embedding",
        node_embedding_key="X_scVI",
        lr_scoring=LRScoringConfig(enabled=False),
    )
    result = build_niche_graphs(tiny_adata, niches, tiny_run_config.data, cfg)
    assert len(result.graphs) == niches.n_niches
    assert all(isinstance(g.x, torch.Tensor) for g in result.graphs)
    # Every node feature dim equals the resolver dim
    expected_dim = tiny_adata.obsm["X_scVI"].shape[1]
    for g in result.graphs:
        assert g.x.shape[1] == expected_dim
        assert g.edge_index.shape[0] == 2
        # both-directions packed
        assert g.edge_attr.shape[0] == g.edge_index.shape[1]
        # niche metadata round-trips
        assert hasattr(g, "niche_id")
        assert hasattr(g, "patient")


def test_graph_no_cross_patient_in_metadata(tiny_adata, tiny_run_config):
    niches = _make_niches(tiny_adata, tiny_run_config)
    cfg = GraphConfig(
        node_feature_source="embedding",
        lr_scoring=LRScoringConfig(enabled=False),
    )
    result = build_niche_graphs(tiny_adata, niches, tiny_run_config.data, cfg)
    patients = tiny_adata.obs["patient"].astype(str).to_numpy()
    for g in result.graphs:
        cells = g.global_cell_indices.cpu().numpy()
        assert set(patients[cells]) == {g.patient}


def test_graph_edge_distances_match_node_coords(tiny_adata, tiny_run_config):
    niches = _make_niches(tiny_adata, tiny_run_config)
    cfg = GraphConfig(
        node_feature_source="embedding",
        edge_feature_normalize_distance=False,
        lr_scoring=LRScoringConfig(enabled=False),
    )
    result = build_niche_graphs(tiny_adata, niches, tiny_run_config.data, cfg)
    coords = np.asarray(tiny_adata.obsm["spatial"])[:, :2]
    for g in result.graphs[:5]:
        cells = g.global_cell_indices.cpu().numpy()
        ei = g.edge_index.cpu().numpy()
        ea = g.edge_attr.cpu().numpy()
        for k in range(min(20, ei.shape[1])):
            a_local, b_local = int(ei[0, k]), int(ei[1, k])
            a_glob, b_glob = cells[a_local], cells[b_local]
            expected = np.linalg.norm(coords[a_glob] - coords[b_glob])
            assert np.isclose(ea[k, 0], expected, atol=1e-4)
