"""Niche characterization stats."""

from __future__ import annotations

import numpy as np

from ecofoundation.config.schemas import NicheConfig
from ecofoundation.niches.assembly import assign_niches
from ecofoundation.niches.characterization import compute_niche_stats


def test_stats_shape(tiny_adata, tiny_run_config):
    cfg = NicheConfig(strategy="knn", knn_k=10, min_cells_per_niche=5)
    niches, _ = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    stats = compute_niche_stats(tiny_adata, niches, tiny_run_config.data)
    assert len(stats.per_niche) == niches.n_niches
    for col in (
        "size",
        "center_celltype",
        "shannon_entropy",
        "center_purity",
        "mean_nn_distance",
        "radius",
        "n_unique_celltypes",
    ):
        assert col in stats.per_niche.columns


def test_purity_bounds(tiny_adata, tiny_run_config):
    cfg = NicheConfig(strategy="knn", knn_k=10, min_cells_per_niche=5)
    niches, _ = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    stats = compute_niche_stats(tiny_adata, niches, tiny_run_config.data)
    p = stats.per_niche["center_purity"].to_numpy()
    assert (p >= 0).all() and (p <= 1).all()


def test_entropy_nonneg_and_bounded(tiny_adata, tiny_run_config):
    cfg = NicheConfig(strategy="knn", knn_k=10, min_cells_per_niche=5)
    niches, _ = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    stats = compute_niche_stats(tiny_adata, niches, tiny_run_config.data)
    e = stats.per_niche["shannon_entropy"].to_numpy()
    max_entropy = np.log(len(stats.cell_types))
    assert (e >= 0).all()
    assert (e <= max_entropy + 1e-9).all()


def test_cooccurrence_matrix_square_and_nonneg(tiny_adata, tiny_run_config):
    cfg = NicheConfig(strategy="knn", knn_k=10, min_cells_per_niche=5)
    niches, _ = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    stats = compute_niche_stats(tiny_adata, niches, tiny_run_config.data)
    co = stats.co_occurrence.values
    assert co.shape[0] == co.shape[1] == len(stats.cell_types)
    assert (co >= 0).all()


def test_density_positive(tiny_adata, tiny_run_config):
    cfg = NicheConfig(strategy="knn", knn_k=10, min_cells_per_niche=5)
    niches, _ = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    stats = compute_niche_stats(tiny_adata, niches, tiny_run_config.data)
    d = stats.per_niche["mean_nn_distance"].to_numpy()
    assert (d > 0).all()
