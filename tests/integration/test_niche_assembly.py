"""Integration: end-to-end niche assembly through ``assign_niches``."""

from __future__ import annotations

import numpy as np
import pytest

from ecofoundation.config.schemas import NicheConfig
from ecofoundation.niches.assembly import assign_niches


def test_assign_delaunay_default(tiny_run_config, tiny_adata):
    niche_cfg = NicheConfig(strategy="delaunay", k_hop=2, min_cells_per_niche=3)
    filtered, info = assign_niches(tiny_adata, tiny_run_config.data, niche_cfg)
    assert filtered.n_niches > 0
    # patient invariant
    for nid in range(filtered.n_niches):
        cells = filtered.cells_per_niche[nid]
        patients = set(tiny_adata.obs["patient"].astype(str).to_numpy()[cells])
        assert len(patients) == 1
    # overlap invariant
    for i in range(filtered.n_niches):
        for j in range(i + 1, filtered.n_niches):
            if filtered.group_label[i] != filtered.group_label[j]:
                continue
            assert filtered.jaccard(i, j) <= niche_cfg.max_overlap_fraction + 1e-9


def test_assign_knn(tiny_run_config, tiny_adata):
    niche_cfg = NicheConfig(strategy="knn", knn_k=10, min_cells_per_niche=5)
    filtered, _ = assign_niches(tiny_adata, tiny_run_config.data, niche_cfg)
    assert filtered.n_niches > 0


def test_assign_radius(tiny_run_config, tiny_adata):
    niche_cfg = NicheConfig(strategy="radius", radius=400.0, min_cells_per_niche=3)
    filtered, _ = assign_niches(tiny_adata, tiny_run_config.data, niche_cfg)
    assert filtered.n_niches > 0


def test_assign_tiling(tiny_run_config, tiny_adata):
    niche_cfg = NicheConfig(strategy="tiling", radius=200.0, min_cells_per_niche=3)
    filtered, _ = assign_niches(tiny_adata, tiny_run_config.data, niche_cfg)
    # Tiling guarantees disjoint cells WITHIN a group, so jaccard within group should be 0
    for i in range(filtered.n_niches):
        for j in range(i + 1, filtered.n_niches):
            if filtered.group_label[i] == filtered.group_label[j]:
                assert filtered.jaccard(i, j) == 0.0


def test_missing_patient_col_raises(tiny_run_config, tiny_adata):
    tiny_run_config.data.patient_id_col = "nonexistent_col"
    niche_cfg = NicheConfig()
    with pytest.raises(ValueError, match="patient_id_col"):
        assign_niches(tiny_adata, tiny_run_config.data, niche_cfg)


def test_summary_dict_shape(tiny_run_config, tiny_adata):
    niche_cfg = NicheConfig(strategy="delaunay", k_hop=2, min_cells_per_niche=3)
    filtered, _ = assign_niches(tiny_adata, tiny_run_config.data, niche_cfg)
    s = filtered.summary()
    assert s["n_niches"] == filtered.n_niches
    assert "median_cells_per_niche" in s
    assert s["params"]["k_hop"] == 2
