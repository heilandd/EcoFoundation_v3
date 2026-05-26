"""AnnData export of unsupervised results."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from ecofoundation.config.schemas import NicheConfig
from ecofoundation.io.anndata_export import (
    UnsupExportInputs,
    export_unsup_results_to_anndata,
)
from ecofoundation.niches.assembly import assign_niches


def test_export_round_trip(tiny_adata, tiny_run_config, tmp_path):
    cfg = NicheConfig(strategy="knn", knn_k=10, min_cells_per_niche=5)
    niches, _ = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    n = niches.n_niches
    embeddings = np.random.RandomState(0).randn(n, 16).astype(np.float32)
    umap_2d = np.random.RandomState(0).randn(n, 2).astype(np.float32)
    labels = np.random.RandomState(0).randint(0, 3, size=n)
    out = export_unsup_results_to_anndata(
        tiny_adata,
        UnsupExportInputs(
            niches=niches,
            niche_embeddings=embeddings,
            niche_umap_2d=umap_2d,
            niche_cluster_labels=labels,
            run_id="test_run",
            config_hash="abc",
        ),
        out_path=tmp_path / "out.h5ad",
        write_compressed=True,
    )
    assert out.exists()

    a = ad.read_h5ad(out)
    assert "ecof_niche_cluster" in a.obs.columns
    assert "ecof_niche_id" in a.obs.columns
    assert "ecof_niche_embedding" in a.obsm
    assert a.obsm["ecof_niche_embedding"].shape == (a.shape[0], 16)
    assert "ecof_niche_umap" in a.obsm
    assert "ecof_run" in a.uns
    assert a.uns["ecof_run"]["run_id"] == "test_run"
    # Every niche's ego cell should have a non-default cluster.
    annotated_ego = (a.obs["ecof_niche_id"] >= 0).sum()
    assert annotated_ego >= n  # cells equal niches or more if dups
