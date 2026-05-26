"""Shared pytest fixtures.

A tiny synthetic AnnData fixture lets unit tests run in milliseconds without
touching the 6.6 GB real dataset.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ecofoundation.config.schemas import DataConfig, RunConfig


@pytest.fixture
def tiny_adata() -> ad.AnnData:
    """A 200-cell, 50-gene, 2-sample, 2-patient AnnData with spatial coords.

    Designed to exercise schema validation and downstream wiring; numerical
    quality is not the point.
    """
    rng = np.random.default_rng(0)
    n_cells, n_genes = 200, 50
    counts = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "samples": pd.Categorical(["S1"] * 100 + ["S2"] * 100),
            "patient": pd.Categorical(["P1"] * 100 + ["P2"] * 100),
            "sonicated": pd.Categorical(
                ["Nonsonicated"] * 100 + ["Sonicated"] * 100
            ),
            "celltype_level_1": pd.Categorical(rng.choice(["A", "B", "C"], size=n_cells)),
        }
    )
    obs.index = [f"cell_{i}" for i in range(n_cells)]
    var = pd.DataFrame(index=[f"GENE_{i}" for i in range(n_genes)])

    spatial = rng.uniform(0, 1000, size=(n_cells, 2))
    embedding = rng.standard_normal((n_cells, 10)).astype(np.float32)

    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.layers["X_exp"] = np.log1p(counts)
    adata.obsm["spatial"] = spatial
    adata.obsm["X_scVI"] = embedding
    return adata


@pytest.fixture
def tiny_adata_path(tiny_adata: ad.AnnData, tmp_path: Path) -> Path:
    """Write the tiny AnnData to a temp h5ad and return its path."""
    p = tmp_path / "tiny.h5ad"
    tiny_adata.write_h5ad(p)
    return p


@pytest.fixture
def tiny_run_config(tiny_adata_path: Path, tmp_path: Path) -> RunConfig:
    """A RunConfig pointing at the tiny dataset."""
    return RunConfig(
        run_name="test_run",
        run_dir=tmp_path / "runs",
        data=DataConfig(
            path=tiny_adata_path,
            sample_id_col="samples",
            patient_id_col="patient",
            condition_col="sonicated",
            celltype_col="celltype_level_1",
            spatial_key="spatial",
            counts_layer="counts",
            normalized_layer="X_exp",
            embedding_key="X_scVI",
        ),
    )
