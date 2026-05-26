"""Classical clustering: Leiden on a kNN graph built over a low-dim representation."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import scanpy as sc

from ecofoundation.config.schemas import LeidenConfig
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)

LEIDEN_OBS_KEY = "ecof_leiden"
UMAP_OBSM_KEY = "ecof_umap"
NEIGHBORS_KEY = "ecof_neighbors"


@dataclass(frozen=True)
class LeidenResult:
    obs_key: str
    n_clusters: int
    resolution: float
    use_rep: str
    n_neighbors: int


def run_leiden(adata: ad.AnnData, cfg: LeidenConfig) -> LeidenResult:
    """Compute kNN graph + Leiden clusters in-place on ``adata``.

    Cluster labels are written to ``adata.obs[LEIDEN_OBS_KEY]`` (string-typed).
    UMAP embedding (computed if not present) is written to
    ``adata.obsm[UMAP_OBSM_KEY]``.

    Re-uses an existing UMAP in ``adata.obsm['X_umap']`` if present — useful
    when the user already shipped one (e.g. after scVI).
    """
    use_rep = cfg.use_rep
    if use_rep is not None and use_rep not in adata.obsm:
        _log.warning(f"obsm['{use_rep}'] missing; falling back to PCA.")
        use_rep = None

    _log.info(
        f"Leiden: building kNN graph (n_neighbors={cfg.n_neighbors}, use_rep={use_rep or 'PCA'})"
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=cfg.n_neighbors,
        use_rep=use_rep,
        random_state=cfg.random_state,
        key_added=NEIGHBORS_KEY,
    )

    _log.info(f"Leiden: clustering (resolution={cfg.resolution})")
    sc.tl.leiden(
        adata,
        resolution=cfg.resolution,
        random_state=cfg.random_state,
        key_added=LEIDEN_OBS_KEY,
        neighbors_key=NEIGHBORS_KEY,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    # Standardize dtype: scanpy may return Categorical; we keep as string for plots
    adata.obs[LEIDEN_OBS_KEY] = adata.obs[LEIDEN_OBS_KEY].astype(str)

    if "X_umap" in adata.obsm:
        _log.info("UMAP: re-using existing obsm['X_umap'].")
        adata.obsm[UMAP_OBSM_KEY] = adata.obsm["X_umap"]
    else:
        _log.info("UMAP: computing fresh embedding.")
        sc.tl.umap(adata, neighbors_key=NEIGHBORS_KEY, random_state=cfg.random_state)
        adata.obsm[UMAP_OBSM_KEY] = adata.obsm["X_umap"]

    n_clusters = int(adata.obs[LEIDEN_OBS_KEY].nunique())
    _log.info(f"Leiden: found {n_clusters} clusters")
    return LeidenResult(
        obs_key=LEIDEN_OBS_KEY,
        n_clusters=n_clusters,
        resolution=cfg.resolution,
        use_rep=use_rep or "PCA",
        n_neighbors=cfg.n_neighbors,
    )


def cluster_composition(
    adata: ad.AnnData,
    cluster_key: str,
    sample_key: str,
) -> "pd.DataFrame":
    """Long-format dataframe: sample × cluster → fraction of cells."""
    import pandas as pd

    counts = (
        adata.obs.groupby([sample_key, cluster_key], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    fractions = counts.div(counts.sum(axis=1), axis=0)
    df = fractions.reset_index().melt(
        id_vars=sample_key, var_name=cluster_key, value_name="fraction"
    )
    return df
