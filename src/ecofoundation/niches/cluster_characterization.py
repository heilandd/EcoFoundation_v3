"""Characterise the unsupervised niche clusters.

For each cluster we summarise:

  - cell-type composition of the ego cells (which cell types dominate)
  - sample / patient distribution
  - average niche size, density, entropy (re-using per-niche stats)
  - top differentially expressed genes (ego-cell-level Wilcoxon vs the rest)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from ecofoundation.niches.base import NicheAssignment
from ecofoundation.niches.characterization import NicheStats
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class NicheClusterStats:
    composition: pd.DataFrame  # rows: cluster × cell_type → fraction
    size_distribution: pd.DataFrame  # per cluster
    sample_distribution: pd.DataFrame  # per cluster × sample
    markers: pd.DataFrame  # per cluster top genes (long form)
    cluster_summary: pd.DataFrame  # per cluster: n_niches, n_cells, avg density, avg entropy, top ct


def characterize_niche_clusters(
    adata: ad.AnnData,
    niches: NicheAssignment,
    *,
    cluster_labels: np.ndarray,
    celltype_col: str | None,
    sample_col: str,
    niche_stats: NicheStats | None = None,
    expression_layer: str = "X_exp",
    n_top_marker_genes: int = 15,
) -> NicheClusterStats:
    if niches.n_niches == 0 or cluster_labels.size == 0:
        empty = pd.DataFrame()
        return NicheClusterStats(empty, empty, empty, empty, empty)

    unique_clusters = sorted(np.unique(cluster_labels).tolist())
    samples = adata.obs[sample_col].astype(str).to_numpy() if sample_col in adata.obs.columns else None
    cts = (
        adata.obs[celltype_col].astype(str).to_numpy()
        if (celltype_col and celltype_col in adata.obs.columns)
        else None
    )

    # ----- composition -----------------------------------------------------
    comp_rows: list[dict] = []
    cluster_to_ego_cells: dict[int, np.ndarray] = {}
    for cl in unique_clusters:
        mask = cluster_labels == cl
        nids = np.flatnonzero(mask)
        ego_cells = niches.ego_cell[nids]
        cluster_to_ego_cells[cl] = ego_cells

        if cts is not None:
            n_ego = len(ego_cells)
            uniq, counts = np.unique(cts[ego_cells], return_counts=True)
            for ct, c in zip(uniq, counts, strict=True):
                comp_rows.append(
                    {
                        "cluster": cl,
                        "cell_type": str(ct),
                        "n": int(c),
                        "fraction": float(c) / max(n_ego, 1),
                    }
                )
    composition = pd.DataFrame(comp_rows)

    # ----- sample distribution --------------------------------------------
    sample_rows: list[dict] = []
    if samples is not None:
        for cl in unique_clusters:
            ego_cells = cluster_to_ego_cells[cl]
            uniq, counts = np.unique(samples[ego_cells], return_counts=True)
            for s, c in zip(uniq, counts, strict=True):
                sample_rows.append(
                    {"cluster": cl, "sample": str(s), "n_niches": int(c)}
                )
    sample_distribution = pd.DataFrame(sample_rows)

    # ----- size & density --------------------------------------------------
    sizes_all = niches.sizes()
    size_rows = []
    for cl in unique_clusters:
        nids = np.flatnonzero(cluster_labels == cl)
        size_rows.append(
            {
                "cluster": cl,
                "n_niches": int(len(nids)),
                "median_size": int(np.median(sizes_all[nids])) if len(nids) else 0,
                "mean_size": float(np.mean(sizes_all[nids])) if len(nids) else 0.0,
            }
        )
    size_distribution = pd.DataFrame(size_rows)

    # ----- marker genes via Wilcoxon on ego cells --------------------------
    markers = _compute_marker_genes_ego(
        adata, cluster_to_ego_cells, expression_layer=expression_layer, n_top=n_top_marker_genes
    )

    # ----- summary ---------------------------------------------------------
    summary_rows = []
    for cl in unique_clusters:
        nids = np.flatnonzero(cluster_labels == cl)
        row: dict = {"cluster": cl, "n_niches": int(len(nids))}
        if niche_stats is not None and not niche_stats.per_niche.empty:
            sub = niche_stats.per_niche.iloc[nids]
            row["median_density_nn"] = float(sub["mean_nn_distance"].median())
            row["median_entropy"] = float(sub["shannon_entropy"].median())
            row["median_purity"] = float(sub["center_purity"].median())
        if cts is not None and len(nids) > 0:
            ego_cells = cluster_to_ego_cells[cl]
            uniq, counts = np.unique(cts[ego_cells], return_counts=True)
            top = uniq[counts.argmax()]
            row["dominant_celltype"] = str(top)
            row["dominant_fraction"] = float(counts.max() / max(len(ego_cells), 1))
        summary_rows.append(row)
    cluster_summary = pd.DataFrame(summary_rows)

    return NicheClusterStats(
        composition=composition,
        size_distribution=size_distribution,
        sample_distribution=sample_distribution,
        markers=markers,
        cluster_summary=cluster_summary,
    )


def _compute_marker_genes_ego(
    adata: ad.AnnData,
    cluster_to_ego_cells: dict[int, np.ndarray],
    *,
    expression_layer: str,
    n_top: int,
) -> pd.DataFrame:
    """Run rank_genes_groups on ego cells grouped by niche cluster."""
    n_cells = adata.shape[0]
    cluster_labels = np.array(["__none__"] * n_cells, dtype=object)
    for cl, ego_cells in cluster_to_ego_cells.items():
        cluster_labels[ego_cells] = f"nc_{cl}"
    # Keep only ego cells of any cluster (others are dropped from the test).
    mask = cluster_labels != "__none__"
    sub = adata[mask].copy()
    sub.obs["__ecof_cluster__"] = pd.Categorical(cluster_labels[mask])

    if sub.shape[0] < 5 or sub.obs["__ecof_cluster__"].cat.categories.size < 2:
        return pd.DataFrame(columns=["cluster", "rank", "gene", "score"])

    try:
        sc.tl.rank_genes_groups(
            sub,
            groupby="__ecof_cluster__",
            method="wilcoxon",
            n_genes=n_top,
            layer=expression_layer if expression_layer in sub.layers else None,
            use_raw=False,
            key_added="ecof_markers",
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(f"rank_genes_groups failed for niche clusters: {e}")
        return pd.DataFrame(columns=["cluster", "rank", "gene", "score"])

    res = sub.uns["ecof_markers"]
    names = pd.DataFrame(res["names"])
    scores = pd.DataFrame(res["scores"])
    rows: list[dict] = []
    for cluster_name in names.columns:
        cl_int = int(cluster_name.replace("nc_", ""))
        for rank in range(min(n_top, len(names))):
            rows.append(
                {
                    "cluster": cl_int,
                    "rank": rank + 1,
                    "gene": names.iloc[rank][cluster_name],
                    "score": float(scores.iloc[rank][cluster_name]),
                }
            )
    return pd.DataFrame(rows)
