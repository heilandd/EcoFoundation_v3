"""Marker-gene discovery per cluster via ``scanpy.tl.rank_genes_groups``."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import pandas as pd
import scanpy as sc

from ecofoundation.config.schemas import MarkerGenesConfig
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)

MARKERS_UNS_KEY = "ecof_markers"


@dataclass(frozen=True)
class MarkerResult:
    cluster_key: str
    method: str
    n_top: int
    table: pd.DataFrame  # columns: cluster, rank, gene, score, pvals_adj


def compute_marker_genes(
    adata: ad.AnnData,
    cluster_key: str,
    cfg: MarkerGenesConfig,
    *,
    layer: str | None = None,
) -> MarkerResult:
    """Run rank_genes_groups and flatten the result into a long dataframe.

    Parameters
    ----------
    adata
        Annotated data with cluster labels in ``obs[cluster_key]``.
    cluster_key
        Column in ``obs`` to group by.
    cfg
        :class:`MarkerGenesConfig` (method + n_top).
    layer
        Expression layer to use; ``None`` uses ``adata.X``. For the user's
        Xenium dataset this should typically be ``"X_exp"`` (normalized).
    """
    _log.info(
        f"rank_genes_groups: method={cfg.method} groupby={cluster_key} layer={layer or 'X'}"
    )
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method=cfg.method,
        n_genes=cfg.n_top,
        layer=layer,
        key_added=MARKERS_UNS_KEY,
    )

    raw = adata.uns[MARKERS_UNS_KEY]
    names = pd.DataFrame(raw["names"])
    scores = pd.DataFrame(raw["scores"])
    pvals_adj = pd.DataFrame(raw["pvals_adj"]) if "pvals_adj" in raw else None
    logfc = pd.DataFrame(raw["logfoldchanges"]) if "logfoldchanges" in raw else None

    rows = []
    for cluster in names.columns:
        for rank in range(min(cfg.n_top, len(names))):
            row = {
                "cluster": cluster,
                "rank": rank + 1,
                "gene": names.iloc[rank][cluster],
                "score": float(scores.iloc[rank][cluster]),
            }
            if pvals_adj is not None:
                row["pvals_adj"] = float(pvals_adj.iloc[rank][cluster])
            if logfc is not None:
                row["logfoldchange"] = float(logfc.iloc[rank][cluster])
            rows.append(row)

    table = pd.DataFrame(rows)
    _log.info(f"Markers: {len(table)} rows across {table['cluster'].nunique()} clusters")
    return MarkerResult(
        cluster_key=cluster_key,
        method=cfg.method,
        n_top=cfg.n_top,
        table=table,
    )
