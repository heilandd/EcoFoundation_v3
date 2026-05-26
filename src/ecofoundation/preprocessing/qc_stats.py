"""QC statistics — descriptive only.

Even when ``QCConfig.enabled=False`` (data already preprocessed), we still
compute and display these stats in the report so the user has a per-sample
sanity baseline. Filtering is a separate concern, lives in qc_filter (Step 2+).
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ecofoundation.config.schemas import DataConfig
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class QCStats:
    """Per-sample QC summary."""

    per_sample: pd.DataFrame
    global_total: int
    global_median_counts: float
    global_median_genes: float

    def to_records(self) -> list[dict]:
        return self.per_sample.reset_index().to_dict("records")


def compute_qc_stats(adata: ad.AnnData, data_cfg: DataConfig) -> QCStats:
    """Per-sample counts / genes-per-cell summary.

    Counts source:
      1. ``adata.layers[data_cfg.counts_layer]`` if present
      2. else ``adata.X``
    """
    sample_col = data_cfg.sample_id_col
    if sample_col not in adata.obs.columns:
        raise KeyError(f"obs['{sample_col}'] missing")

    counts_mat = _resolve_counts(adata, data_cfg.counts_layer)
    n_counts = np.asarray(counts_mat.sum(axis=1)).ravel()
    n_genes = np.asarray((counts_mat > 0).sum(axis=1)).ravel()

    df = pd.DataFrame(
        {
            "sample": adata.obs[sample_col].to_numpy(),
            "n_counts": n_counts,
            "n_genes": n_genes,
        }
    )
    per_sample = (
        df.groupby("sample", observed=True)
        .agg(
            n_cells=("n_counts", "size"),
            median_counts=("n_counts", "median"),
            mean_counts=("n_counts", "mean"),
            median_genes=("n_genes", "median"),
            mean_genes=("n_genes", "mean"),
        )
        .round(2)
    )
    _log.info(f"QC stats computed over {len(adata)} cells in {len(per_sample)} samples")
    return QCStats(
        per_sample=per_sample,
        global_total=int(adata.shape[0]),
        global_median_counts=float(np.median(n_counts)),
        global_median_genes=float(np.median(n_genes)),
    )


def _resolve_counts(adata: ad.AnnData, layer: str | None):
    if layer is not None and layer in adata.layers:
        return adata.layers[layer]
    if sp.issparse(adata.X) or isinstance(adata.X, np.ndarray):
        return adata.X
    return np.asarray(adata.X)
