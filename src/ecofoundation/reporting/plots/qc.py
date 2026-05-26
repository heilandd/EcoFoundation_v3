"""QC plots: per-sample violins of counts and genes (matplotlib)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.figure import Figure

from ecofoundation.preprocessing.qc_stats import QCStats
from ecofoundation.reporting.style import new_figure, style_axes


def _stratified_subsample(
    df: pd.DataFrame, group_col: str, max_per_group: int, seed: int = 0
) -> pd.DataFrame:
    """Sample up to ``max_per_group`` rows per group."""
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, sub in df.groupby(group_col, observed=True, sort=False):
        if len(sub) > max_per_group:
            idx = rng.choice(len(sub), size=max_per_group, replace=False)
            parts.append(sub.iloc[idx])
        else:
            parts.append(sub)
    return pd.concat(parts, axis=0).reset_index(drop=True)


def qc_distributions_figure(
    adata, sample_col: str, qc_stats: QCStats, *, max_points_per_sample: int = 5_000
) -> Figure:
    """Two side-by-side violins (n_counts log-scale, n_genes), split by sample."""
    counts_layer = "counts" if "counts" in adata.layers else None
    X = adata.layers[counts_layer] if counts_layer else adata.X
    if sp.issparse(X):
        n_counts = np.asarray(X.sum(axis=1)).ravel()
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        X_d = np.asarray(X)
        n_counts = X_d.sum(axis=1)
        n_genes = (X_d > 0).sum(axis=1)

    df = pd.DataFrame(
        {
            "sample": adata.obs[sample_col].astype(str).to_numpy(),
            "n_counts": n_counts,
            "n_genes": n_genes,
        }
    )
    sampled = _stratified_subsample(df, "sample", max_points_per_sample)
    samples = sorted(sampled["sample"].unique())

    fig, axes = new_figure(width=6.5, height=2.6, nrows=1, ncols=2)
    ax_counts, ax_genes = axes[0], axes[1]

    counts_data = [sampled.loc[sampled["sample"] == s, "n_counts"].values for s in samples]
    genes_data = [sampled.loc[sampled["sample"] == s, "n_genes"].values for s in samples]

    for ax, data, ylabel in (
        (ax_counts, counts_data, "counts per cell"),
        (ax_genes, genes_data, "genes per cell"),
    ):
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor("#a5b4fc")
            body.set_edgecolor("black")
            body.set_linewidth(0.4)
            body.set_alpha(0.9)
        for key in ("cmedians", "cmaxes", "cmins", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.5)
        ax.set_xticks(range(1, len(samples) + 1))
        ax.set_xticklabels(samples, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        style_axes(ax)
    ax_counts.set_yscale("log")

    fig.suptitle(
        f"QC distributions (≤{max_points_per_sample}/sample · "
        f"global median counts={qc_stats.global_median_counts:.0f}, "
        f"genes={qc_stats.global_median_genes:.0f})",
    )
    fig.tight_layout()
    return fig
