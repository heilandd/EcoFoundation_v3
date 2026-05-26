"""Marker-gene visualisations: dotplot and top-N heatmap (matplotlib)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.figure import Figure

from ecofoundation.reporting.style import new_figure, style_axes


def _natkey(s: str) -> tuple:
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def top_markers_heatmap(
    marker_table: pd.DataFrame,
    *,
    score_col: str = "score",
    n_top: int = 5,
    title: str = "Top marker scores per cluster",
) -> Figure:
    """Heatmap: rows = top genes per cluster, cols = clusters, values = score."""
    top = marker_table[marker_table["rank"] <= n_top]
    genes = pd.unique(top["gene"]).tolist()
    clusters = sorted(marker_table["cluster"].unique(), key=_natkey)
    matrix = (
        marker_table.pivot_table(
            index="gene", columns="cluster", values=score_col, aggfunc="max"
        )
        .reindex(index=genes, columns=clusters)
    )

    fig, ax = new_figure(
        width=max(3.5, 0.18 * len(clusters) + 1.6),
        height=max(2.0, 0.14 * len(genes) + 0.8),
    )
    z = matrix.values.astype(float)
    vmax = float(np.nanmax(np.abs(z))) if z.size else 1.0
    im = ax.imshow(
        z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label(score_col)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig


def marker_dotplot(
    adata: ad.AnnData,
    marker_table: pd.DataFrame,
    cluster_key: str,
    *,
    n_top: int = 3,
    layer: str | None = "X_exp",
    title: str = "Marker expression — mean & % expressing",
) -> Figure:
    """Dotplot: dot size = % cells expressing, colour = mean expression."""
    top = (
        marker_table[marker_table["rank"] <= n_top]
        .drop_duplicates(subset=["cluster", "gene"])
        .sort_values(["cluster", "rank"])
    )
    genes = list(pd.unique(top["gene"]))
    clusters = sorted(marker_table["cluster"].unique(), key=_natkey)
    var_ix = {g: i for i, g in enumerate(adata.var_names)}
    genes = [g for g in genes if g in var_ix]
    cols = [var_ix[g] for g in genes]

    X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    rows = []
    groups = adata.obs[cluster_key].astype(str)
    for cluster in clusters:
        mask = (groups == cluster).to_numpy()
        if mask.sum() == 0:
            continue
        sub = X[mask][:, cols]
        if sp.issparse(sub):
            mean = np.asarray(sub.mean(axis=0)).ravel()
            pct = np.asarray((sub > 0).mean(axis=0)).ravel()
        else:
            sub = np.asarray(sub)
            mean = sub.mean(axis=0)
            pct = (sub > 0).mean(axis=0)
        for j, g in enumerate(genes):
            rows.append({"cluster": cluster, "gene": g, "mean": float(mean[j]), "pct": float(pct[j])})

    df = pd.DataFrame(rows)
    if df.empty:
        fig, ax = new_figure(width=4, height=2)
        ax.set_title(title + " (no data)")
        style_axes(ax)
        return fig

    fig, ax = new_figure(
        width=max(3.5, 0.22 * len(genes) + 1.6),
        height=max(2.0, 0.22 * len(clusters) + 0.8),
    )
    sizes = (df["pct"] * 30).clip(2, 30)
    sc = ax.scatter(
        df["gene"], df["cluster"], s=sizes, c=df["mean"], cmap="viridis", edgecolors="white", linewidth=0.3
    )
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", labelrotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.set_title(title)
    ax.invert_yaxis()
    style_axes(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("Mean expression")
    cb.ax.tick_params(labelsize=5)
    fig.tight_layout()
    return fig
