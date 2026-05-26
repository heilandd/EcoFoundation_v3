"""Niche-characterisation plots (matplotlib)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.niches.characterization import NicheStats
from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes


def co_occurrence_heatmap(
    stats: NicheStats,
    *,
    normalize: str = "center",  # "center" rows or "none"
    log: bool = True,
    title: str = "Center vs neighbor cell-type co-occurrence",
) -> Figure:
    """Heatmap: rows = center cell type, cols = neighbor cell type, values = frequency."""
    co = stats.co_occurrence.astype(float).copy()
    if normalize == "center":
        row_sums = co.sum(axis=1).replace(0, np.nan)
        co = co.div(row_sums, axis=0).fillna(0.0)
    z = co.values
    if log:
        z = np.log1p(z)

    fig, ax = new_figure(
        width=max(3.5, 0.18 * len(stats.cell_types) + 1.8),
        height=max(3.0, 0.16 * len(stats.cell_types) + 1.0),
    )
    im = ax.imshow(z, cmap="magma", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(stats.cell_types)))
    ax.set_xticklabels(stats.cell_types, rotation=60, ha="right")
    ax.set_yticks(range(len(stats.cell_types)))
    ax.set_yticklabels(stats.cell_types)
    ax.set_xlabel("Neighbor cell type")
    ax.set_ylabel("Center cell type")
    ax.set_title(title + (" (row-normalised)" if normalize == "center" else "") + (" — log1p" if log else ""))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig


def niche_density_histogram(stats: NicheStats, *, title: str = "Cellular density per niche") -> Figure:
    """Histogram of mean-nearest-neighbor distance per niche (proxy for density)."""
    fig, ax = new_figure(width=4.6, height=2.4)
    d = stats.per_niche["mean_nn_distance"].to_numpy()
    if d.size == 0:
        ax.set_title(title + " (empty)")
        style_axes(ax)
        return fig
    ax.hist(d, bins=50, color="#a5b4fc", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Mean nearest-neighbor distance within niche (coord units)")
    ax.set_ylabel("# niches")
    ax.set_title(title + f"   (n={len(d)}, median={np.median(d):.1f})")
    style_axes(ax)
    fig.tight_layout()
    return fig


def heterogeneity_histogram(stats: NicheStats) -> Figure:
    """Shannon-entropy distribution of niche cell-type composition."""
    fig, ax = new_figure(width=4.6, height=2.4)
    e = stats.per_niche["shannon_entropy"].to_numpy()
    if e.size == 0:
        ax.set_title("Shannon entropy (empty)")
        style_axes(ax)
        return fig
    ax.hist(e, bins=50, color="#fdba74", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Shannon entropy of cell-type composition")
    ax.set_ylabel("# niches")
    ax.set_title(f"Niche heterogeneity (median={np.median(e):.2f}, max={np.log(len(stats.cell_types)):.2f})")
    style_axes(ax)
    fig.tight_layout()
    return fig


def center_purity_by_celltype(stats: NicheStats, *, max_types: int = 25) -> Figure:
    """Box-plot of center-type purity, one box per center cell type."""
    d = stats.per_niche
    if d.empty:
        fig, ax = new_figure(width=3, height=2)
        ax.set_title("Center purity (empty)")
        style_axes(ax)
        return fig
    counts = d["center_celltype"].value_counts()
    top_types = counts.head(max_types).index.tolist()
    sub = d[d["center_celltype"].isin(top_types)]
    grouped = [sub.loc[sub["center_celltype"] == t, "center_purity"].to_numpy() for t in top_types]

    fig, ax = new_figure(width=max(4.0, 0.24 * len(top_types) + 1.6), height=2.6)
    bp = ax.boxplot(
        grouped,
        showfliers=False,
        widths=0.65,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 0.6},
    )
    cmap = _build_palette(len(top_types))
    for patch, color in zip(bp["boxes"], cmap, strict=False):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.4)
    ax.set_xticks(range(1, len(top_types) + 1))
    ax.set_xticklabels(top_types, rotation=45, ha="right")
    ax.set_ylabel("center purity")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Center-cell-type purity (top {len(top_types)} center types)")
    style_axes(ax)
    fig.tight_layout()
    return fig


def size_vs_density_scatter(stats: NicheStats) -> Figure:
    """Niche size vs density per niche — diagnostic plot."""
    d = stats.per_niche
    fig, ax = new_figure(width=4.4, height=3.2)
    if d.empty:
        ax.set_title("size vs density (empty)")
        style_axes(ax)
        return fig
    ax.scatter(
        d["size"], d["mean_nn_distance"],
        s=1.2, c="#2563eb", alpha=0.4, linewidth=0, rasterized=True,
    )
    ax.set_xlabel("cells per niche")
    ax.set_ylabel("mean NN distance (coord units)")
    ax.set_title("Niche size vs cellular density")
    style_axes(ax)
    fig.tight_layout()
    return fig


def n_unique_celltypes_histogram(stats: NicheStats) -> Figure:
    """Distribution of how many distinct cell types each niche contains."""
    d = stats.per_niche["n_unique_celltypes"].to_numpy()
    fig, ax = new_figure(width=4.0, height=2.2)
    if d.size == 0:
        ax.set_title("# unique cell types per niche (empty)")
        style_axes(ax)
        return fig
    bins = np.arange(d.min(), d.max() + 2) - 0.5
    ax.hist(d, bins=bins, color="#86efac", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("# distinct cell types in niche")
    ax.set_ylabel("# niches")
    ax.set_title(f"Niche diversity (median={int(np.median(d))})")
    style_axes(ax)
    fig.tight_layout()
    return fig
