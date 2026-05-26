"""Plots for the unsupervised niche-clustering pipeline."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.niches.base import NicheAssignment
from ecofoundation.niches.cluster_characterization import NicheClusterStats
from ecofoundation.reporting.plots.qc import _stratified_subsample
from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes


def unsup_training_loss(history: list[dict]) -> Figure:
    fig, ax = new_figure(width=4.4, height=2.4)
    if not history:
        ax.set_title("Unsupervised training (no data)")
        style_axes(ax)
        return fig
    df = pd.DataFrame(history)
    ax.plot(df["epoch"], df["loss"], color="#2563eb", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Unsupervised GNN training loss")
    style_axes(ax)
    fig.tight_layout()
    return fig


def niche_cluster_embedding_umap(
    umap_coords: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    title: str = "Niche embedding UMAP (Leiden clusters)",
) -> Figure:
    fig, ax = new_figure(width=5.0, height=4.0)
    clusters = sorted(np.unique(cluster_labels).tolist())
    cmap = _build_palette(len(clusters))
    for i, cl in enumerate(clusters):
        mask = cluster_labels == cl
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            s=2.0, c=[cmap[i]], alpha=0.7, linewidth=0, rasterized=True,
            label=f"nc_{cl}",
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    if len(clusters) <= 30:
        ax.legend(
            markerscale=3, loc="center left", bbox_to_anchor=(1.0, 0.5),
            fontsize=5, handlelength=0.7, handletextpad=0.4,
        )
    style_axes(ax)
    fig.tight_layout()
    return fig


def niche_cluster_spatial(
    adata: ad.AnnData,
    *,
    sample_key: str,
    spatial_key: str,
    cluster_per_cell: np.ndarray,
    max_points_per_sample: int = 8_000,
    n_cols: int = 3,
    title: str = "Niche clusters in spatial coords",
) -> Figure:
    """Per-sample spatial scatter: each cell coloured by its niche cluster."""
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    samples = adata.obs[sample_key].astype(str).to_numpy()
    df = pd.DataFrame(
        {
            "x": coords[:, 0], "y": coords[:, 1],
            "sample": samples, "cluster": cluster_per_cell.astype(str),
        }
    )
    # Drop unassigned cells from the scatter (they would dominate visually).
    df = df[df["cluster"] != "unassigned"]
    sampled = _stratified_subsample(df, "sample", max_points_per_sample)
    sample_ids = sorted(sampled["sample"].unique().tolist())
    n = len(sample_ids)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    clusters = sorted(sampled["cluster"].unique().tolist())
    cmap = _build_palette(len(clusters))
    color_map = {c: cmap[i] for i, c in enumerate(clusters)}

    fig, axes = new_figure(
        width=2.4 * n_cols, height=2.2 * n_rows + 0.6, nrows=n_rows, ncols=n_cols
    )
    axes = np.atleast_2d(axes)
    for k, sid in enumerate(sample_ids):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        sub = sampled[sampled["sample"] == sid]
        for cl in clusters:
            pts = sub[sub["cluster"] == cl]
            if pts.empty:
                continue
            ax.scatter(
                pts["x"], pts["y"], s=0.9, c=[color_map[cl]],
                linewidth=0, alpha=0.75, rasterized=True,
            )
        ax.set_title(sid)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        style_axes(ax)
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].set_visible(False)

    if len(clusters) <= 30:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=3, color=color_map[c], markeredgewidth=0)
            for c in clusters
        ]
        fig.legend(
            handles, clusters,
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=5, handlelength=0.7, handletextpad=0.4,
        )
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    return fig


def niche_cluster_composition_bar(stats: NicheClusterStats) -> Figure:
    """Stacked bar: per niche cluster, fraction of each ego cell type."""
    df = stats.composition
    fig, ax = new_figure(width=max(4.0, 0.35 * df["cluster"].nunique() + 1.6), height=3.2)
    if df.empty:
        ax.set_title("Niche cluster composition (empty)")
        style_axes(ax)
        return fig

    wide = df.pivot_table(
        index="cluster", columns="cell_type", values="fraction", aggfunc="sum"
    ).fillna(0.0)
    clusters = wide.index.tolist()
    cell_types = sorted(wide.columns.tolist())
    wide = wide[cell_types]
    cmap = _build_palette(len(cell_types))
    bottom = np.zeros(len(clusters))
    x = np.arange(len(clusters))
    for ct, color in zip(cell_types, cmap, strict=False):
        vals = wide[ct].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.2, label=ct)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"nc_{c}" for c in clusters], rotation=45, ha="right")
    ax.set_ylabel("ego-cell fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title("Niche cluster cell-type composition (ego cells)")
    if len(cell_types) <= 30:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=5, handlelength=0.7, handletextpad=0.4, borderaxespad=0.2,
            ncol=1 if len(cell_types) <= 15 else 2,
        )
    style_axes(ax)
    fig.tight_layout(rect=[0, 0, 0.85, 0.97])
    return fig


def niche_cluster_marker_heatmap(
    stats: NicheClusterStats, *, n_top: int = 5
) -> Figure:
    """Heatmap of the top-N marker genes per niche cluster."""
    df = stats.markers
    fig, ax = new_figure(width=max(4.0, 0.18 * df["cluster"].nunique() + 1.6), height=4.0)
    if df.empty:
        ax.set_title("Niche cluster markers (empty)")
        style_axes(ax)
        return fig
    top = df[df["rank"] <= n_top].copy()
    genes = pd.unique(top["gene"]).tolist()
    clusters = sorted(top["cluster"].unique().tolist())
    matrix = (
        top.pivot_table(index="gene", columns="cluster", values="score", aggfunc="max")
        .reindex(index=genes, columns=clusters)
    )
    z = matrix.values.astype(float)
    vmax = float(np.nanmax(np.abs(z))) if z.size else 1.0
    im = ax.imshow(z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"nc_{c}" for c in clusters], rotation=45, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    ax.set_title("Top marker genes per niche cluster (Wilcoxon score)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig
