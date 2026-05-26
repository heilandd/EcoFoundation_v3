"""Niche visualisations (matplotlib): size distribution, per-patient bar, centroid overlay."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ecofoundation.niches.base import NicheAssignment
from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes


def niche_size_distribution(niches: NicheAssignment) -> Figure:
    """Overlaid histograms of niche size per group."""
    sizes = niches.sizes()
    groups = niches.group_label.astype(str)
    unique_groups = sorted(np.unique(groups).tolist())
    cmap = _build_palette(len(unique_groups))

    fig, ax = new_figure(width=4.8, height=2.4)
    bins = np.linspace(0, max(int(sizes.max()) + 1, 10), 41) if sizes.size else np.linspace(0, 10, 11)
    for g, color in zip(unique_groups, cmap, strict=False):
        mask = groups == g
        ax.hist(sizes[mask], bins=bins, alpha=0.55, label=g, color=color, edgecolor="black", linewidth=0.2)
    ax.set_xlabel("cells per niche")
    ax.set_ylabel("# niches")
    ax.set_title(f"Niche size distribution (n={niches.n_niches})")
    if len(unique_groups) <= 20:
        ax.legend(
            fontsize=5,
            handlelength=0.7,
            handletextpad=0.4,
            loc="upper right",
        )
    style_axes(ax)
    fig.tight_layout()
    return fig


def niches_per_group_bar(niches: NicheAssignment) -> Figure:
    groups, counts = np.unique(niches.group_label.astype(str), return_counts=True)
    fig, ax = new_figure(width=4.0, height=2.4)
    cmap = _build_palette(len(groups))
    ax.bar(range(len(groups)), counts, color=cmap, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("# niches")
    ax.set_title("Niches per patient")
    style_axes(ax)
    fig.tight_layout()
    return fig


def niche_centroids_spatial(
    adata,
    niches: NicheAssignment,
    *,
    sample_key: str,
    spatial_key: str = "spatial",
    cell_sample: int = 8_000,
    n_cols: int = 3,
) -> Figure:
    """Per-sample spatial scatter: cells (subsampled) + niche centroids overlaid."""
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    samples = adata.obs[sample_key].astype(str).to_numpy()
    sample_per_niche = (
        niches.sample_label.astype(str)
        if niches.sample_label is not None
        else samples[niches.ego_cell].astype(str)
    )
    sample_ids = sorted(np.unique(samples).tolist())
    n_cols = min(n_cols, len(sample_ids))
    n_rows = int(np.ceil(len(sample_ids) / n_cols))

    rng = np.random.default_rng(0)
    fig, axes = new_figure(
        width=2.4 * n_cols, height=2.2 * n_rows + 0.6, nrows=n_rows, ncols=n_cols
    )
    axes = np.atleast_2d(axes)
    sizes_all = niches.sizes()
    for k, sid in enumerate(sample_ids):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        cell_mask = samples == sid
        cell_idx = np.flatnonzero(cell_mask)
        if cell_idx.size > cell_sample:
            cell_idx = rng.choice(cell_idx, size=cell_sample, replace=False)
        ax.scatter(
            coords[cell_idx, 0], coords[cell_idx, 1],
            s=0.4, c="#cccccc", linewidth=0, alpha=0.55, rasterized=True,
        )
        niche_mask = sample_per_niche == sid
        if niche_mask.any():
            cents = niches.centroid[niche_mask]
            sz = sizes_all[niche_mask]
            ax.scatter(
                cents[:, 0], cents[:, 1],
                s=np.clip(np.sqrt(sz) * 1.5, 1.5, 12),
                c="#d62728", edgecolors="white", linewidth=0.2, alpha=0.85,
            )
        ax.set_title(sid)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        style_axes(ax)
    for k in range(len(sample_ids), n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].set_visible(False)
    fig.suptitle(f"Niche centroids on spatial (background sampled ≤{cell_sample}/sample)")
    fig.tight_layout()
    return fig
