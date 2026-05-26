"""Spatial scatter plots, one subplot per sample (matplotlib)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.reporting.plots.qc import _stratified_subsample
from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes


def spatial_figure(
    adata: ad.AnnData,
    *,
    color_key: str,
    sample_key: str,
    spatial_key: str = "spatial",
    max_points_per_sample: int = 8_000,
    n_cols: int = 3,
    title: str | None = None,
    point_size: float = 0.9,
) -> Figure:
    """Faceted spatial scatter: one subplot per sample, coloured by ``color_key``."""
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] missing")

    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "sample": adata.obs[sample_key].astype(str).to_numpy(),
            color_key: adata.obs[color_key].astype(str).to_numpy(),
        }
    )
    sampled = _stratified_subsample(df, "sample", max_points_per_sample)

    samples = sorted(sampled["sample"].unique())
    n = len(samples)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    categories = sorted(sampled[color_key].unique().tolist())
    cmap = _build_palette(len(categories))
    color_map = {c: cmap[i] for i, c in enumerate(categories)}

    fig, axes = new_figure(
        width=2.4 * n_cols, height=2.2 * n_rows + 0.6, nrows=n_rows, ncols=n_cols
    )
    axes = np.atleast_2d(axes)

    for k, sample in enumerate(samples):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        sub = sampled[sampled["sample"] == sample]
        for cat in categories:
            pts = sub[sub[color_key] == cat]
            if pts.empty:
                continue
            ax.scatter(
                pts["x"], pts["y"], s=point_size, c=[color_map[cat]],
                linewidth=0, alpha=0.75, rasterized=True,
            )
        ax.set_title(sample)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        style_axes(ax)

    # Hide unused axes
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].set_visible(False)

    # Single shared legend on the right if categorical count is reasonable.
    if len(categories) <= 30:
        handles = [
            _proxy_marker(color_map[c]) for c in categories
        ]
        fig.legend(
            handles,
            categories,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=5,
            markerscale=1.0,
            handlelength=0.7,
            handletextpad=0.4,
            borderaxespad=0.2,
            ncol=1 if len(categories) <= 20 else 2,
        )
    fig.suptitle(title or f"Spatial — {color_key} (≤{max_points_per_sample}/sample)")
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    return fig


def _proxy_marker(color):
    from matplotlib.lines import Line2D

    return Line2D([0], [0], marker="o", linestyle="", markersize=3, color=color, markeredgewidth=0)
