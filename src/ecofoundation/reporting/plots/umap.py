"""UMAP scatter plots (matplotlib)."""

from __future__ import annotations

import anndata as ad
import numpy as np
from matplotlib.figure import Figure

from ecofoundation.reporting.style import new_figure, style_axes


def umap_figure(
    adata: ad.AnnData,
    *,
    color_key: str,
    obsm_key: str = "X_umap",
    max_points: int = 30_000,
    title: str | None = None,
    point_size: float = 1.4,
) -> Figure:
    """Interactive UMAP scatter coloured by an obs column."""
    if obsm_key not in adata.obsm:
        raise KeyError(f"obsm['{obsm_key}'] missing")
    coords = np.asarray(adata.obsm[obsm_key])[:, :2]
    color = adata.obs[color_key].astype(str).to_numpy()

    n = coords.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        coords = coords[idx]
        color = color[idx]
        suffix = f" (sampled {max_points}/{n})"
    else:
        suffix = ""

    fig, ax = new_figure(width=5.5, height=4.2)
    categories = sorted(np.unique(color).tolist())
    cmap = _build_palette(len(categories))
    color_map = {c: cmap[i] for i, c in enumerate(categories)}

    for cat in categories:
        m = color == cat
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            s=point_size,
            c=[color_map[cat]],
            label=cat,
            linewidth=0,
            alpha=0.8,
            rasterized=True,
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title((title or f"UMAP — {color_key}") + suffix)
    if len(categories) <= 30:
        ax.legend(
            markerscale=4,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            ncol=1 if len(categories) <= 15 else 2,
            handlelength=0.7,
            handletextpad=0.4,
            borderaxespad=0.2,
        )
    style_axes(ax)
    fig.tight_layout()
    return fig


def _build_palette(n: int) -> list:
    """A categorical palette: 20 distinct hues via tab20, then cycle."""
    import matplotlib.cm as cm

    base = list(cm.tab20.colors) + list(cm.tab20b.colors) + list(cm.tab20c.colors)
    if n <= len(base):
        return base[:n]
    # Extend with HSV cycle if needed
    extra = cm.hsv(np.linspace(0, 1, n - len(base), endpoint=False))
    return base + list(extra)
