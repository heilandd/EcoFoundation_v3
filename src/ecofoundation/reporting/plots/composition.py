"""Sample × cluster composition plot (matplotlib stacked bar)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes


def _nat(s: str) -> tuple:
    try:
        return (0, int(s))
    except ValueError:
        return (1, str(s))


def composition_bar(
    composition_long: pd.DataFrame,
    *,
    sample_col: str,
    cluster_col: str,
    title: str = "Cluster composition per sample",
) -> Figure:
    """Stacked-bar chart: x = sample, y = fraction, color = cluster."""
    wide = composition_long.pivot_table(
        index=sample_col, columns=cluster_col, values="fraction", aggfunc="sum"
    ).fillna(0.0)
    clusters = sorted(wide.columns, key=_nat)
    wide = wide[clusters]
    samples = wide.index.tolist()

    cmap = _build_palette(len(clusters))
    fig, ax = new_figure(width=max(4, 0.35 * len(samples) + 1.5), height=3.2)
    bottom = np.zeros(len(samples))
    x = np.arange(len(samples))
    for c, color in zip(clusters, cmap, strict=False):
        vals = wide[c].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.2, label=str(c))
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title(title)
    if len(clusters) <= 30:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=5,
            handlelength=0.7,
            handletextpad=0.4,
            borderaxespad=0.2,
            ncol=1 if len(clusters) <= 15 else 2,
        )
    style_axes(ax)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    return fig
