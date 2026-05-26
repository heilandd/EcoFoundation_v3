"""Cell-type-importance visualisations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.interpretation.cell_type_attribution import CellTypeAttribution
from ecofoundation.reporting.style import new_figure, style_axes


def cell_type_importance_heatmap(attribution: CellTypeAttribution) -> Figure:
    """Heatmap: rows = class, cols = cell type, values = mean |attribution|."""
    df = attribution.by_class_celltype
    fig, ax = new_figure(
        width=max(3.5, 0.18 * len(df.columns) + 1.6),
        height=max(2.0, 0.45 * len(df.index) + 1.0),
    )
    if df.empty:
        ax.set_title("Cell-type importance (empty)")
        style_axes(ax)
        return fig
    z = df.values
    vmax = float(np.nanmax(z)) if z.size else 1.0
    im = ax.imshow(z, cmap="magma", aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("cell type")
    ax.set_ylabel("class")
    ax.set_title("Cell-type importance per class (mean |IG attribution|)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig


def cell_type_importance_bar(
    attribution: CellTypeAttribution, *, class_label: str, top_n: int = 15
) -> Figure:
    """Horizontal bar of top cell types for ONE class."""
    sub = (
        attribution.per_class[attribution.per_class["class_label"] == class_label]
        .sort_values("mean_abs_attr", ascending=False)
        .head(top_n)
    )
    fig, ax = new_figure(width=4.0, height=max(2.2, 0.22 * len(sub) + 0.6))
    if sub.empty:
        ax.set_title(f"Cell-type importance — {class_label} (empty)")
        style_axes(ax)
        return fig
    y = np.arange(len(sub))
    ax.barh(y, sub["mean_abs_attr"], color="#a5b4fc", edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["cell_type"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("mean |IG attribution|")
    ax.set_title(f"Top cell types — {class_label}")
    style_axes(ax)
    fig.tight_layout()
    return fig
