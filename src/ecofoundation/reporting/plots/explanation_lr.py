"""LR-interaction explainability plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.interpretation.lr_interaction_attribution import (
    LRInteractionAttribution,
)
from ecofoundation.reporting.style import new_figure, style_axes


def top_lr_interactions_table(
    attribution: LRInteractionAttribution,
    *,
    class_label: str,
    top_n: int = 25,
) -> pd.DataFrame:
    """Return a tidy table of top LR interactions for one class."""
    df = attribution.per_class_lr
    if df.empty:
        return df
    sub = (
        df[df["class_label"] == class_label]
        .sort_values("weighted_score", ascending=False)
        .head(top_n)
        .copy()
    )
    sub["lr_pair"] = sub["ligand"] + " → " + sub["receptor"]
    sub["ct_pair"] = sub["ct_pair_a"] + " ↔ " + sub["ct_pair_b"]
    return sub[
        ["ct_pair", "lr_pair", "weighted_score", "n_edges"]
    ].reset_index(drop=True)


def lr_celltype_pair_heatmap(
    attribution: LRInteractionAttribution,
    *,
    class_label: str,
    top_n_cell_pairs: int = 15,
    top_n_lr_pairs: int = 20,
) -> Figure:
    """Heatmap: rows = cell-type pair, cols = LR pair, values = weighted score."""
    df = attribution.per_class_lr
    fig, ax = new_figure(width=6.0, height=4.0)
    if df.empty:
        ax.set_title(f"LR × celltype-pair — {class_label} (empty)")
        style_axes(ax)
        return fig

    sub = df[df["class_label"] == class_label].copy()
    if sub.empty:
        ax.set_title(f"LR × celltype-pair — {class_label} (empty)")
        style_axes(ax)
        return fig

    sub["lr_pair"] = sub["ligand"] + "->" + sub["receptor"]
    sub["ct_pair"] = sub["ct_pair_a"] + "<->" + sub["ct_pair_b"]
    top_lr = (
        sub.groupby("lr_pair", as_index=False)["weighted_score"].sum()
        .sort_values("weighted_score", ascending=False)
        .head(top_n_lr_pairs)["lr_pair"].tolist()
    )
    top_ct = (
        sub.groupby("ct_pair", as_index=False)["weighted_score"].sum()
        .sort_values("weighted_score", ascending=False)
        .head(top_n_cell_pairs)["ct_pair"].tolist()
    )
    pivot = (
        sub[sub["lr_pair"].isin(top_lr) & sub["ct_pair"].isin(top_ct)]
        .pivot_table(index="ct_pair", columns="lr_pair", values="weighted_score", aggfunc="sum")
        .reindex(index=top_ct, columns=top_lr)
        .fillna(0.0)
    )

    z = pivot.values
    vmax = float(np.nanmax(z)) if z.size else 1.0
    im = ax.imshow(z, cmap="magma", aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("ligand -> receptor")
    ax.set_ylabel("cell-type pair (canonical)")
    ax.set_title(f"LR pair x cell-type pair — {class_label}")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig
