"""Explainability plots: importance-coloured niche graphs + top-feature bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch_geometric.data import Data

from ecofoundation.interpretation.aggregation import ClassAttribution
from ecofoundation.interpretation.gnn_explainer import NicheExplanation
from ecofoundation.reporting.style import new_figure, style_axes


def niche_explanation_figure(
    data: Data,
    explanation: NicheExplanation,
    *,
    title: str | None = None,
) -> Figure:
    """Draw a niche graph with edge thickness = edge importance, node color = node importance."""
    coords = getattr(data, "pos", None)
    if coords is None:
        n = data.num_nodes
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
    elif isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    else:
        coords = np.asarray(coords)

    ei = data.edge_index.cpu().numpy()
    em = explanation.edge_mask
    nm = explanation.node_mask
    # Reduce node_mask to scalar per node (sum over features) if 2-D.
    if nm.ndim == 2:
        node_score = np.abs(nm).sum(axis=1)
    else:
        node_score = np.abs(nm)

    fig, ax = new_figure(width=4.0, height=3.4)
    if ei.shape[1] > 0:
        wmax = float(em.max()) if em.size and em.max() > 0 else 1.0
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            if a >= b:
                continue
            w = float(em[k]) / wmax if wmax > 0 else 0.0
            ax.plot(
                [coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                color="black",
                linewidth=0.3 + 1.6 * w,
                alpha=float(np.clip(0.15 + 0.7 * w, 0.1, 0.9)),
                solid_capstyle="round",
            )
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        s=20, c=node_score, cmap="viridis", edgecolors="white", linewidth=0.4
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title
        or (
            f"Niche {explanation.niche_id} — explained for class {explanation.target_class}"
            + (f" (prob={explanation.target_prob:.2f})" if explanation.target_prob is not None else "")
        )
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("|node importance|", fontsize=5)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig


def top_gene_importance_bar(
    attribution: ClassAttribution,
    *,
    top_n: int = 20,
    class_label: str | None = None,
) -> Figure:
    """Horizontal bar of the top-N genes by mean |IG attribution|."""
    df = attribution.gene_importance.head(top_n)
    fig, ax = new_figure(width=4.0, height=max(2.2, 0.18 * len(df) + 0.8))
    if df.empty:
        ax.set_title("No gene importance data")
        style_axes(ax)
        return fig
    y = np.arange(len(df))
    ax.barh(y, df["mean_abs_attr"], color="#a5b4fc", edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("mean |IG attribution|")
    ax.set_title(
        f"Top {len(df)} genes — class {class_label if class_label else attribution.target_class}"
    )
    style_axes(ax)
    fig.tight_layout()
    return fig


def edge_channel_importance_bar(
    attribution: ClassAttribution, *, class_label: str | None = None
) -> Figure:
    """Bar of edge-feature channel importance (distance vs LR vs ...)."""
    df = attribution.edge_channel_importance
    fig, ax = new_figure(width=3.5, height=2.4)
    if df.empty:
        ax.set_title("No edge channel data")
        style_axes(ax)
        return fig
    x = np.arange(len(df))
    ax.bar(x, df["mean_abs_attr"], color="#fdba74", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["channel"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("mean |IG attribution|")
    ax.set_title(
        f"Edge feature importance — class {class_label if class_label else attribution.target_class}"
    )
    style_axes(ax)
    fig.tight_layout()
    return fig
