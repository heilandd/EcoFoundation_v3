"""Niche-graph visualisations (matplotlib)."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from torch_geometric.data import Data

from ecofoundation.reporting.style import new_figure, style_axes


def niche_graph_figure(
    data: Data,
    *,
    edge_feature_index: int = 1,
    edge_feature_name: str = "lr_score_tier1",
    title: str | None = None,
) -> Figure:
    """Draw one niche graph using its actual spatial coords."""
    coords = getattr(data, "pos", None)
    if coords is None:
        n = data.num_nodes
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        coords = np.asarray(coords)

    ei = data.edge_index.cpu().numpy()
    ea = data.edge_attr.cpu().numpy() if data.edge_attr is not None else None

    fig, ax = new_figure(width=3.6, height=3.2)
    # Edges
    if ei.shape[1] > 0:
        if ea is not None and ea.shape[1] > edge_feature_index:
            w = ea[:, edge_feature_index]
            wmax = float(w.max()) if w.size and w.max() > 0 else 1.0
        else:
            w = np.ones(ei.shape[1])
            wmax = 1.0
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            if a >= b:
                continue
            alpha = 0.2 + 0.6 * (w[k] / wmax)
            ax.plot(
                [coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                color="black",
                linewidth=0.4 + 0.8 * (w[k] / wmax),
                alpha=float(np.clip(alpha, 0.1, 0.9)),
                solid_capstyle="round",
            )
    ax.scatter(coords[:, 0], coords[:, 1], s=10, c="#2563eb", edgecolors="white", linewidth=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"Niche graph (n={data.num_nodes})")
    style_axes(ax)
    fig.tight_layout()
    return fig


def edge_feature_distributions(
    edge_feature_matrix: np.ndarray,
    feature_names: list[str],
    *,
    max_sample: int = 200_000,
) -> Figure:
    """Side-by-side histograms of each edge-feature channel across all niches."""
    if edge_feature_matrix.shape[0] > max_sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(edge_feature_matrix.shape[0], size=max_sample, replace=False)
        edge_feature_matrix = edge_feature_matrix[idx]
    n_channels = edge_feature_matrix.shape[1]

    fig, axes = new_figure(width=2.6 * n_channels, height=2.4, nrows=1, ncols=n_channels)
    if n_channels == 1:
        axes = np.array([axes])
    for k in range(n_channels):
        ax = axes[k]
        ax.hist(edge_feature_matrix[:, k], bins=50, color="#a5b4fc", edgecolor="black", linewidth=0.3)
        ax.set_xlabel(feature_names[k])
        ax.set_ylabel("# edges")
        style_axes(ax)
    fig.suptitle(f"Edge feature distributions (≤{max_sample} edges)")
    fig.tight_layout()
    return fig


def graph_size_distributions(graphs: list[Data]) -> Figure:
    n_nodes = np.array([g.num_nodes for g in graphs])
    n_edges = np.array([g.edge_index.shape[1] // 2 for g in graphs])
    fig, axes = new_figure(width=5.4, height=2.2, nrows=1, ncols=2)
    axes[0].hist(n_nodes, bins=40, color="#a5b4fc", edgecolor="black", linewidth=0.3)
    axes[0].set_title("# nodes per niche")
    axes[0].set_xlabel("nodes")
    axes[0].set_ylabel("# niches")
    style_axes(axes[0])
    axes[1].hist(n_edges, bins=40, color="#fdba74", edgecolor="black", linewidth=0.3)
    axes[1].set_title("# edges per niche")
    axes[1].set_xlabel("edges (undirected)")
    axes[1].set_ylabel("# niches")
    style_axes(axes[1])
    fig.tight_layout()
    return fig
