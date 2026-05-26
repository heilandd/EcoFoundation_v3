"""Per-cluster biology plots (Step 7).

Adds three plot types to the report when the unsupervised pipeline produces
:class:`~ecofoundation.niches.cluster_biology.ClusterBiology`:

  - **cluster_pathway_dotplot** — rows = cluster, cols = enriched pathway term,
    dot size = -log10(p), dot color = combined score.
  - **cluster_lr_heatmap** — rows = cluster, cols = top LR pairs (across all
    clusters), values = aggregated weighted LR score.
  - **example_niche_celltypes** — spatial scatter of one example niche, cells
    coloured by cell-type annotation, with the intra-niche edges overlaid.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch_geometric.data import Data

from ecofoundation.niches.cluster_biology import ClusterBiology
from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes


# ---------------------------------------------------------------------------
# Pathway dotplot per cluster
# ---------------------------------------------------------------------------


def cluster_pathway_dotplot(
    biology: ClusterBiology,
    *,
    top_n_terms_per_cluster: int = 5,
    p_value_col: str = "adjusted_p_value",
    title: str = "Pathway enrichment per niche cluster",
) -> Figure:
    df = biology.pathways
    fig, ax = new_figure(width=7.5, height=5.0)
    if df.empty:
        ax.set_title(title + " (no data)")
        style_axes(ax)
        return fig

    sub = (
        df.sort_values([p_value_col])
        .groupby("cluster", group_keys=False)
        .head(top_n_terms_per_cluster)
        .copy()
    )
    if sub.empty:
        ax.set_title(title + " (empty)")
        style_axes(ax)
        return fig

    sub["neg_log_p"] = -np.log10(np.clip(sub[p_value_col].astype(float), 1e-300, None))
    terms = list(dict.fromkeys(sub["term"].tolist()))
    clusters = sorted(sub["cluster"].unique().tolist())
    term_to_x = {t: i for i, t in enumerate(terms)}
    cl_to_y = {c: i for i, c in enumerate(clusters)}

    xs = [term_to_x[t] for t in sub["term"]]
    ys = [cl_to_y[c] for c in sub["cluster"]]
    sizes = (sub["neg_log_p"] * 6).clip(2, 80)
    colors = sub["combined_score"].astype(float).clip(lower=0)

    sc = ax.scatter(
        xs, ys, s=sizes, c=colors, cmap="viridis",
        edgecolors="white", linewidth=0.3,
    )
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=45, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"nc_{c}" for c in clusters])
    ax.set_xlabel("pathway term")
    ax.set_ylabel("niche cluster")
    ax.set_title(title)
    ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("combined score", fontsize=5)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LR pair heatmap per cluster
# ---------------------------------------------------------------------------


def cluster_lr_heatmap(
    biology: ClusterBiology,
    *,
    top_n_lr_pairs: int = 25,
    title: str = "Top LR pairs per niche cluster",
) -> Figure:
    df = biology.lr_interactions
    fig, ax = new_figure(width=7.5, height=5.5)
    if df.empty:
        ax.set_title(title + " (no data)")
        style_axes(ax)
        return fig

    df = df.copy()
    df["lr_pair"] = df["ligand"] + "->" + df["receptor"]
    # Pick top LR pairs across all clusters by total score
    top_pairs = (
        df.groupby("lr_pair", as_index=False)["score_sum"].sum()
        .sort_values("score_sum", ascending=False)
        .head(top_n_lr_pairs)["lr_pair"].tolist()
    )
    pivot = (
        df[df["lr_pair"].isin(top_pairs)]
        .groupby(["cluster", "lr_pair"], as_index=False)["score_sum"].sum()
        .pivot_table(index="cluster", columns="lr_pair", values="score_sum", aggfunc="sum")
        .reindex(columns=top_pairs)
        .fillna(0.0)
    )

    z = pivot.values
    vmax = float(np.nanmax(z)) if z.size else 1.0
    im = ax.imshow(z, cmap="magma", aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"nc_{c}" for c in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
    ax.set_xlabel("ligand -> receptor")
    ax.set_ylabel("niche cluster")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("aggregated score", fontsize=5)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Example niche overlay with cell types
# ---------------------------------------------------------------------------


def example_niche_celltypes_figure(
    data: Data,
    *,
    adata: ad.AnnData,
    celltype_col: str,
    coords_global: np.ndarray,
    cluster_label: str,
    title: str | None = None,
) -> Figure:
    """Spatial scatter of one niche, cells coloured by cell type, edges overlaid."""
    global_ix = data.global_cell_indices.cpu().numpy()
    xy = coords_global[global_ix]
    cts = adata.obs[celltype_col].astype(str).to_numpy()[global_ix]

    cats = sorted(set(cts.tolist()))
    cmap = _build_palette(len(cats))
    color_map = {c: cmap[i] for i, c in enumerate(cats)}

    fig, ax = new_figure(width=4.4, height=3.6)

    # Edges
    ei = data.edge_index.cpu().numpy()
    if ei.shape[1] > 0:
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            if a >= b:
                continue
            ax.plot(
                [xy[a, 0], xy[b, 0]],
                [xy[a, 1], xy[b, 1]],
                color="grey", linewidth=0.3, alpha=0.5, zorder=1,
            )

    # Nodes
    for cat in cats:
        m = cts == cat
        ax.scatter(
            xy[m, 0], xy[m, 1],
            s=22, c=[color_map[cat]],
            edgecolors="white", linewidth=0.4,
            label=cat, zorder=2,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"Niche {int(getattr(data, 'niche_id', -1))} — {cluster_label}")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=5, handlelength=0.7, handletextpad=0.4, borderaxespad=0.2,
        markerscale=0.6,
    )
    style_axes(ax)
    fig.tight_layout()
    return fig
