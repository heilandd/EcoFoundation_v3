"""Pathway-enrichment plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.interpretation.pathway_enrichment import PathwayEnrichmentResult
from ecofoundation.reporting.style import new_figure, style_axes


def pathway_dotplot(
    result: PathwayEnrichmentResult,
    *,
    class_label: str,
    top_n_terms: int = 8,
    p_value_col: str = "adjusted_p_value",
) -> Figure:
    """Dotplot: rows = cell type, columns = pathway term, dot size = -log10(p), color = combined score."""
    df = result.per_class_celltype
    fig, ax = new_figure(width=6.5, height=4.0)
    if df.empty:
        ax.set_title(f"Pathway enrichment — {class_label} (no data)")
        style_axes(ax)
        return fig

    sub = df[df["class_label"] == class_label].copy()
    if sub.empty:
        ax.set_title(f"Pathway enrichment — {class_label} (empty)")
        style_axes(ax)
        return fig

    # Pick top terms per cell type
    sub = (
        sub.sort_values([p_value_col])
        .groupby("cell_type", group_keys=False)
        .head(top_n_terms)
    )
    if sub.empty:
        ax.set_title(f"Pathway enrichment — {class_label} (empty)")
        style_axes(ax)
        return fig

    sub["neg_log_p"] = -np.log10(np.clip(sub[p_value_col], 1e-300, None))
    terms = list(dict.fromkeys(sub["term"].tolist()))
    cell_types = sorted(sub["cell_type"].unique().tolist())

    term_to_x = {t: i for i, t in enumerate(terms)}
    ct_to_y = {c: i for i, c in enumerate(cell_types)}

    xs = [term_to_x[t] for t in sub["term"]]
    ys = [ct_to_y[c] for c in sub["cell_type"]]
    sizes = (sub["neg_log_p"] * 6).clip(2, 80)
    colors = sub["combined_score"].clip(lower=0)

    sc = ax.scatter(
        xs, ys, s=sizes, c=colors, cmap="viridis",
        edgecolors="white", linewidth=0.3,
    )
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=45, ha="right")
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types)
    ax.set_title(f"Pathway enrichment — {class_label}")
    ax.set_xlim(-0.5, len(terms) - 0.5)
    ax.set_ylim(-0.5, len(cell_types) - 0.5)
    ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("combined score", fontsize=5)
    cb.ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig
