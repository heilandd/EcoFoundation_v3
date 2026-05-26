"""Niche-embedding UMAP plot."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecofoundation.interpretation.embeddings import NicheEmbeddings
from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes


def niche_embedding_umap_figure(
    embeddings: NicheEmbeddings,
    *,
    labels: np.ndarray,
    class_label_map: list[str],
    explained_niche_ids: np.ndarray | None = None,
    title: str = "Niche embedding UMAP",
) -> Figure:
    """UMAP scatter of niche embeddings coloured by predicted/true class.

    Explained niches are highlighted with a black outline.
    """
    if embeddings.umap_2d is None:
        fig, ax = new_figure(width=4, height=3)
        ax.set_title(title + " (no UMAP computed)")
        style_axes(ax)
        return fig

    coords = embeddings.umap_2d
    fig, ax = new_figure(width=5.0, height=4.0)
    classes = sorted(set(labels.tolist()))
    cmap = _build_palette(max(len(classes), 1))
    label_to_name = {i: (class_label_map[i] if i < len(class_label_map) else str(i)) for i in classes}
    for ci, c in enumerate(classes):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=2.5, c=[cmap[ci]], alpha=0.55, linewidth=0, rasterized=True,
            label=label_to_name[c],
        )
    if explained_niche_ids is not None and len(explained_niche_ids):
        # Highlight the explained niches with a larger empty marker.
        nid_to_pos = {int(nid): i for i, nid in enumerate(embeddings.niche_ids.tolist())}
        positions = [nid_to_pos[int(nid)] for nid in explained_niche_ids.tolist() if int(nid) in nid_to_pos]
        if positions:
            highlighted = coords[positions]
            ax.scatter(
                highlighted[:, 0], highlighted[:, 1],
                s=18, facecolors="none", edgecolors="black", linewidth=0.6,
                label="explained",
            )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(
        markerscale=2, loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=5, handlelength=0.7, handletextpad=0.4, borderaxespad=0.2,
    )
    style_axes(ax)
    fig.tight_layout()
    return fig
