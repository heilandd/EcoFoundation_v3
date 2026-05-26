"""Per-class cell-type importance from IG attributions.

For each explained niche we have ``ig.node_attrs`` of shape (n_nodes, n_features).
We collapse the per-feature dimension to a single score per cell
(``sum(|attr|)``), then group cells by their cell-type annotation and
aggregate. The result is "which cell types in which class consistently
contribute strong attribution to the model's prediction".
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from ecofoundation.interpretation.integrated_gradients import IGResult


@dataclass
class CellTypeAttribution:
    """Per-class cell-type importance summary."""

    per_class: pd.DataFrame  # cols: class_idx, cell_type, mean_abs_attr, n_cells
    by_class_celltype: pd.DataFrame  # wide: rows = class_idx, cols = cell_type


def compute_cell_type_attribution(
    *,
    adata: ad.AnnData,
    celltype_col: str,
    ig_results_by_class: dict[int, list[IGResult]],
    explained_data_by_class: dict[int, list[Data]],
    class_label_map: list[str],
) -> CellTypeAttribution:
    """Aggregate per-class IG attribution at the cell-type level."""
    if celltype_col not in adata.obs.columns:
        return CellTypeAttribution(
            per_class=pd.DataFrame(
                columns=["class_idx", "class_label", "cell_type", "mean_abs_attr", "n_cells"]
            ),
            by_class_celltype=pd.DataFrame(),
        )
    celltype_global = adata.obs[celltype_col].astype(str).to_numpy()

    rows: list[dict] = []
    for cls_idx, igs in ig_results_by_class.items():
        datas = explained_data_by_class.get(cls_idx, [])
        if not igs or not datas:
            continue
        # Sum-of-abs node attributions per cell (across features).
        # Pair each IG with its matching Data object via niche_id order.
        ig_by_nid = {ig.niche_id: ig for ig in igs}
        ct_scores: dict[str, list[float]] = {}
        for data in datas:
            ig = ig_by_nid.get(int(data.niche_id))
            if ig is None:
                continue
            attrs_per_cell = np.abs(ig.node_attrs).sum(axis=1)  # (n_nodes,)
            global_ix = data.global_cell_indices.cpu().numpy()
            cts_in_niche = celltype_global[global_ix]
            for c, a in zip(cts_in_niche, attrs_per_cell, strict=True):
                ct_scores.setdefault(str(c), []).append(float(a))

        for ct, vals in ct_scores.items():
            rows.append(
                {
                    "class_idx": cls_idx,
                    "class_label": (
                        class_label_map[cls_idx]
                        if cls_idx < len(class_label_map)
                        else str(cls_idx)
                    ),
                    "cell_type": ct,
                    "mean_abs_attr": float(np.mean(vals)),
                    "n_cells": len(vals),
                }
            )

    per_class = pd.DataFrame(rows)
    if per_class.empty:
        return CellTypeAttribution(per_class=per_class, by_class_celltype=pd.DataFrame())
    wide = per_class.pivot_table(
        index="class_label", columns="cell_type", values="mean_abs_attr", aggfunc="mean"
    ).fillna(0.0)
    return CellTypeAttribution(per_class=per_class, by_class_celltype=wide)
