"""Aggregate per-niche explanations into class-level summaries.

For each predicted class:

  - **Top gene-importance** — mean(|node_attr|) per gene across all explained
    niches of that class. Identifies discriminative genes regardless of cell.
  - **Top edge-feature channels** — mean(|edge_attr_attr|) per edge feature
    channel (distance, LR-score, ...). Identifies which channel of edge
    information the model relies on.
  - **Median edge importance** — mean(edge_mask) within each cell-type-pair.
    Identifies cell-cell interactions that drive predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecofoundation.interpretation.gnn_explainer import NicheExplanation
from ecofoundation.interpretation.integrated_gradients import IGResult


@dataclass
class ClassAttribution:
    """Aggregated attribution for one class."""

    target_class: int
    n_niches_explained: int

    # Node-feature attributions
    gene_importance: pd.DataFrame  # cols: gene, mean_abs_attr, mean_attr
    # Edge-feature attributions
    edge_channel_importance: pd.DataFrame  # cols: channel, mean_abs_attr
    # GNNExplainer outputs
    node_mask_summary: pd.DataFrame  # cols: gene, mean_node_mask (if attributes mode)


def aggregate_explanations(
    explanations: list[NicheExplanation],
    ig_results: list[IGResult] | None,
    *,
    feature_names: list[str],
    edge_channel_names: list[str],
    target_class: int,
) -> ClassAttribution:
    """Build a :class:`ClassAttribution` from per-niche explanations."""
    if not explanations and not (ig_results or []):
        return ClassAttribution(
            target_class=target_class,
            n_niches_explained=0,
            gene_importance=pd.DataFrame(columns=["gene", "mean_abs_attr", "mean_attr"]),
            edge_channel_importance=pd.DataFrame(columns=["channel", "mean_abs_attr"]),
            node_mask_summary=pd.DataFrame(columns=["gene", "mean_node_mask"]),
        )

    n_features = len(feature_names)
    abs_sum = np.zeros(n_features, dtype=np.float64)
    raw_sum = np.zeros(n_features, dtype=np.float64)
    abs_n = 0
    node_mask_per_gene = np.zeros(n_features, dtype=np.float64)
    node_mask_n = 0

    if ig_results is None:
        ig_results = []

    # --- node-feature attributions: IG + GNNExplainer ----------------------
    for ig in ig_results:
        # ig.node_attrs shape (n_nodes, n_features) — mean over nodes for niche-level signal
        per_niche = ig.node_attrs
        if per_niche.shape[1] != n_features:
            continue
        abs_sum += np.abs(per_niche).mean(axis=0)
        raw_sum += per_niche.mean(axis=0)
        abs_n += 1

    for ex in explanations:
        nm = ex.node_mask
        if nm.ndim == 2 and nm.shape[1] == n_features:
            node_mask_per_gene += nm.mean(axis=0)
            node_mask_n += 1

    if abs_n > 0:
        abs_sum /= abs_n
        raw_sum /= abs_n
    if node_mask_n > 0:
        node_mask_per_gene /= node_mask_n

    gene_imp = pd.DataFrame(
        {
            "gene": feature_names,
            "mean_abs_attr": abs_sum,
            "mean_attr": raw_sum,
        }
    ).sort_values("mean_abs_attr", ascending=False, ignore_index=True)

    node_mask_summary = pd.DataFrame(
        {"gene": feature_names, "mean_node_mask": node_mask_per_gene}
    ).sort_values("mean_node_mask", ascending=False, ignore_index=True)

    # --- edge-feature channel attributions ---------------------------------
    channel_abs = np.zeros(len(edge_channel_names), dtype=np.float64)
    edge_n = 0
    for ig in ig_results:
        if ig.edge_attr_attrs is None:
            continue
        if ig.edge_attr_attrs.shape[1] != len(edge_channel_names):
            continue
        channel_abs += np.abs(ig.edge_attr_attrs).mean(axis=0)
        edge_n += 1
    if edge_n > 0:
        channel_abs /= edge_n
    edge_channel_imp = pd.DataFrame(
        {"channel": edge_channel_names, "mean_abs_attr": channel_abs}
    ).sort_values("mean_abs_attr", ascending=False, ignore_index=True)

    return ClassAttribution(
        target_class=target_class,
        n_niches_explained=len(explanations),
        gene_importance=gene_imp,
        edge_channel_importance=edge_channel_imp,
        node_mask_summary=node_mask_summary,
    )
