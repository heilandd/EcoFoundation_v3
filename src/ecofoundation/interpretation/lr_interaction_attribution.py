"""Post-hoc LR-pair × cell-type-pair decomposition of important edges.

After GNNExplainer / IG identify which edges matter most for a class, we
decompose each edge's importance into individual ligand-receptor pair
contributions using the underlying expression data.

Pipeline:

  1. For each explained niche:
     a) For each edge ``(cell_a, cell_b)``, compute IG-based edge importance
        from ``ig.edge_attr_attrs`` (sum of |attr| across edge feature channels).
     b) Keep the top-K most important edges.
     c) For each top edge, score every LR pair as
        ``expr[a, L] * expr[b, R] + expr[b, L] * expr[a, R]``
        and rank.
     d) Annotate each edge with sender and receiver cell types.

  2. Aggregate across niches per class:
     - ``(sender_ct, receiver_ct, ligand, receptor)`` → total weighted score
     - Top-K tuples per class
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from ecofoundation.graph.lr_scoring import LRResource
from ecofoundation.interpretation.integrated_gradients import IGResult


@dataclass
class LRInteractionAttribution:
    """Top LR-pair × celltype-pair interactions per class."""

    per_class_lr: pd.DataFrame
    # cols: class_idx, class_label, sender_celltype, receiver_celltype,
    #       ligand, receptor, weighted_score, n_edges_supporting


def compute_lr_interaction_attribution(
    *,
    adata: ad.AnnData,
    celltype_col: str,
    lr_resource: LRResource,
    expression_layer: str,
    ig_results_by_class: dict[int, list[IGResult]],
    explained_data_by_class: dict[int, list[Data]],
    class_label_map: list[str],
    top_k_edges_per_niche: int = 20,
    top_k_lr_pairs_per_edge: int = 5,
) -> LRInteractionAttribution:
    """Compute per-class LR-pair × cell-type-pair attribution."""
    if (
        celltype_col not in adata.obs.columns
        or lr_resource.n_pairs_kept == 0
    ):
        return LRInteractionAttribution(per_class_lr=pd.DataFrame())

    celltype_global = adata.obs[celltype_col].astype(str).to_numpy()
    expr_full = adata.layers.get(expression_layer, adata.X)

    rows: list[dict] = []
    for cls_idx, igs in ig_results_by_class.items():
        datas = explained_data_by_class.get(cls_idx, [])
        if not igs or not datas:
            continue
        ig_by_nid = {ig.niche_id: ig for ig in igs}

        for data in datas:
            ig = ig_by_nid.get(int(data.niche_id))
            if ig is None or ig.edge_attr_attrs is None:
                continue

            # Sum |edge attr attribution| across channels → edge importance.
            edge_imp = np.abs(ig.edge_attr_attrs).sum(axis=1)  # (n_edges,)
            if edge_imp.size == 0:
                continue
            n_edges = edge_imp.shape[0]
            n_top = min(top_k_edges_per_niche, n_edges)
            top_idx = np.argpartition(-edge_imp, n_top - 1)[:n_top]

            edge_index = data.edge_index.cpu().numpy()
            global_ix = data.global_cell_indices.cpu().numpy()

            # Densify only the L/R gene columns for this niche.
            niche_expr = expr_full[global_ix, :]
            if sp.issparse(niche_expr):
                L_mat = niche_expr[:, lr_resource.ligand_gene_idx].toarray()
                R_mat = niche_expr[:, lr_resource.receptor_gene_idx].toarray()
            else:
                L_mat = np.asarray(niche_expr[:, lr_resource.ligand_gene_idx], dtype=np.float32)
                R_mat = np.asarray(niche_expr[:, lr_resource.receptor_gene_idx], dtype=np.float32)

            for ei in top_idx:
                a_local, b_local = int(edge_index[0, ei]), int(edge_index[1, ei])
                if a_local >= b_local:
                    # Skip the reverse-direction duplicate of the same undirected edge.
                    continue
                w = float(edge_imp[ei])
                if w <= 0:
                    continue
                ct_a = str(celltype_global[global_ix[a_local]])
                ct_b = str(celltype_global[global_ix[b_local]])

                # Per-LR-pair scores (bidirectional).
                fwd = L_mat[a_local] * R_mat[b_local]
                bwd = L_mat[b_local] * R_mat[a_local]
                pair_score = fwd + bwd  # (n_pairs,)
                n_lr = pair_score.shape[0]
                k_lr = min(top_k_lr_pairs_per_edge, n_lr)
                if k_lr <= 0:
                    continue
                top_lr_idx = np.argpartition(-pair_score, k_lr - 1)[:k_lr]

                # canonicalise CT-pair order to make sender/receiver symmetric
                ct_pair = tuple(sorted((ct_a, ct_b)))
                for li in top_lr_idx:
                    if pair_score[li] <= 0:
                        continue
                    rows.append(
                        {
                            "class_idx": cls_idx,
                            "class_label": (
                                class_label_map[cls_idx]
                                if cls_idx < len(class_label_map)
                                else str(cls_idx)
                            ),
                            "ct_pair_a": ct_pair[0],
                            "ct_pair_b": ct_pair[1],
                            "ligand": lr_resource.ligand_names[int(li)],
                            "receptor": lr_resource.receptor_names[int(li)],
                            "weighted_score": float(w * pair_score[li]),
                            "edge_importance": w,
                            "lr_score": float(pair_score[li]),
                        }
                    )

    if not rows:
        return LRInteractionAttribution(per_class_lr=pd.DataFrame())

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(
            ["class_idx", "class_label", "ct_pair_a", "ct_pair_b", "ligand", "receptor"],
            observed=True,
            as_index=False,
        )
        .agg(weighted_score=("weighted_score", "sum"), n_edges=("weighted_score", "size"))
        .sort_values(["class_idx", "weighted_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return LRInteractionAttribution(per_class_lr=agg)
