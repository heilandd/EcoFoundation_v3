"""Biological deep-dive per niche cluster (Step 7).

For every niche cluster produced by the unsupervised GAE/DGI run, compute:

  - **Pathway enrichment** on the cluster's top marker genes (Enrichr).
  - **Ligand-receptor interactions** aggregated over the cluster's niches,
    annotated with sender/receiver cell types.
  - **Example niches** that best represent the cluster (for spatial overlays).

The functions in this module are pipeline-agnostic — they take the cluster
labels + raw inputs they need and return tidy DataFrames + dicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.data import Data

from ecofoundation.graph.lr_scoring import LRResource
from ecofoundation.niches.base import NicheAssignment
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class ClusterBiology:
    """Aggregated outputs of :func:`compute_cluster_biology`."""

    pathways: pd.DataFrame  # cluster × pathway
    lr_interactions: pd.DataFrame  # cluster × ligand × receptor × ct_pair
    example_niches: dict[int, list[int]]  # cluster → list of niche_ids


# ---------------------------------------------------------------------------
# Pathway enrichment
# ---------------------------------------------------------------------------


def compute_cluster_pathway_enrichment(
    adata: ad.AnnData,
    markers: pd.DataFrame,
    *,
    n_top_genes: int = 50,
    gene_sets: str = "MSigDB_Hallmark_2020",
    n_terms_per_cluster: int = 15,
) -> pd.DataFrame:
    """Enrichr-based pathway enrichment of each cluster's top marker genes.

    ``markers`` is the long DataFrame from
    :func:`ecofoundation.niches.cluster_characterization.characterize_niche_clusters`
    (cols: cluster, rank, gene, score).
    """
    if markers.empty:
        return pd.DataFrame(
            columns=[
                "cluster", "gene_set", "term", "overlap",
                "p_value", "adjusted_p_value", "combined_score", "genes",
            ]
        )

    try:
        import gseapy as gp
    except ImportError:
        _log.warning("gseapy not installed — skipping pathway enrichment.")
        return pd.DataFrame()

    background = list(adata.var_names)
    rows: list[dict] = []
    for cl in sorted(markers["cluster"].unique()):
        top_genes = (
            markers[markers["cluster"] == cl]
            .sort_values("rank")
            .head(n_top_genes)["gene"]
            .tolist()
        )
        if not top_genes:
            continue
        try:
            enr = gp.enrichr(
                gene_list=top_genes,
                gene_sets=[gene_sets],
                background=background,
                organism="human",
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            res = enr.results
        except Exception as e:  # noqa: BLE001
            _log.warning(f"Enrichr failed for cluster {cl}: {e}")
            continue

        if res is None or res.empty:
            continue
        for _, r in res.head(n_terms_per_cluster).iterrows():
            rows.append(
                {
                    "cluster": int(cl),
                    "gene_set": r.get("Gene_set", gene_sets),
                    "term": r.get("Term", ""),
                    "overlap": r.get("Overlap", ""),
                    "p_value": float(r.get("P-value", float("nan"))),
                    "adjusted_p_value": float(r.get("Adjusted P-value", float("nan"))),
                    "combined_score": float(r.get("Combined Score", float("nan"))),
                    "genes": r.get("Genes", ""),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["cluster", "adjusted_p_value"]).reset_index(drop=True)
    _log.info(f"Cluster pathway enrichment: {len(df)} terms across {df['cluster'].nunique() if not df.empty else 0} clusters")
    return df


# ---------------------------------------------------------------------------
# LR interactions per cluster
# ---------------------------------------------------------------------------


def compute_cluster_lr_interactions(
    adata: ad.AnnData,
    niches: NicheAssignment,
    graphs: list[Data],
    *,
    cluster_labels: np.ndarray,
    lr_resource: LRResource,
    expression_layer: str,
    celltype_col: str,
    n_niches_per_cluster: int = 50,
    top_k_lr_per_edge: int = 5,
    top_k_per_cluster: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate LR pair scores over each cluster's niches.

    For each cluster: subsample up to ``n_niches_per_cluster`` niches, compute
    bidirectional LR pair scores per intra-niche edge, retain the top-K LR
    pairs per edge, group by cell-type pair × LR pair, sum the scores.

    Returns a long DataFrame with columns ``cluster, ct_pair_a, ct_pair_b,
    ligand, receptor, score_sum, n_edges``.
    """
    if (
        celltype_col not in adata.obs.columns
        or lr_resource.n_pairs_kept == 0
        or niches.n_niches == 0
    ):
        return pd.DataFrame()

    celltype = adata.obs[celltype_col].astype(str).to_numpy()
    expr_full = adata.layers.get(expression_layer, adata.X)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for cl in sorted(np.unique(cluster_labels).tolist()):
        cluster_nids = np.where(cluster_labels == cl)[0]
        if len(cluster_nids) > n_niches_per_cluster:
            cluster_nids = rng.choice(
                cluster_nids, size=n_niches_per_cluster, replace=False
            )
        for nid in cluster_nids:
            data = graphs[int(nid)]
            global_cells = niches.cells_per_niche[int(nid)]
            edges = data.edge_index.cpu().numpy()
            if edges.shape[1] == 0:
                continue
            niche_expr = expr_full[global_cells, :]
            if sp.issparse(niche_expr):
                L_mat = niche_expr[:, lr_resource.ligand_gene_idx].toarray()
                R_mat = niche_expr[:, lr_resource.receptor_gene_idx].toarray()
            else:
                L_mat = np.asarray(
                    niche_expr[:, lr_resource.ligand_gene_idx], dtype=np.float32
                )
                R_mat = np.asarray(
                    niche_expr[:, lr_resource.receptor_gene_idx], dtype=np.float32
                )

            for k in range(edges.shape[1]):
                a, b = int(edges[0, k]), int(edges[1, k])
                if a >= b:
                    continue
                pair_score = L_mat[a] * R_mat[b] + L_mat[b] * R_mat[a]
                n_lr = pair_score.shape[0]
                kk = min(top_k_lr_per_edge, n_lr)
                if kk <= 0:
                    continue
                top_idx = np.argpartition(-pair_score, kk - 1)[:kk]
                ct_a = celltype[global_cells[a]]
                ct_b = celltype[global_cells[b]]
                pair = sorted((str(ct_a), str(ct_b)))
                for li in top_idx:
                    if pair_score[li] <= 0:
                        continue
                    rows.append(
                        {
                            "cluster": int(cl),
                            "ct_pair_a": pair[0],
                            "ct_pair_b": pair[1],
                            "ligand": lr_resource.ligand_names[int(li)],
                            "receptor": lr_resource.receptor_names[int(li)],
                            "score": float(pair_score[li]),
                        }
                    )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(
            ["cluster", "ct_pair_a", "ct_pair_b", "ligand", "receptor"],
            observed=True, as_index=False,
        )
        .agg(score_sum=("score", "sum"), n_edges=("score", "size"))
    )
    # Top-K per cluster
    out = (
        agg.sort_values(["cluster", "score_sum"], ascending=[True, False])
        .groupby("cluster", group_keys=False)
        .head(top_k_per_cluster)
        .reset_index(drop=True)
    )
    _log.info(
        f"Cluster LR interactions: {len(out)} entries across {out['cluster'].nunique()} clusters"
    )
    return out


# ---------------------------------------------------------------------------
# Example niches per cluster
# ---------------------------------------------------------------------------


def pick_example_niches_per_cluster(
    niches: NicheAssignment,
    *,
    cluster_labels: np.ndarray,
    n_per_cluster: int = 3,
    seed: int = 0,
) -> dict[int, list[int]]:
    """For each cluster, pick ``n_per_cluster`` representative niche ids.

    Strategy: median-sized niches drawn deterministically (with a seeded RNG)
    from the cluster — gives a visually-typical example rather than outliers.
    """
    rng = np.random.default_rng(seed)
    sizes = niches.sizes()
    picks: dict[int, list[int]] = {}
    for cl in sorted(np.unique(cluster_labels).tolist()):
        cluster_nids = np.where(cluster_labels == cl)[0]
        if cluster_nids.size == 0:
            continue
        cl_sizes = sizes[cluster_nids]
        med = float(np.median(cl_sizes))
        order = np.argsort(np.abs(cl_sizes - med))
        top = cluster_nids[order[: max(n_per_cluster * 3, n_per_cluster)]]
        if len(top) > n_per_cluster:
            chosen = rng.choice(top, size=n_per_cluster, replace=False)
        else:
            chosen = top
        picks[int(cl)] = [int(x) for x in chosen]
    return picks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_cluster_biology(
    adata: ad.AnnData,
    niches: NicheAssignment,
    graphs: list[Data],
    *,
    cluster_labels: np.ndarray,
    markers: pd.DataFrame,
    lr_resource: LRResource,
    expression_layer: str,
    celltype_col: str | None,
    pathway_gene_sets: str = "MSigDB_Hallmark_2020",
    n_top_genes_for_pathway: int = 50,
    n_niches_per_cluster_for_lr: int = 50,
    n_example_niches_per_cluster: int = 3,
    seed: int = 0,
) -> ClusterBiology:
    """Run the three sub-analyses and bundle them."""
    pathways = compute_cluster_pathway_enrichment(
        adata, markers, n_top_genes=n_top_genes_for_pathway, gene_sets=pathway_gene_sets,
    )
    lr_df = (
        compute_cluster_lr_interactions(
            adata, niches, graphs,
            cluster_labels=cluster_labels,
            lr_resource=lr_resource,
            expression_layer=expression_layer,
            celltype_col=celltype_col,
            n_niches_per_cluster=n_niches_per_cluster_for_lr,
            seed=seed,
        )
        if celltype_col
        else pd.DataFrame()
    )
    examples = pick_example_niches_per_cluster(
        niches, cluster_labels=cluster_labels, n_per_cluster=n_example_niches_per_cluster, seed=seed,
    )
    return ClusterBiology(pathways=pathways, lr_interactions=lr_df, example_niches=examples)
