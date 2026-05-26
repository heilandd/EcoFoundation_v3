"""Per-cell-type pathway enrichment of top-attributed genes.

For each (class, cell type) pair we collect the cells of that type across
all explained niches, rank genes by mean |IG attribution| of those cells,
and run an Enrichr-style overlap test against curated gene-set databases.

Default database: ``MSigDB_Hallmark_2020`` (50 cancer hallmark sets, small +
fast). Other useful gene-set names you can pass via the config:
``Reactome_Pathways_2024``, ``GO_Biological_Process_2023``, ``KEGG_2021_Human``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from ecofoundation.interpretation.integrated_gradients import IGResult
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class PathwayEnrichmentResult:
    per_class_celltype: pd.DataFrame
    # cols: class_idx, class_label, cell_type, gene_set, term,
    #       overlap, p_value, adjusted_p_value, combined_score, genes


def compute_pathway_enrichment(
    *,
    adata: ad.AnnData,
    celltype_col: str,
    feature_names: list[str],
    ig_results_by_class: dict[int, list[IGResult]],
    explained_data_by_class: dict[int, list[Data]],
    class_label_map: list[str],
    n_top_genes: int = 100,
    gene_sets: str | list[str] = "MSigDB_Hallmark_2020",
    use_enrichr: bool = True,
    background_genes: list[str] | None = None,
) -> PathwayEnrichmentResult:
    """Run enrichment per (class, cell type)."""
    if celltype_col not in adata.obs.columns:
        return PathwayEnrichmentResult(per_class_celltype=pd.DataFrame())

    celltype_global = adata.obs[celltype_col].astype(str).to_numpy()
    background = background_genes or feature_names

    rows: list[dict] = []
    for cls_idx, igs in ig_results_by_class.items():
        datas = explained_data_by_class.get(cls_idx, [])
        if not igs or not datas:
            continue
        ig_by_nid = {ig.niche_id: ig for ig in igs}

        # Accumulate per-(celltype, gene) abs attribution scores
        ct_gene_acc: dict[str, np.ndarray] = {}
        ct_gene_count: dict[str, int] = {}
        for data in datas:
            ig = ig_by_nid.get(int(data.niche_id))
            if ig is None:
                continue
            attrs = np.abs(ig.node_attrs)  # (n_nodes, n_features)
            global_ix = data.global_cell_indices.cpu().numpy()
            cts = celltype_global[global_ix]
            for k, ct in enumerate(cts):
                ct_str = str(ct)
                if ct_str not in ct_gene_acc:
                    ct_gene_acc[ct_str] = np.zeros(attrs.shape[1], dtype=np.float64)
                    ct_gene_count[ct_str] = 0
                ct_gene_acc[ct_str] += attrs[k]
                ct_gene_count[ct_str] += 1

        # Run enrichment per cell type
        for ct, score_sum in ct_gene_acc.items():
            n_cells = ct_gene_count[ct]
            if n_cells < 5:
                continue
            mean_score = score_sum / n_cells
            order = np.argsort(-mean_score)
            top_genes = [feature_names[i] for i in order[:n_top_genes]]

            enr_df = _run_enrichr_safely(
                top_genes,
                gene_sets=gene_sets if isinstance(gene_sets, list) else [gene_sets],
                background=background,
                use_enrichr=use_enrichr,
            )
            if enr_df is None or enr_df.empty:
                continue

            class_label = (
                class_label_map[cls_idx] if cls_idx < len(class_label_map) else str(cls_idx)
            )
            for _, r in enr_df.head(15).iterrows():
                rows.append(
                    {
                        "class_idx": cls_idx,
                        "class_label": class_label,
                        "cell_type": ct,
                        "gene_set": r.get("Gene_set", ""),
                        "term": r.get("Term", ""),
                        "overlap": r.get("Overlap", ""),
                        "p_value": float(r.get("P-value", float("nan"))),
                        "adjusted_p_value": float(r.get("Adjusted P-value", float("nan"))),
                        "combined_score": float(r.get("Combined Score", float("nan"))),
                        "genes": r.get("Genes", ""),
                        "n_cells_supporting": n_cells,
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["class_idx", "cell_type", "adjusted_p_value"]).reset_index(drop=True)
    return PathwayEnrichmentResult(per_class_celltype=df)


def _run_enrichr_safely(
    top_genes: list[str], *, gene_sets: list[str], background: list[str], use_enrichr: bool
):
    """Wrap gseapy.enrichr with graceful failure (no Internet, empty result, ...)."""
    if not use_enrichr or not top_genes:
        return None
    try:
        import gseapy as gp

        enr = gp.enrichr(
            gene_list=top_genes,
            gene_sets=gene_sets,
            background=background,
            organism="human",
            outdir=None,
            no_plot=True,
            verbose=False,
        )
        return enr.results
    except Exception as e:  # noqa: BLE001 — Enrichr is online; many failure modes
        _log.warning(f"Enrichr failed ({type(e).__name__}: {e}); skipping pathway enrichment.")
        return None
