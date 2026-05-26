"""Niche-graph-construction pipeline (Step 3).

Builds niches AND per-niche PyG graphs, then writes a report covering both
stages. Graphs are persisted as a ``.pt`` file for downstream Step-4 training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecofoundation.config.schemas import RunConfig
from ecofoundation.graph.construction import build_niche_graphs
from ecofoundation.io.readers import load_anndata, validate_schema
from ecofoundation.io.writers import RunFolder, create_run_folder
from ecofoundation.niches.assembly import assign_niches
from ecofoundation.niches.characterization import compute_niche_stats
from ecofoundation.reporting.plots import (
    co_occurrence_heatmap,
    edge_feature_distributions,
    graph_size_distributions,
    heterogeneity_histogram,
    niche_centroids_spatial,
    niche_density_histogram,
    niche_graph_figure,
    niche_size_distribution,
    niches_per_group_bar,
    size_vs_density_scatter,
)
from ecofoundation.reporting.report import ReportBuilder
from ecofoundation.utils.device import resolve_device
from ecofoundation.utils.logging import configure_logging, get_logger
from ecofoundation.utils.seeding import set_global_seed

_log = get_logger(__name__)


def run_graph_construction_pipeline(cfg: RunConfig) -> RunFolder:
    set_global_seed(cfg.seed)
    folder = create_run_folder(cfg)
    configure_logging(level="INFO", log_file=folder.log_path)
    device = resolve_device(cfg.device)
    _log.info(f"Run id: {folder.run_id} | device: {device}")

    pdf_dir = folder.root / "pdf"
    rb = ReportBuilder(run_name=cfg.run_name, cfg=cfg, pdf_dir=pdf_dir)

    adata = load_anndata(cfg.data)
    schema = validate_schema(adata, cfg.data)

    rb.add_overview(
        {
            "Cells": adata.shape[0],
            "Genes": adata.shape[1],
            "Samples": schema.n_samples,
            "Patients": schema.n_patients,
            "Strategy": cfg.niches.strategy,
            "Overlap filter": cfg.niches.overlap_filter_enabled,
            "k_hop": cfg.niches.k_hop,
            "Node features": cfg.graph.node_feature_source,
            "Gene subset": cfg.graph.gene_subset,
            "Edge topology": cfg.graph.edge_topology,
            "LR scoring": cfg.graph.lr_scoring.enabled,
            "Device": device,
        },
        title="Graph construction overview",
    )

    # ----- niches ------------------------------------------------------------
    niches, overlap_info = assign_niches(adata, cfg.data, cfg.niches)
    summary = niches.summary()

    rb.add_text(
        "Niche construction",
        (
            f"Strategy: {niches.strategy_name}. Kept {niches.n_niches} niches "
            f"({len(overlap_info.dropped_indices)} dropped by overlap filter)."
        ),
    )
    rb.add_table(
        "Niche summary",
        rows=[{"metric": k, "value": str(v)} for k, v in summary.items() if k != "niches_per_group"],
        parameters=summary["params"],
    )
    if niches.n_niches > 0:
        rb.add_plot("Niche size distribution", niche_size_distribution(niches))
        rb.add_plot("Niches per patient", niches_per_group_bar(niches))
        rb.add_plot(
            "Niche centroids on spatial",
            niche_centroids_spatial(
                adata,
                niches,
                sample_key=cfg.data.sample_id_col,
                spatial_key=cfg.data.spatial_key,
                cell_sample=6000,
            ),
        )

        # Standardised characterisation (Step 2.5)
        stats = compute_niche_stats(adata, niches, cfg.data)
        rb.add_table(
            "Characterisation summary",
            rows=[{"metric": k, "value": str(v)} for k, v in stats.summary().items()],
        )
        if not stats.per_niche.empty:
            rb.add_plot("Center vs neighbor co-occurrence", co_occurrence_heatmap(stats))
            rb.add_plot("Cellular density per niche", niche_density_histogram(stats))
            rb.add_plot("Niche heterogeneity (Shannon)", heterogeneity_histogram(stats))
            rb.add_plot("Niche size vs density", size_vs_density_scatter(stats))
            stats.per_niche.to_parquet(folder.artifact("niche_characterization.parquet"))
            stats.co_occurrence.to_csv(folder.artifact("co_occurrence_matrix.csv"))

    # ----- graphs ------------------------------------------------------------
    if niches.n_niches == 0:
        rb.write(folder.report_path)
        _log.warning("No niches — skipping graph construction.")
        return folder

    result = build_niche_graphs(adata, niches, cfg.data, cfg.graph)
    _log.info(f"Built {len(result.graphs)} PyG graphs | summary: {result.summary}")

    rb.add_text(
        "Graph construction",
        (
            f"Built {len(result.graphs)} PyG `Data` objects. "
            f"Node features = {result.summary['n_node_features']}-dim ({cfg.graph.node_feature_source}). "
            f"Edge features = {result.summary['n_edge_features']}-dim "
            f"({', '.join(result.edge_feature_names)}). "
            f"LR pairs used: {result.summary['lr_pairs_used']}."
        ),
        parameters={
            "node_feature_source": cfg.graph.node_feature_source,
            "gene_subset": cfg.graph.gene_subset,
            "edge_topology": cfg.graph.edge_topology,
            "lr_resource": cfg.graph.lr_scoring.resource if cfg.graph.lr_scoring.enabled else None,
        },
    )
    rb.add_table(
        "Graph construction summary",
        rows=[{"metric": k, "value": str(v)} for k, v in result.summary.items()],
    )
    rb.add_plot("Niche-graph size distributions", graph_size_distributions(result.graphs))

    # Edge-feature distributions: stack the per-niche edge_attr matrices
    if result.graphs:
        all_attrs = torch.cat([g.edge_attr for g in result.graphs if g.edge_attr is not None], dim=0)
        rb.add_plot(
            "Edge feature distributions",
            edge_feature_distributions(all_attrs.cpu().numpy(), result.edge_feature_names),
        )

    # Visualize a handful of representative niches
    example_graphs = _pick_example_graphs(result.graphs, k=3)
    for g in example_graphs:
        # Attach pos for plotting from the actual cell coords
        coords = np.asarray(adata.obsm[cfg.data.spatial_key])[:, :2]
        g.pos = coords[g.global_cell_indices.cpu().numpy()]
        rb.add_plot(
            f"Example niche graph #{int(g.niche_id)} (patient {g.patient})",
            niche_graph_figure(
                g,
                edge_feature_index=min(1, g.edge_attr.shape[1] - 1),
                edge_feature_name=(
                    result.edge_feature_names[min(1, len(result.edge_feature_names) - 1)]
                ),
                title=(
                    f"Niche #{int(g.niche_id)} — patient={g.patient} "
                    f"sample={g.sample} n={g.num_nodes} cells"
                ),
            ),
        )

    # ----- persistence -------------------------------------------------------
    # Save graphs + the per-niche metadata table
    graphs_path = folder.artifact("niche_graphs.pt")
    torch.save(
        {
            "graphs": result.graphs,
            "node_feature_names": result.node_feature_names,
            "edge_feature_names": result.edge_feature_names,
            "summary": result.summary,
        },
        graphs_path,
    )
    meta_rows = []
    for g in result.graphs:
        meta_rows.append(
            {
                "niche_id": int(g.niche_id),
                "patient": str(g.patient),
                "sample": str(g.sample),
                "n_nodes": int(g.num_nodes),
                "n_edges_undirected": int(g.edge_index.shape[1] // 2),
            }
        )
    pd.DataFrame(meta_rows).to_parquet(folder.artifact("niche_graphs_metadata.parquet"))

    rb.write(folder.report_path)
    _log.info(f"Pipeline complete: {folder.report_path}")
    return folder


def _pick_example_graphs(graphs, k: int = 3):
    """Pick representative graphs covering different patients / edge counts.

    Strategy: pick the niche with the median edge count from each of up to ``k``
    distinct patients. Falls back to first ``k`` if patient diversity is limited.
    """
    if not graphs:
        return []

    by_patient: dict = {}
    for g in graphs:
        by_patient.setdefault(str(g.patient), []).append(g)

    picks: list = []
    for patient in sorted(by_patient.keys())[:k]:
        bucket = by_patient[patient]
        edges = np.array([gg.edge_index.shape[1] // 2 for gg in bucket])
        med_idx = int(np.argmin(np.abs(edges - np.median(edges))))
        picks.append(bucket[med_idx])
    if len(picks) < k:
        # pad from remaining graphs deterministically
        remaining = [g for g in graphs if g not in picks]
        picks.extend(remaining[: k - len(picks)])
    return picks[:k]
