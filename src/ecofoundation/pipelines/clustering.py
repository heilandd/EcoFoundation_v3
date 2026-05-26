"""Hello-World clustering pipeline.

End-to-end orchestrator for the **unsupervised, classical** clustering path:

    load -> QC stats -> (optional preprocess) -> Leiden -> markers -> report

This is the Step-1 pipeline. The custom spatial-GNN clustering variant
(Step 6) plugs into the same orchestrator by swapping ``run_leiden`` for the
GNN clustering call.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from ecofoundation.clustering.classical import (
    LEIDEN_OBS_KEY,
    UMAP_OBSM_KEY,
    cluster_composition,
    run_leiden,
)
from ecofoundation.clustering.markers import compute_marker_genes
from ecofoundation.config.schemas import RunConfig
from ecofoundation.io.readers import load_anndata, validate_schema
from ecofoundation.io.writers import RunFolder, create_run_folder
from ecofoundation.preprocessing.qc_stats import compute_qc_stats
from ecofoundation.reporting.plots import (
    composition_bar,
    marker_dotplot,
    qc_distributions_figure,
    spatial_figure,
    top_markers_heatmap,
    umap_figure,
)
from ecofoundation.reporting.report import ReportBuilder
from ecofoundation.utils.device import resolve_device
from ecofoundation.utils.logging import configure_logging, get_logger
from ecofoundation.utils.seeding import set_global_seed

_log = get_logger(__name__)


def run_clustering_pipeline(cfg: RunConfig) -> RunFolder:
    """Run the Hello-World clustering pipeline end-to-end.

    Returns the :class:`RunFolder` handle pointing at the persisted artifacts.
    """
    # ---- bootstrap ----------------------------------------------------------
    set_global_seed(cfg.seed)
    folder = create_run_folder(cfg)
    configure_logging(level="INFO", log_file=folder.log_path)
    device = resolve_device(cfg.device)
    _log.info(f"Run id: {folder.run_id} | device: {device}")

    pdf_dir = folder.root / "pdf"
    rb = ReportBuilder(run_name=cfg.run_name, cfg=cfg, pdf_dir=pdf_dir)

    # ---- 1. load ------------------------------------------------------------
    adata = load_anndata(cfg.data)
    schema = validate_schema(adata, cfg.data)

    rb.add_overview(
        {
            "Cells": adata.shape[0],
            "Genes": adata.shape[1],
            "Samples": schema.n_samples,
            "Patients": schema.n_patients,
            "Spatial coords": "obsm['" + cfg.data.spatial_key + "']",
            "Embedding": cfg.data.embedding_key or "(none)",
            "Counts layer": cfg.data.counts_layer or "(adata.X)",
            "Cell-type col": cfg.data.celltype_col or "(none)",
            "Device": device,
        },
        title="Run overview",
    )

    # ---- 2. QC stats (descriptive only) -------------------------------------
    try:
        qc_stats = compute_qc_stats(adata, cfg.data)
        rb.add_table(
            "QC summary per sample",
            qc_stats.to_records(),
            description="Median / mean counts and detected genes per sample.",
        )
        rb.add_plot(
            "QC distributions",
            qc_distributions_figure(adata, cfg.data.sample_id_col, qc_stats),
            description="Counts (log scale) and detected genes per cell, split by sample.",
        )
    except Exception as e:  # surface QC failure but keep running
        _log.warning(f"QC stats failed: {e}")
        rb.add_text("QC summary", f"QC computation failed: {e}")

    # ---- 3. (cell-type overview, if present) --------------------------------
    if cfg.data.celltype_col and cfg.data.celltype_col in adata.obs.columns:
        _log.info("Adding cell-type spatial / UMAP overview from existing annotation.")
        rb.add_plot(
            "UMAP — pre-existing cell-type annotation",
            umap_figure(
                adata,
                color_key=cfg.data.celltype_col,
                obsm_key="X_umap" if "X_umap" in adata.obsm else cfg.data.embedding_key or "X_umap",
                max_points=cfg.report.spatial_max_points,
            ),
            description=f"Coloured by adata.obs['{cfg.data.celltype_col}'] (provided in the input).",
        )
        rb.add_plot(
            "Spatial — pre-existing cell-type annotation",
            spatial_figure(
                adata,
                color_key=cfg.data.celltype_col,
                sample_key=cfg.data.sample_id_col,
                spatial_key=cfg.data.spatial_key,
                max_points_per_sample=cfg.report.spatial_max_points // max(schema.n_samples or 1, 1),
            ),
        )

    # ---- 4. Leiden ----------------------------------------------------------
    if cfg.leiden.enabled:
        leiden = run_leiden(adata, cfg.leiden)
        rb.add_text(
            "Leiden clustering",
            (
                f"Computed Leiden on obsm['{leiden.use_rep}'] using a kNN graph "
                f"with n_neighbors={leiden.n_neighbors} and resolution={leiden.resolution}. "
                f"Found {leiden.n_clusters} clusters; labels in obs['{leiden.obs_key}']."
            ),
            description="Classical clustering baseline.",
        )
        rb.add_plot(
            "UMAP — Leiden clusters",
            umap_figure(
                adata,
                color_key=LEIDEN_OBS_KEY,
                obsm_key=UMAP_OBSM_KEY,
                max_points=cfg.report.spatial_max_points,
                title=f"UMAP coloured by Leiden (res={leiden.resolution})",
            ),
            parameters={
                "resolution": leiden.resolution,
                "n_neighbors": leiden.n_neighbors,
                "use_rep": leiden.use_rep,
            },
        )
        rb.add_plot(
            "Spatial — Leiden clusters",
            spatial_figure(
                adata,
                color_key=LEIDEN_OBS_KEY,
                sample_key=cfg.data.sample_id_col,
                spatial_key=cfg.data.spatial_key,
                max_points_per_sample=cfg.report.spatial_max_points // max(schema.n_samples or 1, 1),
                title="Leiden clusters in spatial coords",
            ),
        )

        # Composition
        comp = cluster_composition(adata, LEIDEN_OBS_KEY, cfg.data.sample_id_col)
        rb.add_plot(
            "Cluster composition per sample",
            composition_bar(comp, sample_col=cfg.data.sample_id_col, cluster_col=LEIDEN_OBS_KEY),
        )

        # ---- 5. Markers -----------------------------------------------------
        if cfg.markers.enabled:
            marker = compute_marker_genes(
                adata,
                cluster_key=LEIDEN_OBS_KEY,
                cfg=cfg.markers,
                layer=cfg.data.normalized_layer,
            )
            rb.add_table(
                f"Top {cfg.markers.n_top} marker genes per cluster",
                marker.table.to_dict("records"),
                description=f"rank_genes_groups({cfg.markers.method})",
                parameters={"method": marker.method, "n_top": marker.n_top},
            )
            rb.add_plot(
                "Marker dotplot",
                marker_dotplot(
                    adata,
                    marker.table,
                    cluster_key=LEIDEN_OBS_KEY,
                    n_top=3,
                    layer=cfg.data.normalized_layer,
                ),
            )
            rb.add_plot(
                "Top marker scores heatmap",
                top_markers_heatmap(marker.table, n_top=5),
            )

            # Persist markers
            marker.table.to_csv(folder.artifact("marker_genes.csv"), index=False)

    # ---- 6. Persist + write report -----------------------------------------
    rb.write(folder.report_path)
    _log.info(f"Pipeline complete: {folder.report_path}")
    return folder


def run_from_yaml(config_path: Path | str) -> RunFolder:
    """Convenience: load a YAML config and run the pipeline."""
    from ecofoundation.config.loader import load_config

    cfg = load_config(config_path)
    return run_clustering_pipeline(cfg)
