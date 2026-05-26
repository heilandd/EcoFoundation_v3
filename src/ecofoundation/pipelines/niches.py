"""Niche-construction + characterisation pipeline (Step 2 + 2.5).

Builds niches, computes the standardised characterisation statistics, and
writes a self-contained matplotlib/SVG report plus per-plot PDF artefacts
(Illustrator-editable) under ``runs/<run_id>/pdf/``.
"""

from __future__ import annotations

import pandas as pd

from ecofoundation.config.schemas import RunConfig
from ecofoundation.io.readers import load_anndata, validate_schema
from ecofoundation.io.writers import RunFolder, create_run_folder
from ecofoundation.niches.assembly import assign_niches
from ecofoundation.niches.characterization import compute_niche_stats
from ecofoundation.reporting.plots import (
    center_purity_by_celltype,
    co_occurrence_heatmap,
    heterogeneity_histogram,
    n_unique_celltypes_histogram,
    niche_centroids_spatial,
    niche_density_histogram,
    niche_size_distribution,
    niches_per_group_bar,
    size_vs_density_scatter,
)
from ecofoundation.reporting.report import ReportBuilder
from ecofoundation.utils.device import resolve_device
from ecofoundation.utils.logging import configure_logging, get_logger
from ecofoundation.utils.seeding import set_global_seed

_log = get_logger(__name__)


def run_niche_pipeline(cfg: RunConfig) -> RunFolder:
    """Build niches + characterisation and write a report."""
    set_global_seed(cfg.seed)
    folder = create_run_folder(cfg)
    configure_logging(level="INFO", log_file=folder.log_path)
    device = resolve_device(cfg.device)
    _log.info(f"Run id: {folder.run_id} | device: {device}")

    pdf_dir = folder.root / "pdf"
    rb = ReportBuilder(run_name=cfg.run_name, cfg=cfg, pdf_dir=pdf_dir)

    # ----- load --------------------------------------------------------------
    adata = load_anndata(cfg.data)
    schema = validate_schema(adata, cfg.data)

    rb.add_overview(
        {
            "Cells": adata.shape[0],
            "Genes": adata.shape[1],
            "Samples": schema.n_samples,
            "Patients": schema.n_patients,
            "Strategy": cfg.niches.strategy,
            "knn_k": cfg.niches.knn_k if cfg.niches.strategy == "knn" else "—",
            "Overlap filter": cfg.niches.overlap_filter_enabled,
            "Max overlap fraction": cfg.niches.max_overlap_fraction,
            "Cell-type col": cfg.data.celltype_col or "(none)",
            "Device": device,
        },
        title="Niche construction overview",
    )

    # ----- niches ------------------------------------------------------------
    niches, overlap_info = assign_niches(adata, cfg.data, cfg.niches)

    rb.add_text(
        "Niche construction",
        (
            f"Strategy: {niches.strategy_name}. Built {niches.n_niches} niches "
            f"across {len(set(niches.group_label))} patients. "
            + (
                "Overlap filter disabled — every ego cell kept (unsupervised mode)."
                if not cfg.niches.overlap_filter_enabled
                else (
                    f"Overlap controller (cap={overlap_info.max_overlap_fraction:.2f}) "
                    f"kept {len(overlap_info.kept_indices)} / dropped "
                    f"{len(overlap_info.dropped_indices)} candidates."
                )
            )
        ),
    )
    summary = niches.summary()
    rb.add_table(
        "Niche assembly summary",
        rows=[
            {"metric": k, "value": str(v)}
            for k, v in summary.items()
            if k not in ("niches_per_group", "params")
        ],
        parameters=summary["params"],
    )
    rb.add_table(
        "Niches per patient",
        rows=[{"patient": p, "n_niches": n} for p, n in summary["niches_per_group"].items()],
    )

    if niches.n_niches == 0:
        rb.write(folder.report_path)
        return folder

    rb.add_plot("Niche size distribution", niche_size_distribution(niches))
    rb.add_plot("Niches per patient", niches_per_group_bar(niches))
    rb.add_plot(
        "Niche centroids on spatial",
        niche_centroids_spatial(
            adata, niches,
            sample_key=cfg.data.sample_id_col,
            spatial_key=cfg.data.spatial_key,
            cell_sample=6000,
        ),
    )

    # ----- characterisation --------------------------------------------------
    stats = compute_niche_stats(adata, niches, cfg.data)
    rb.add_text(
        "Niche characterisation",
        (
            "Standardised per-niche statistics: cellular density (mean nearest-neighbor distance), "
            "cell-type composition entropy (Shannon), center-cell-type purity, niche radius, "
            "and a center-vs-neighbor co-occurrence matrix."
        ),
    )
    rb.add_table(
        "Characterisation summary",
        rows=[{"metric": k, "value": str(v)} for k, v in stats.summary().items()],
    )
    if not stats.per_niche.empty:
        rb.add_plot("Center vs neighbor co-occurrence", co_occurrence_heatmap(stats))
        rb.add_plot("Cellular density per niche", niche_density_histogram(stats))
        rb.add_plot("Niche heterogeneity (Shannon)", heterogeneity_histogram(stats))
        rb.add_plot("# distinct cell types per niche", n_unique_celltypes_histogram(stats))
        rb.add_plot("Niche size vs density", size_vs_density_scatter(stats))
        if cfg.data.celltype_col and cfg.data.celltype_col in adata.obs.columns:
            rb.add_plot("Center-type purity by cell type", center_purity_by_celltype(stats))

    # ----- persistence -------------------------------------------------------
    long_rows = []
    for nid in range(niches.n_niches):
        for c in niches.cells_per_niche[nid].tolist():
            long_rows.append(
                {
                    "niche_id": nid,
                    "cell_index": c,
                    "ego_cell": int(niches.ego_cell[nid]),
                    "patient": str(niches.group_label[nid]),
                    "sample": (
                        str(niches.sample_label[nid])
                        if niches.sample_label is not None
                        else None
                    ),
                }
            )
    if long_rows:
        pd.DataFrame(long_rows).to_parquet(folder.artifact("niche_assignment.parquet"))
    if not stats.per_niche.empty:
        stats.per_niche.to_parquet(folder.artifact("niche_characterization.parquet"))
        stats.co_occurrence.to_csv(folder.artifact("co_occurrence_matrix.csv"))

    rb.write(folder.report_path)
    _log.info(f"Pipeline complete: {folder.report_path}")
    return folder
