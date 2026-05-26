"""Unsupervised spatial-niche clustering pipeline (Step 6).

End-to-end orchestrator:

  load -> niches (no overlap filter -> every cell gets a niche)
       -> per-niche graphs
       -> train GAE/VGAE/DGI
       -> per-niche pooled embedding
       -> UMAP + Leiden clustering on embeddings
       -> per-cluster characterisation (composition, markers, density, ...)
       -> AnnData export (cell-level: niche_cluster, niche_embedding, niche_umap)
       -> HTML/PDF report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from ecofoundation.config.schemas import RunConfig
from ecofoundation.graph.construction import build_niche_graphs
from ecofoundation.interpretation.embeddings import (
    NicheEmbeddings,
    compute_niche_umap,
)
from ecofoundation.io.anndata_export import (
    UnsupExportInputs,
    export_unsup_results_to_anndata,
)
from ecofoundation.io.readers import load_anndata, validate_schema
from ecofoundation.io.writers import RunFolder, create_run_folder
from ecofoundation.models.adversarial import AdversarialBatchHead
from ecofoundation.models.unsup import build_unsup_model
from ecofoundation.niches.assembly import assign_niches
from ecofoundation.niches.characterization import compute_niche_stats
from ecofoundation.niches.cluster_biology import compute_cluster_biology
from ecofoundation.niches.cluster_characterization import (
    characterize_niche_clusters,
)
from ecofoundation.reporting.plots import (
    cluster_lr_heatmap,
    cluster_pathway_dotplot,
    example_niche_celltypes_figure,
    niche_cluster_composition_bar,
    niche_cluster_embedding_umap,
    niche_cluster_marker_heatmap,
    niche_cluster_spatial,
    niche_size_distribution,
    niches_per_group_bar,
    unsup_training_loss,
)
from ecofoundation.reporting.report import ReportBuilder
from ecofoundation.training.unsup_trainer import (
    attach_adv_labels,
    encode_niche_embeddings,
    train_unsup,
)
from ecofoundation.utils.device import resolve_device
from ecofoundation.utils.logging import configure_logging, get_logger
from ecofoundation.utils.seeding import set_global_seed

_log = get_logger(__name__)


def run_unsup_clustering_pipeline(cfg: RunConfig) -> RunFolder:
    if cfg.unsup is None:
        raise ValueError("RunConfig.unsup is required for the unsupervised clustering pipeline.")

    set_global_seed(cfg.seed)
    folder = create_run_folder(cfg)
    configure_logging(level="INFO", log_file=folder.log_path)
    device = resolve_device(cfg.device)
    _log.info(f"Run id: {folder.run_id} | device: {device}")
    pdf_dir = folder.root / "pdf"
    rb = ReportBuilder(run_name=cfg.run_name, cfg=cfg, pdf_dir=pdf_dir)

    # ----- load + niches + graphs ------------------------------------------
    adata = load_anndata(cfg.data)
    schema = validate_schema(adata, cfg.data)

    niches, _ = assign_niches(adata, cfg.data, cfg.niches)
    if niches.n_niches == 0:
        rb.add_text("No niches", "Niche assembly returned 0 niches — aborting.")
        rb.write(folder.report_path)
        return folder
    _log.info(f"Built {niches.n_niches} niches (overlap_filter={cfg.niches.overlap_filter_enabled})")

    gr = build_niche_graphs(adata, niches, cfg.data, cfg.graph)
    _log.info(f"Built {len(gr.graphs)} PyG graphs")

    # ----- overview --------------------------------------------------------
    rb.add_overview(
        {
            "Cells": adata.shape[0],
            "Genes": adata.shape[1],
            "Samples": schema.n_samples,
            "Patients": schema.n_patients,
            "Niche strategy": cfg.niches.strategy,
            "Overlap filter": cfg.niches.overlap_filter_enabled,
            "Niches built": niches.n_niches,
            "Unsup model": cfg.unsup.model.architecture,
            "Hidden dim": cfg.unsup.model.hidden_dim,
            "Layers": cfg.unsup.model.n_layers,
            "Epochs": cfg.unsup.epochs,
            "Device": device,
        },
        title="Unsupervised niche clustering — overview",
    )
    rb.add_plot("Niche size distribution", niche_size_distribution(niches))
    rb.add_plot("Niches per patient", niches_per_group_bar(niches))

    # ----- build model -----------------------------------------------------
    model = build_unsup_model(
        cfg.unsup.model.architecture,
        node_dim=gr.summary["n_node_features"],
        edge_dim=gr.summary["n_edge_features"],
        hidden_dim=cfg.unsup.model.hidden_dim,
        n_layers=cfg.unsup.model.n_layers,
        dropout=cfg.unsup.model.dropout,
        batch_norm=cfg.unsup.model.batch_norm,
        dgi_corruption=cfg.unsup.dgi_corruption,
    )
    _log.info(f"Unsupervised model: {type(model).__name__}")

    # ----- adversarial debiasing labels per niche --------------------------
    adv_batch_labels: np.ndarray | None = None
    adv_head: AdversarialBatchHead | None = None
    if cfg.unsup.adversarial.enabled:
        adv_col = cfg.unsup.adversarial.batch_col or cfg.data.patient_id_col
        if adv_col and adv_col in adata.obs.columns:
            adv_values = adata.obs[adv_col].astype(str).to_numpy()
            niche_adv = np.array(
                [adv_values[int(niches.ego_cell[nid])] for nid in range(niches.n_niches)]
            )
            uniq_adv = sorted(set(niche_adv.tolist()))
            adv_lookup = {v: i for i, v in enumerate(uniq_adv)}
            adv_batch_labels = np.array([adv_lookup[v] for v in niche_adv], dtype=np.int64)
            adv_head = AdversarialBatchHead(
                in_dim=cfg.unsup.model.hidden_dim,
                n_batches=len(uniq_adv),
                hidden_dim=cfg.unsup.adversarial.hidden_dim,
                dropout=cfg.unsup.adversarial.dropout,
            )
            _log.info(
                f"Adversarial debiasing ENABLED — col='{adv_col}', "
                f"n_batches={len(uniq_adv)}, lambda_max={cfg.unsup.adversarial.lambda_max}"
            )
            rb.add_text(
                "Adversarial debiasing",
                (
                    f"Active — encoder is pushed AWAY from features that predict "
                    f"obs['{adv_col}'] ({len(uniq_adv)} unique values). "
                    f"lambda_max={cfg.unsup.adversarial.lambda_max}, "
                    f"warmup_epochs={cfg.unsup.adversarial.warmup_epochs}."
                ),
            )
        else:
            _log.warning(
                f"Adversarial enabled but column '{adv_col}' not in obs — disabling."
            )

    # ----- train (subsample fraction or absolute cap) ----------------------
    rng = np.random.default_rng(cfg.unsup.seed)
    n_total = len(gr.graphs)
    if cfg.unsup.subsample_fraction_for_training is not None:
        n_train = int(round(cfg.unsup.subsample_fraction_for_training * n_total))
    elif cfg.unsup.max_niches_for_training is not None:
        n_train = min(cfg.unsup.max_niches_for_training, n_total)
    else:
        n_train = n_total

    if n_train < n_total:
        idx = rng.choice(n_total, size=n_train, replace=False)
        graphs_for_training = [gr.graphs[i] for i in idx]
        adv_for_training = (
            adv_batch_labels[idx] if adv_batch_labels is not None else None
        )
        _log.info(
            f"Subsampling training set: {n_train} / {n_total} niches "
            f"(fraction={cfg.unsup.subsample_fraction_for_training}, "
            f"max={cfg.unsup.max_niches_for_training})"
        )
    else:
        graphs_for_training = gr.graphs
        adv_for_training = adv_batch_labels

    if adv_for_training is not None:
        graphs_for_training = attach_adv_labels(graphs_for_training, adv_for_training)

    rb.add_text(
        "Training plan",
        (
            f"Train {type(model).__name__} on {len(graphs_for_training)} / {n_total} niches "
            f"for {cfg.unsup.epochs} epochs. Inference runs over all {n_total} niches."
        ),
    )
    train_result = train_unsup(
        model, graphs_for_training, cfg.unsup, device=device, adv_head=adv_head
    )
    rb.add_plot("Unsupervised training loss", unsup_training_loss(train_result.history))

    # ----- extract embeddings over ALL niches -----------------------------
    embeddings_arr, niche_ids = encode_niche_embeddings(
        model, gr.graphs, batch_size=cfg.unsup.batch_size,
        device=device, pooling=cfg.unsup.model.pooling,
    )
    emb_obj = NicheEmbeddings(
        embeddings=embeddings_arr, umap_2d=None, niche_ids=niche_ids
    )

    # ----- UMAP + Leiden on embeddings ------------------------------------
    try:
        emb_obj = compute_niche_umap(
            emb_obj,
            n_neighbors=cfg.unsup.leiden_n_neighbors,
            min_dist=0.3,
            random_state=cfg.unsup.seed,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning(f"UMAP failed: {e}")

    cluster_labels = _leiden_on_embeddings(
        embeddings_arr,
        n_neighbors=cfg.unsup.leiden_n_neighbors,
        resolution=cfg.unsup.leiden_resolution,
        random_state=cfg.unsup.seed,
    )
    n_clusters = int(np.unique(cluster_labels).size)
    rb.add_text(
        "Clustering on niche embeddings",
        (
            f"Built {len(embeddings_arr)} niche embeddings ({embeddings_arr.shape[1]} dim) "
            f"and ran Leiden (n_neighbors={cfg.unsup.leiden_n_neighbors}, "
            f"resolution={cfg.unsup.leiden_resolution}) -> {n_clusters} clusters."
        ),
    )

    if emb_obj.umap_2d is not None:
        rb.add_plot(
            "Niche embedding UMAP (Leiden clusters)",
            niche_cluster_embedding_umap(emb_obj.umap_2d, cluster_labels),
        )

    # ----- per-cell niche-cluster mapping ----------------------------------
    n_cells = adata.shape[0]
    cell_cluster_str = np.array(["unassigned"] * n_cells, dtype=object)
    for nid in range(niches.n_niches):
        ego = int(niches.ego_cell[nid])
        cell_cluster_str[ego] = f"nc_{int(cluster_labels[nid])}"

    rb.add_plot(
        "Niche clusters in spatial coords",
        niche_cluster_spatial(
            adata,
            sample_key=cfg.data.sample_id_col,
            spatial_key=cfg.data.spatial_key,
            cluster_per_cell=cell_cluster_str,
        ),
    )

    # ----- characterise clusters ------------------------------------------
    base_stats = compute_niche_stats(adata, niches, cfg.data)
    cluster_stats = characterize_niche_clusters(
        adata, niches,
        cluster_labels=cluster_labels,
        celltype_col=cfg.data.celltype_col,
        sample_col=cfg.data.sample_id_col,
        niche_stats=base_stats,
        expression_layer=cfg.graph.node_expression_layer,
        n_top_marker_genes=15,
    )
    if not cluster_stats.composition.empty:
        rb.add_plot(
            "Niche cluster composition (ego cell types)",
            niche_cluster_composition_bar(cluster_stats),
        )
    if not cluster_stats.markers.empty:
        rb.add_plot(
            "Niche cluster marker genes",
            niche_cluster_marker_heatmap(cluster_stats, n_top=5),
        )
        rb.add_table(
            "Top markers per niche cluster",
            rows=cluster_stats.markers.head(200).round(3).to_dict("records"),
        )
    if not cluster_stats.cluster_summary.empty:
        rb.add_table(
            "Niche cluster summary",
            rows=cluster_stats.cluster_summary.round(3).to_dict("records"),
        )

    # ----- cluster biology deep-dive (Step 7) ------------------------------
    rb.add_text(
        "Cluster biology — pathways, LR interactions, examples",
        (
            "For each niche cluster we look up Enrichr pathway terms on its top "
            "marker genes, aggregate ligand-receptor interactions over a "
            "representative subsample of niches, and render a few example niche "
            "graphs with cell-type colouring."
        ),
    )
    biology = compute_cluster_biology(
        adata, niches, gr.graphs,
        cluster_labels=cluster_labels,
        markers=cluster_stats.markers,
        lr_resource=gr.lr_resource,
        expression_layer=cfg.graph.node_expression_layer,
        celltype_col=cfg.data.celltype_col,
        seed=cfg.seed,
    )

    if not biology.pathways.empty:
        rb.add_plot("Cluster pathway enrichment", cluster_pathway_dotplot(biology))
        rb.add_table(
            "Top pathway terms per cluster",
            rows=biology.pathways.head(200).round(4).to_dict("records"),
        )
        biology.pathways.to_parquet(folder.artifact("cluster_pathways.parquet"))

    if not biology.lr_interactions.empty:
        rb.add_plot("Top LR pairs per cluster", cluster_lr_heatmap(biology))
        rb.add_table(
            "Top LR interactions per cluster",
            rows=biology.lr_interactions.head(200).round(5).to_dict("records"),
        )
        biology.lr_interactions.to_parquet(
            folder.artifact("cluster_lr_interactions.parquet")
        )

    if cfg.data.celltype_col and biology.example_niches:
        coords_global = np.asarray(adata.obsm[cfg.data.spatial_key])[:, :2]
        # Attach pos for visualisation
        for cl, nids in list(biology.example_niches.items())[:8]:
            for nid in nids[:2]:  # at most 2 examples per cluster in the report
                d = gr.graphs[int(nid)].clone()
                d.pos = torch.from_numpy(
                    coords_global[d.global_cell_indices.cpu().numpy()].astype(np.float32)
                )
                rb.add_plot(
                    f"Example niche {nid} — cluster nc_{cl}",
                    example_niche_celltypes_figure(
                        d,
                        adata=adata,
                        celltype_col=cfg.data.celltype_col,
                        coords_global=coords_global,
                        cluster_label=f"nc_{cl}",
                        title=f"Niche {nid}, cluster nc_{cl}",
                    ),
                )

    # ----- persistence ----------------------------------------------------
    # Embeddings + UMAP
    np.savez_compressed(
        folder.artifact("niche_embeddings.npz"),
        embeddings=embeddings_arr,
        umap_2d=emb_obj.umap_2d if emb_obj.umap_2d is not None else np.empty((0, 2)),
        niche_ids=niche_ids,
        cluster_labels=cluster_labels,
    )
    if not cluster_stats.composition.empty:
        cluster_stats.composition.to_parquet(folder.artifact("niche_cluster_composition.parquet"))
    if not cluster_stats.markers.empty:
        cluster_stats.markers.to_parquet(folder.artifact("niche_cluster_markers.parquet"))
    if not cluster_stats.cluster_summary.empty:
        cluster_stats.cluster_summary.to_parquet(folder.artifact("niche_cluster_summary.parquet"))

    # Model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": cfg.unsup.model.architecture,
            "node_dim": gr.summary["n_node_features"],
            "edge_dim": gr.summary["n_edge_features"],
            "model_config": cfg.unsup.model.model_dump(),
        },
        folder.artifact(f"unsup_{cfg.unsup.model.architecture}.pt"),
    )

    # AnnData export — the deliverable: a self-contained h5ad with all results.
    if cfg.unsup.export_anndata:
        config_hash = folder.run_id.split("__")[-1]
        export_unsup_results_to_anndata(
            adata,
            UnsupExportInputs(
                niches=niches,
                niche_embeddings=embeddings_arr,
                niche_umap_2d=emb_obj.umap_2d,
                niche_cluster_labels=cluster_labels,
                run_id=folder.run_id,
                config_hash=config_hash,
                cluster_labels_str=np.array([f"nc_{c}" for c in cluster_labels]),
            ),
            out_path=folder.artifact("ecof_annotated.h5ad"),
            write_compressed=cfg.unsup.write_compressed_h5ad,
        )

    rb.write(folder.report_path)
    _log.info(f"Pipeline complete: {folder.report_path}")
    return folder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leiden_on_embeddings(
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
    resolution: float,
    random_state: int,
) -> np.ndarray:
    """Run Leiden on the niche-embedding matrix.

    Uses scanpy's pipeline (kNN graph + Leiden) since that's already a project
    dependency and well-tuned. We wrap it in a temporary AnnData.
    """
    import anndata as ad

    sub = ad.AnnData(X=embeddings.astype(np.float32))
    sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep="X", random_state=random_state)
    sc.tl.leiden(
        sub,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    return sub.obs["leiden"].astype(int).to_numpy()
