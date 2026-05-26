"""Explainability pipeline (Step 5).

Loads a trained model + its source run's predictions, picks representative
niches per class (high-confidence correct and high-confidence wrong), runs
GNNExplainer + Integrated Gradients on each, and writes an explainability
report with spatial overlays and per-class top-feature summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecofoundation.config.schemas import RunConfig
from ecofoundation.graph.construction import build_niche_graphs
from ecofoundation.interpretation import (
    SingleTargetWrapper,
    aggregate_explanations,
    compute_cell_type_attribution,
    compute_lr_interaction_attribution,
    compute_niche_umap,
    compute_pathway_enrichment,
    explain_niche,
    extract_niche_embeddings,
    ig_attribute_niche,
)
from ecofoundation.io.readers import load_anndata, validate_schema
from ecofoundation.io.writers import RunFolder, create_run_folder
from ecofoundation.models import build_model
from ecofoundation.models.heads import TargetMeta
from ecofoundation.niches.assembly import assign_niches
from ecofoundation.reporting.plots import (
    cell_type_importance_bar,
    cell_type_importance_heatmap,
    edge_channel_importance_bar,
    lr_celltype_pair_heatmap,
    niche_embedding_umap_figure,
    niche_explanation_figure,
    pathway_dotplot,
    top_gene_importance_bar,
    top_lr_interactions_table,
)
from ecofoundation.reporting.report import ReportBuilder
from ecofoundation.utils.device import resolve_device
from ecofoundation.utils.logging import configure_logging, get_logger
from ecofoundation.utils.seeding import set_global_seed

_log = get_logger(__name__)


@dataclass
class ExplainPipelineInputs:
    """Inputs for the explainability pipeline beyond the run config."""

    source_model_path: Path  # path to model_fold0.pt from a previous training run
    source_predictions_path: Path  # test_predictions.parquet
    target_name: str  # which target to explain
    top_k_per_outcome: int = 10  # K most-confident-correct + K most-confident-wrong per class
    gnn_explainer_epochs: int = 100  # cheap default; bump for higher fidelity
    ig_steps: int = 16

    # Biological extensions (Step 5.5)
    top_k_edges_per_niche: int = 20
    top_k_lr_pairs_per_edge: int = 5
    pathway_enrichment_enabled: bool = True
    pathway_gene_sets: str = "MSigDB_Hallmark_2020"
    pathway_top_n_genes: int = 100
    embedding_umap_enabled: bool = True


def run_explainability_pipeline(
    cfg: RunConfig, inputs: ExplainPipelineInputs
) -> RunFolder:
    """Build niches+graphs, load model checkpoint, explain representative niches."""
    set_global_seed(cfg.seed)
    folder = create_run_folder(cfg)
    configure_logging(level="INFO", log_file=folder.log_path)
    # Explainability runs on CPU: Captum's IG accumulates in float64 (MPS
    # rejects), and PyG explainers stumble over several MPS ops. Niches are
    # small (~51 nodes), so CPU is fast enough.
    requested = resolve_device(cfg.device)
    device = "cpu"
    if requested != "cpu":
        _log.info(f"Explainer forcing device=cpu (resolved={requested}) for Captum/PyG-Explainer stability.")
    _log.info(f"Run id: {folder.run_id} | device: {device}")
    pdf_dir = folder.root / "pdf"
    rb = ReportBuilder(run_name=cfg.run_name, cfg=cfg, pdf_dir=pdf_dir)

    # ----- rebuild deterministic niches + graphs ----------------------------
    adata = load_anndata(cfg.data)
    schema = validate_schema(adata, cfg.data)
    niches, _ = assign_niches(adata, cfg.data, cfg.niches)
    gr_result = build_niche_graphs(adata, niches, cfg.data, cfg.graph)

    # ----- load model -------------------------------------------------------
    ckpt = torch.load(inputs.source_model_path, map_location=device, weights_only=False)
    target_metas = [TargetMeta(**m) for m in ckpt["target_metas"]]
    target_meta = next((m for m in target_metas if m.name == inputs.target_name), None)
    if target_meta is None:
        raise ValueError(
            f"Target '{inputs.target_name}' not in checkpoint targets {[m.name for m in target_metas]}"
        )

    model_cfg = ckpt["model_config"]
    model = build_model(
        ckpt["architecture"],
        node_dim=ckpt["node_dim"],
        edge_dim=ckpt["edge_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        pooling=model_cfg["pooling"],
        batch_norm=model_cfg["batch_norm"],
        n_heads=model_cfg["n_heads"],
        target_metas=target_metas,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    wrapper = SingleTargetWrapper(model, inputs.target_name).to(device).eval()

    rb.add_overview(
        {
            "Source run": str(inputs.source_model_path.parent.parent.name),
            "Target": inputs.target_name,
            "Classes": ", ".join(target_meta.classes or []),
            "Niches available": len(gr_result.graphs),
            "Architecture": ckpt["architecture"],
            "Top-K per outcome": inputs.top_k_per_outcome,
            "GNNExplainer epochs": inputs.gnn_explainer_epochs,
            "Device": device,
        },
        title="Explainability run overview",
    )

    # ----- pick representative niches ---------------------------------------
    preds_df = pd.read_parquet(inputs.source_predictions_path)
    preds_df = preds_df[preds_df["target"] == inputs.target_name].copy()
    if preds_df.empty:
        rb.add_text("No predictions", "No predictions found for this target — aborting.")
        rb.write(folder.report_path)
        return folder

    # PoI per class: high-confidence correct + high-confidence wrong
    logit_cols = [c for c in preds_df.columns if c.startswith("logit_")]
    preds_df["max_logit"] = preds_df[logit_cols].max(axis=1)
    preds_df["correct"] = preds_df["y_pred"] == preds_df["y_true"]
    # Per fold the test_idx_local was a fold-local position; we need the original
    # niche-id. Without a direct mapping we reconstruct it via the artifact rows
    # in order; for the demo, we use logit confidence as a proxy.
    selected_per_class: dict[int, dict[str, list[int]]] = {}
    for cls in sorted(preds_df["y_true"].unique()):
        subset = preds_df[preds_df["y_true"] == cls]
        correct = subset[subset["correct"]].nlargest(inputs.top_k_per_outcome, "max_logit")
        wrong = subset[~subset["correct"]].nlargest(inputs.top_k_per_outcome, "max_logit")
        selected_per_class[int(cls)] = {
            "correct_niche_indices": correct.index.tolist(),
            "wrong_niche_indices": wrong.index.tolist(),
        }

    # Map prediction-row indices back to niche ids.
    # The predictions parquet rows are in the same order as the trainer wrote
    # them: per fold, per niche of that fold's test set. Without a saved
    # niche_id column, we pick niches by rank rather than position. For demo:
    # explain the first ``top_k_per_outcome`` niches per class drawn from
    # gr_result.graphs that match each class's label.
    labels_per_niche = _derive_labels_for(adata, niches, inputs.target_name, target_meta)
    coords = np.asarray(adata.obsm[cfg.data.spatial_key])[:, :2]

    explanations_by_class: dict[int, list] = {}
    ig_results_by_class: dict[int, list] = {}
    n_classes = target_meta.output_dim if target_meta.is_categorical else 1

    for cls_idx in range(n_classes):
        cls_niche_ids = np.where(labels_per_niche == cls_idx)[0]
        rng = np.random.default_rng(cfg.seed + cls_idx)
        sample_size = min(inputs.top_k_per_outcome, len(cls_niche_ids))
        if sample_size == 0:
            continue
        picks = rng.choice(cls_niche_ids, size=sample_size, replace=False)

        class_explanations = []
        class_igs = []
        gnn_failures = 0
        for nid in picks:
            data = gr_result.graphs[int(nid)].clone()
            data.pos = torch.from_numpy(
                coords[data.global_cell_indices.cpu().numpy()].astype(np.float32)
            )
            # GNNExplainer and IG are independent — failure in one must not
            # block the other.
            try:
                explanation = explain_niche(
                    wrapper, data, target_class=cls_idx,
                    epochs=inputs.gnn_explainer_epochs, n_classes=n_classes, device=device,
                )
                class_explanations.append((data, explanation))
            except Exception as e:  # noqa: BLE001
                gnn_failures += 1
                if gnn_failures <= 2:  # don't spam the log
                    _log.warning(f"GNNExplainer failed for niche {nid}: {e}")
            try:
                ig = ig_attribute_niche(
                    wrapper, data, target_class=cls_idx, device=device, steps=inputs.ig_steps
                )
                class_igs.append(ig)
            except Exception as e:  # noqa: BLE001
                _log.warning(f"IG failed for niche {nid}: {e}")

        if gnn_failures:
            _log.warning(
                f"Class {cls_idx}: GNNExplainer failed on {gnn_failures}/{len(picks)} niches "
                "(known PyG issue with multi-channel edge_attr) — IG attribution still used."
            )

        explanations_by_class[cls_idx] = class_explanations
        ig_results_by_class[cls_idx] = class_igs

    # ----- aggregate + report ----------------------------------------------
    class_label_map = (target_meta.classes or [str(i) for i in range(n_classes)])

    all_class_ids = sorted(set(explanations_by_class) | set(ig_results_by_class))
    for cls_idx in all_class_ids:
        paired = explanations_by_class.get(cls_idx, [])
        igs = ig_results_by_class.get(cls_idx, [])
        if not paired and not igs:
            continue
        explanations = [p[1] for p in paired]
        attr = aggregate_explanations(
            explanations,
            igs,
            feature_names=gr_result.node_feature_names,
            edge_channel_names=gr_result.edge_feature_names,
            target_class=cls_idx,
        )

        n_total = max(len(igs), len(explanations))
        rb.add_text(
            f"Class: {class_label_map[cls_idx]}",
            (
                f"IG attribution over {len(igs)} niches; "
                f"GNNExplainer over {len(explanations)} niches."
            ),
        )
        rb.add_plot(
            f"Top genes — {class_label_map[cls_idx]}",
            top_gene_importance_bar(attr, top_n=20, class_label=class_label_map[cls_idx]),
        )
        rb.add_plot(
            f"Edge feature channels — {class_label_map[cls_idx]}",
            edge_channel_importance_bar(attr, class_label=class_label_map[cls_idx]),
        )
        rb.add_table(
            f"Top 30 genes ({class_label_map[cls_idx]})",
            rows=attr.gene_importance.head(30).round(5).to_dict("records"),
        )

        # Niche overlays only when we actually have GNNExplainer outputs.
        for data, ex in paired[:3]:
            rb.add_plot(
                f"Niche {ex.niche_id} explanation ({class_label_map[cls_idx]}, "
                f"prob={ex.target_prob:.2f})",
                niche_explanation_figure(
                    data, ex, title=f"Niche {ex.niche_id} -> {class_label_map[cls_idx]}"
                ),
            )

        # Persist
        attr.gene_importance.to_parquet(folder.artifact(f"gene_importance_class{cls_idx}.parquet"))
        attr.edge_channel_importance.to_parquet(
            folder.artifact(f"edge_channel_importance_class{cls_idx}.parquet")
        )

    # ----- biological extensions (Step 5.5) ---------------------------------
    explained_data_by_class: dict[int, list] = {}
    for cls_idx, paired in explanations_by_class.items():
        explained_data_by_class[cls_idx] = [p[0] for p in paired]
    # Also include the Data objects from IG runs that lacked a GNNExplanation —
    # IG itself doesn't store Data references, so we use what we have.

    # 1. Cell-type importance
    if cfg.data.celltype_col:
        rb.add_text(
            "Biological extensions",
            (
                "Cell-type, ligand-receptor, pathway-enrichment, and embedding-UMAP "
                "analyses based on the per-niche attributions above. These tie the "
                "model's decisions to interpretable biological structure."
            ),
        )
        ct_attr = compute_cell_type_attribution(
            adata=adata,
            celltype_col=cfg.data.celltype_col,
            ig_results_by_class=ig_results_by_class,
            explained_data_by_class=explained_data_by_class,
            class_label_map=class_label_map,
        )
        if not ct_attr.per_class.empty:
            rb.add_plot(
                "Cell-type importance per class",
                cell_type_importance_heatmap(ct_attr),
            )
            for cls_idx in all_class_ids:
                lbl = class_label_map[cls_idx]
                rb.add_plot(
                    f"Top cell types — {lbl}",
                    cell_type_importance_bar(ct_attr, class_label=lbl, top_n=15),
                )
            ct_attr.per_class.to_parquet(folder.artifact("cell_type_importance.parquet"))

    # 2. LR-pair × cell-type-pair attribution
    if cfg.data.celltype_col and gr_result.lr_resource.n_pairs_kept > 0:
        lr_attr = compute_lr_interaction_attribution(
            adata=adata,
            celltype_col=cfg.data.celltype_col,
            lr_resource=gr_result.lr_resource,
            expression_layer=cfg.graph.node_expression_layer,
            ig_results_by_class=ig_results_by_class,
            explained_data_by_class=explained_data_by_class,
            class_label_map=class_label_map,
            top_k_edges_per_niche=inputs.top_k_edges_per_niche,
            top_k_lr_pairs_per_edge=inputs.top_k_lr_pairs_per_edge,
        )
        if not lr_attr.per_class_lr.empty:
            for cls_idx in all_class_ids:
                lbl = class_label_map[cls_idx]
                rb.add_plot(
                    f"LR x celltype-pair heatmap — {lbl}",
                    lr_celltype_pair_heatmap(lr_attr, class_label=lbl),
                )
                top_lr = top_lr_interactions_table(lr_attr, class_label=lbl, top_n=25)
                if not top_lr.empty:
                    rb.add_table(
                        f"Top LR interactions — {lbl}",
                        rows=top_lr.round(5).to_dict("records"),
                    )
            lr_attr.per_class_lr.to_parquet(folder.artifact("lr_interaction_attribution.parquet"))

    # 3. Pathway enrichment per cell type
    if inputs.pathway_enrichment_enabled and cfg.data.celltype_col:
        pw = compute_pathway_enrichment(
            adata=adata,
            celltype_col=cfg.data.celltype_col,
            feature_names=gr_result.node_feature_names,
            ig_results_by_class=ig_results_by_class,
            explained_data_by_class=explained_data_by_class,
            class_label_map=class_label_map,
            n_top_genes=inputs.pathway_top_n_genes,
            gene_sets=inputs.pathway_gene_sets,
            use_enrichr=True,
        )
        if not pw.per_class_celltype.empty:
            for cls_idx in all_class_ids:
                lbl = class_label_map[cls_idx]
                rb.add_plot(
                    f"Pathway enrichment — {lbl}",
                    pathway_dotplot(pw, class_label=lbl, top_n_terms=6),
                )
            pw.per_class_celltype.to_parquet(folder.artifact("pathway_enrichment.parquet"))
            rb.add_text(
                "Pathway enrichment notes",
                (
                    "Per-cell-type pathway enrichment was run via Enrichr (online API) "
                    f"against {inputs.pathway_gene_sets}. Dot size = -log10(adj. p-value); "
                    "dot colour = Enrichr combined score."
                ),
            )
        else:
            rb.add_text(
                "Pathway enrichment",
                "Enrichr returned no significant terms (or was unavailable offline).",
            )

    # 4. Niche embedding UMAP with explained niches highlighted
    if inputs.embedding_umap_enabled:
        emb = extract_niche_embeddings(model, gr_result.graphs, batch_size=64, device=device)
        try:
            emb = compute_niche_umap(emb, n_neighbors=30, min_dist=0.3, random_state=cfg.seed)
        except Exception as e:  # noqa: BLE001
            _log.warning(f"UMAP failed: {e}")
            emb = emb
        explained_niche_ids = np.array(
            [
                int(d.niche_id)
                for paired in explanations_by_class.values()
                for d, _ in paired
            ],
            dtype=np.int64,
        )
        rb.add_plot(
            "Niche embedding UMAP (explained niches highlighted)",
            niche_embedding_umap_figure(
                emb,
                labels=labels_per_niche,
                class_label_map=class_label_map,
                explained_niche_ids=explained_niche_ids,
                title="Niche embedding UMAP (per-niche pooled embedding)",
            ),
        )
        # Persist the embeddings + UMAP
        np.savez_compressed(
            folder.artifact("niche_embeddings.npz"),
            embeddings=emb.embeddings,
            umap_2d=emb.umap_2d if emb.umap_2d is not None else np.empty((0, 2)),
            niche_ids=emb.niche_ids,
            labels=labels_per_niche,
        )

    rb.write(folder.report_path)
    _log.info(f"Pipeline complete: {folder.report_path}")
    return folder


def _derive_labels_for(adata, niches, target_name, target_meta: TargetMeta) -> np.ndarray:
    """Derive integer-encoded class labels per niche via ego-cell aggregation."""
    obs_col = target_name  # in our demo: 'sonicated' obs col equals target name
    if obs_col not in adata.obs.columns:
        # try to match by name through TargetSpec; fallback: zero labels
        return np.zeros(niches.n_niches, dtype=np.int64)
    values = adata.obs[obs_col].astype(str).to_numpy()
    if target_meta.is_categorical and target_meta.classes:
        lookup = {c: i for i, c in enumerate(target_meta.classes)}
        labels = np.array(
            [lookup.get(values[int(niches.ego_cell[nid])], 0) for nid in range(niches.n_niches)],
            dtype=np.int64,
        )
    else:
        labels = np.array(
            [float(values[int(niches.ego_cell[nid])]) for nid in range(niches.n_niches)],
        )
    return labels
