"""Supervised GNN training pipeline (Step 4).

End-to-end orchestrator:

  load -> niches -> graphs -> labels -> CV splits -> train per fold ->
  aggregate metrics -> report (training curves, confusion matrices, ROC).
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch

from ecofoundation.config.schemas import RunConfig, TargetSpec, TrainingConfig
from ecofoundation.graph.construction import build_niche_graphs
from ecofoundation.io.readers import load_anndata, validate_schema
from ecofoundation.io.writers import RunFolder, create_run_folder
from ecofoundation.models import build_model, resolve_target_metas
from ecofoundation.models.heads import TargetMeta
from ecofoundation.niches.assembly import assign_niches
from ecofoundation.niches.base import NicheAssignment
from ecofoundation.reporting.plots import (
    confusion_matrix_figure,
    loss_curve_figure,
    metric_curve_figure,
    niche_size_distribution,
    niches_per_group_bar,
    per_fold_metric_bar,
    roc_curves_figure,
)
from ecofoundation.reporting.report import ReportBuilder
from ecofoundation.training.cv import build_splits
from ecofoundation.training.trainer import FoldResult, train_one_fold
from ecofoundation.utils.device import resolve_device
from ecofoundation.utils.logging import configure_logging, get_logger
from ecofoundation.utils.seeding import set_global_seed

_log = get_logger(__name__)


def run_training_pipeline(cfg: RunConfig) -> RunFolder:
    if cfg.training is None:
        raise ValueError("RunConfig.training is required for the training pipeline.")

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
            "Architecture": cfg.model.architecture,
            "Hidden dim": cfg.model.hidden_dim,
            "Layers": cfg.model.n_layers,
            "Pooling": cfg.model.pooling,
            "Targets": ", ".join(t.name for t in cfg.training.targets),
            "Device": device,
        },
        title="Training run overview",
    )

    # ----- niches + graphs ---------------------------------------------------
    niches, _overlap = assign_niches(adata, cfg.data, cfg.niches)
    if niches.n_niches == 0:
        rb.add_text("Niches", "No niches built — aborting.")
        rb.write(folder.report_path)
        return folder

    rb.add_plot("Niche size distribution", niche_size_distribution(niches))
    rb.add_plot("Niches per patient", niches_per_group_bar(niches))

    gr_result = build_niche_graphs(adata, niches, cfg.data, cfg.graph)
    rb.add_text(
        "Graphs",
        (
            f"Built {len(gr_result.graphs)} PyG graphs. "
            f"Node features = {gr_result.summary['n_node_features']}-dim ({cfg.graph.node_feature_source}); "
            f"edge features = {gr_result.summary['n_edge_features']}-dim "
            f"({', '.join(gr_result.edge_feature_names)})."
        ),
    )

    # ----- derive per-niche labels -------------------------------------------
    labels = _derive_niche_labels(adata, niches, cfg.training.targets, cfg.data.spatial_key)
    label_lookup = {name: arr.tolist() for name, arr in labels.items()}
    target_metas = resolve_target_metas(cfg.training.targets, label_lookup)

    # Encode labels into integer / float arrays for the trainer
    encoded_labels = {
        m.name: m.encode(labels[m.name]).numpy() for m in target_metas
    }
    primary_label = None
    primary_classes: list[str] | None = None
    for m in target_metas:
        if m.is_categorical:
            primary_label = encoded_labels[m.name]
            primary_classes = m.classes
            break

    rb.add_table(
        "Target summary",
        rows=[
            {
                "name": m.name,
                "type": m.type,
                "output_dim": m.output_dim,
                "classes": ", ".join(m.classes) if m.classes else "—",
                "loss": m.loss,
                "weight": m.weight,
            }
            for m in target_metas
        ],
    )

    # Label balance for primary target
    if primary_label is not None and primary_classes is not None:
        unique, counts = np.unique(primary_label, return_counts=True)
        rb.add_table(
            "Primary target class balance",
            rows=[
                {"class": primary_classes[i], "count": int(c)}
                for i, c in zip(unique.tolist(), counts.tolist(), strict=True)
            ],
        )

    # ----- adversarial debiasing labels (per-niche batch / patient ids) -----
    adv_batch_labels: np.ndarray | None = None
    adv_n_batches: int | None = None
    if cfg.training.adversarial.enabled:
        adv_col = cfg.training.adversarial.batch_col or cfg.data.patient_id_col
        if adv_col and adv_col in adata.obs.columns:
            adv_values = adata.obs[adv_col].astype(str).to_numpy()
            niche_adv = np.array(
                [adv_values[int(niches.ego_cell[nid])] for nid in range(niches.n_niches)]
            )
            uniq_adv = sorted(set(niche_adv.tolist()))
            adv_lookup = {v: i for i, v in enumerate(uniq_adv)}
            adv_batch_labels = np.array([adv_lookup[v] for v in niche_adv], dtype=np.int64)
            adv_n_batches = len(uniq_adv)
            _log.info(
                f"Adversarial debiasing: col='{adv_col}' "
                f"({adv_n_batches} unique batches) lambda_max={cfg.training.adversarial.lambda_max}"
            )
            rb.add_text(
                "Adversarial debiasing",
                (
                    f"Active — encoder is pushed AWAY from features that predict "
                    f"obs['{adv_col}'] ({adv_n_batches} unique values). "
                    f"lambda_max={cfg.training.adversarial.lambda_max}, "
                    f"warmup_epochs={cfg.training.adversarial.warmup_epochs}."
                ),
            )
        else:
            _log.warning(
                f"Adversarial enabled but column '{adv_col}' not in obs — disabling."
            )

    # ----- splits ------------------------------------------------------------
    patients = niches.group_label.astype(str)
    splits = build_splits(
        patients=patients,
        primary_label=primary_label,
        cfg=cfg.training,
        seed=cfg.seed,
    )
    rb.add_text(
        "CV strategy",
        f"{splits.rationale} → {len(splits.folds)} fold(s).",
        parameters={
            "n_patients": int(np.unique(patients).size),
            "min_patients_for_kfold": cfg.training.min_patients_for_kfold,
            "strategy": splits.strategy,
        },
    )

    # ----- train -------------------------------------------------------------
    def _model_builder():
        return build_model(
            cfg.model.architecture,
            node_dim=gr_result.summary["n_node_features"],
            edge_dim=gr_result.summary["n_edge_features"],
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
            dropout=cfg.model.dropout,
            pooling=cfg.model.pooling,
            batch_norm=cfg.model.batch_norm,
            n_heads=cfg.model.n_heads,
            target_metas=target_metas,
        )

    fold_results: list[FoldResult] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits.folds):
        _log.info(
            f"--- fold {fold_idx}: train={len(train_idx)} test={len(test_idx)} ---"
        )
        fr = train_one_fold(
            fold=fold_idx,
            graphs=gr_result.graphs,
            train_idx=np.asarray(train_idx),
            test_idx=np.asarray(test_idx),
            labels=encoded_labels,
            target_metas=target_metas,
            model_builder=_model_builder,
            cfg=cfg.training,
            device=device,
            seed=cfg.seed,
            adv_batch_labels=adv_batch_labels,
            adv_n_batches=adv_n_batches,
        )
        fold_results.append(fr)

        # Per-fold report
        rb.add_text(
            f"Fold {fold_idx} summary",
            (
                f"Epochs run: {fr.n_epochs_run} | best val loss: {fr.best_val_loss:.4f} | "
                f"train={len(fr.train_indices)} val={len(fr.val_indices)} test={len(fr.test_indices)}"
            ),
        )
        rb.add_plot(f"Fold {fold_idx} — loss curve", loss_curve_figure(fr.history))
        for m in target_metas:
            metric_name = "auroc" if m.is_categorical else "r2"
            rb.add_plot(
                f"Fold {fold_idx} — {m.name} {metric_name}",
                metric_curve_figure(fr.history, m.name, metric_name),
            )
            if m.is_categorical and "confusion_matrix" in fr.test_metrics.get(m.name, {}):
                cm = np.asarray(fr.test_metrics[m.name]["confusion_matrix"])
                rb.add_plot(
                    f"Fold {fold_idx} — {m.name} confusion",
                    confusion_matrix_figure(cm, m.classes or [str(i) for i in range(m.output_dim)]),
                )
                if m.output_dim >= 2:
                    rb.add_plot(
                        f"Fold {fold_idx} — {m.name} ROC",
                        roc_curves_figure(
                            fr.test_targets[m.name],
                            fr.test_predictions[m.name],
                            m.classes or [str(i) for i in range(m.output_dim)],
                        ),
                    )

    # ----- aggregated metrics across folds -----------------------------------
    agg_rows = []
    per_fold_for_plot: dict[str, list[dict[str, float]]] = {}
    for m in target_metas:
        per_fold_for_plot[m.name] = [fr.test_metrics[m.name] for fr in fold_results]
        row: dict[str, Any] = {"target": m.name, "type": m.type}
        if m.is_categorical:
            metrics = ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "cohen_kappa")
        else:
            metrics = ("mae", "rmse", "r2", "pearson_r")
        for metric in metrics:
            vals = [fr.test_metrics[m.name].get(metric) for fr in fold_results]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
        agg_rows.append(row)
    rb.add_table("Aggregated test metrics across folds", rows=agg_rows)

    for m in target_metas:
        metric = "auroc" if m.is_categorical else "r2"
        rb.add_plot(
            f"{m.name} — {metric} per fold",
            per_fold_metric_bar(per_fold_for_plot[m.name], metric, title=f"{m.name} {metric} per fold"),
        )

    # ----- persistence -------------------------------------------------------
    # Per-fold predictions
    pred_rows = []
    for fr in fold_results:
        for m in target_metas:
            p = fr.test_predictions[m.name]
            y = fr.test_targets[m.name]
            for i in range(p.shape[0]):
                row = {
                    "fold": fr.fold,
                    "target": m.name,
                    "test_idx_local": int(i),
                    "y_true": int(y[i]) if m.is_categorical else float(y[i]),
                }
                if m.is_categorical:
                    pred_class = int(np.argmax(p[i]))
                    row["y_pred"] = pred_class
                    for k in range(p.shape[1]):
                        row[f"logit_{k}"] = float(p[i, k])
                else:
                    row["y_pred"] = float(p[i].ravel()[0])
                pred_rows.append(row)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(folder.artifact("test_predictions.parquet"))

    # Save best model from fold 0 (or last fold)
    if fold_results and fold_results[0].best_state_dict is not None:
        torch.save(
            {
                "state_dict": fold_results[0].best_state_dict,
                "target_metas": [vars(m) for m in target_metas],
                "node_dim": gr_result.summary["n_node_features"],
                "edge_dim": gr_result.summary["n_edge_features"],
                "architecture": cfg.model.architecture,
                "model_config": cfg.model.model_dump(),
            },
            folder.artifact("model_fold0.pt"),
        )

    rb.write(folder.report_path)
    _log.info(f"Pipeline complete: {folder.report_path}")
    return folder


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------


def _derive_niche_labels(
    adata: ad.AnnData,
    niches: NicheAssignment,
    targets: list[TargetSpec],
    spatial_key: str,
) -> dict[str, np.ndarray]:
    """For each target, build a length-n_niches array of niche labels."""
    out: dict[str, np.ndarray] = {}
    for t in targets:
        if t.obs_column not in adata.obs.columns:
            raise KeyError(f"obs column '{t.obs_column}' missing for target '{t.name}'")
        col = adata.obs[t.obs_column]
        values = col.astype(str).to_numpy() if t.type == "categorical" else col.to_numpy()
        labels = np.empty(niches.n_niches, dtype=object)
        for nid in range(niches.n_niches):
            cells = niches.cells_per_niche[nid]
            if t.label_aggregation == "ego":
                labels[nid] = values[int(niches.ego_cell[nid])]
            elif t.label_aggregation == "first":
                labels[nid] = values[int(cells[0])]
            elif t.label_aggregation == "majority":
                v = values[cells]
                if t.type == "categorical":
                    uniq, counts = np.unique(v, return_counts=True)
                    labels[nid] = uniq[counts.argmax()]
                else:
                    labels[nid] = float(np.mean(v.astype(float)))
            else:
                raise ValueError(f"Unknown aggregation: {t.label_aggregation}")
        out[t.name] = labels
    return out
