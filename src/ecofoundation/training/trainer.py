"""Training loop for the supervised graph classifier.

One ``train_one_fold`` call per CV fold:

  - assemble train/val/test ``DataLoader``s from PyG graphs
  - train ``cfg.epochs`` epochs with early stopping on val loss
  - return best-model checkpoints, per-epoch metric history,
    and final test predictions
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ecofoundation.config.schemas import TrainingConfig
from ecofoundation.models.adversarial import AdversarialBatchHead, lambda_schedule
from ecofoundation.models.heads import TargetMeta
from ecofoundation.training.metrics import (
    EpochMetrics,
    categorical_metrics,
    numeric_metrics,
)
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class FoldResult:
    """Output of one CV fold."""

    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    val_indices: np.ndarray
    history: list[EpochMetrics] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_state_dict: dict[str, Any] | None = None
    test_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    test_predictions: dict[str, np.ndarray] = field(default_factory=dict)
    test_targets: dict[str, np.ndarray] = field(default_factory=dict)
    n_epochs_run: int = 0


def train_one_fold(
    *,
    fold: int,
    graphs: list[Data],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: dict[str, np.ndarray],
    target_metas: list[TargetMeta],
    model_builder,
    cfg: TrainingConfig,
    device: str,
    val_fraction: float = 0.15,
    seed: int = 0,
    adv_batch_labels: np.ndarray | None = None,
    adv_n_batches: int | None = None,
) -> FoldResult:
    """Train and evaluate one CV fold.

    When ``cfg.adversarial.enabled`` and ``adv_batch_labels`` is provided,
    a Domain-Adversarial branch (Gradient Reversal + small MLP) is added
    on top of the encoder's pooled embedding. The branch tries to predict
    ``adv_batch_labels[niche_id]``; via GRL the encoder is pushed AWAY from
    features that distinguish those labels.
    """
    rng = np.random.default_rng(seed + fold)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(round(len(perm) * val_fraction)))
    val_idx = perm[:n_val]
    actual_train_idx = perm[n_val:]

    train_graphs = [
        _attach_labels(graphs[i], i, labels, target_metas, adv_batch_labels)
        for i in actual_train_idx
    ]
    val_graphs = [
        _attach_labels(graphs[i], i, labels, target_metas, adv_batch_labels)
        for i in val_idx
    ]
    test_graphs = [
        _attach_labels(graphs[i], i, labels, target_metas, adv_batch_labels)
        for i in test_idx
    ]

    train_loader = DataLoader(
        train_graphs, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_graphs, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_graphs, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = model_builder().to(device)
    adv_head: AdversarialBatchHead | None = None
    if cfg.adversarial.enabled and adv_n_batches is not None and adv_n_batches >= 2:
        adv_head = AdversarialBatchHead(
            in_dim=model.hidden_dim,
            n_batches=adv_n_batches,
            hidden_dim=cfg.adversarial.hidden_dim,
            dropout=cfg.adversarial.dropout,
        ).to(device)
        _log.info(
            f"[fold {fold}] adversarial debiasing ENABLED "
            f"(n_batches={adv_n_batches}, lambda_max={cfg.adversarial.lambda_max})"
        )
    opt = _make_optimizer_with_adv(model, adv_head, cfg)

    fold_result = FoldResult(
        fold=fold,
        train_indices=actual_train_idx,
        test_indices=test_idx,
        val_indices=val_idx,
    )

    patience = 0
    for epoch in range(1, cfg.epochs + 1):
        lam = lambda_schedule(
            epoch - 1,
            lambda_max=cfg.adversarial.lambda_max,
            warmup_epochs=cfg.adversarial.warmup_epochs,
        ) if adv_head is not None else 0.0
        train_m = _run_epoch(
            model, train_loader, target_metas, opt, cfg, device,
            train=True, adv_head=adv_head, adv_lambda=lam,
        )
        train_m.epoch = epoch
        train_m.split = "train"

        with torch.no_grad():
            val_m = _run_epoch(
                model, val_loader, target_metas, None, cfg, device,
                train=False, adv_head=adv_head, adv_lambda=0.0,
            )
            val_m.epoch = epoch
            val_m.split = "val"

        fold_result.history.extend([train_m, val_m])
        fold_result.n_epochs_run = epoch

        _log.info(
            f"[fold {fold}] epoch {epoch:3d} | "
            f"train_loss={train_m.loss_total:.4f}  val_loss={val_m.loss_total:.4f}"
        )

        if val_m.loss_total < fold_result.best_val_loss - 1e-6:
            fold_result.best_val_loss = val_m.loss_total
            fold_result.best_state_dict = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                _log.info(f"[fold {fold}] early stop at epoch {epoch}")
                break

    # ----- restore best & test ---------------------------------------------
    if fold_result.best_state_dict is not None:
        model.load_state_dict(fold_result.best_state_dict)
    model.eval()

    test_preds: dict[str, list[np.ndarray]] = {m.name: [] for m in target_metas}
    test_targets: dict[str, list[np.ndarray]] = {m.name: [] for m in target_metas}
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            for m in target_metas:
                test_preds[m.name].append(out[m.name].detach().cpu().numpy())
                test_targets[m.name].append(getattr(batch, _target_attr(m.name)).cpu().numpy())

    test_metrics: dict[str, dict[str, Any]] = {}
    for m in target_metas:
        p = np.concatenate(test_preds[m.name], axis=0)
        y = np.concatenate(test_targets[m.name], axis=0)
        fold_result.test_predictions[m.name] = p
        fold_result.test_targets[m.name] = y
        if m.is_categorical:
            test_metrics[m.name] = categorical_metrics(p, y, m.output_dim)
        else:
            test_metrics[m.name] = numeric_metrics(p, y)
    fold_result.test_metrics = test_metrics
    _log.info(
        f"[fold {fold}] DONE — test metrics: "
        + ", ".join(
            f"{name}=(acc={mv.get('accuracy', mv.get('r2', '?'))!s:.5s}, "
            f"auroc={mv.get('auroc', float('nan')):.3f})"
            if mv.get("accuracy") is not None
            else f"{name}=(r2={mv.get('r2', float('nan')):.3f})"
            for name, mv in test_metrics.items()
        )
    )
    return fold_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _target_attr(name: str) -> str:
    return f"y_{name}"


ADV_BATCH_ATTR = "y_adv_batch"


def _attach_labels(
    data: Data,
    idx: int,
    labels: dict[str, np.ndarray],
    target_metas: list[TargetMeta],
    adv_batch_labels: np.ndarray | None = None,
) -> Data:
    """Attach per-target labels (+ optional adversarial label) to a Data object."""
    new = data.clone()
    for m in target_metas:
        val = labels[m.name][idx]
        attr = _target_attr(m.name)
        if m.is_categorical:
            new[attr] = torch.tensor([int(val)], dtype=torch.long)
        else:
            new[attr] = torch.tensor([float(val)], dtype=torch.float32)
    if adv_batch_labels is not None:
        new[ADV_BATCH_ATTR] = torch.tensor([int(adv_batch_labels[idx])], dtype=torch.long)
    return new


def _make_optimizer_with_adv(
    model: nn.Module, adv_head: nn.Module | None, cfg: TrainingConfig
):
    params = list(model.parameters())
    if adv_head is not None:
        params = params + list(adv_head.parameters())
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    target_metas: list[TargetMeta],
    opt: torch.optim.Optimizer | None,
    cfg: TrainingConfig,
    device: str,
    *,
    train: bool,
    adv_head: AdversarialBatchHead | None = None,
    adv_lambda: float = 0.0,
) -> EpochMetrics:
    import torch.nn.functional as F

    model.train(train)
    if adv_head is not None:
        adv_head.train(train)
    total_loss = 0.0
    total_adv_loss = 0.0
    per_target_loss = {m.name: 0.0 for m in target_metas}
    n_batches = 0

    all_preds: dict[str, list[np.ndarray]] = {m.name: [] for m in target_metas}
    all_targets: dict[str, list[np.ndarray]] = {m.name: [] for m in target_metas}

    for batch in loader:
        batch = batch.to(device)
        out, z = model(batch, return_embedding=True)
        targets = {m.name: getattr(batch, _target_attr(m.name)) for m in target_metas}
        task_loss, per_loss = model.head.compute_losses(out, targets)

        loss = task_loss
        if adv_head is not None and hasattr(batch, ADV_BATCH_ATTR) and adv_lambda > 0:
            adv_target = getattr(batch, ADV_BATCH_ATTR)
            adv_logits = adv_head(z, adv_lambda)
            adv_loss = F.cross_entropy(adv_logits, adv_target)
            loss = loss + adv_loss
            total_adv_loss += float(adv_loss.detach().cpu())

        if train and opt is not None:
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                params = list(model.parameters())
                if adv_head is not None:
                    params = params + list(adv_head.parameters())
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
        total_loss += float(loss.detach().cpu())
        for k, v in per_loss.items():
            per_target_loss[k] += float(v.detach().cpu())
        n_batches += 1
        for m in target_metas:
            all_preds[m.name].append(out[m.name].detach().cpu().numpy())
            all_targets[m.name].append(targets[m.name].detach().cpu().numpy())

    avg = total_loss / max(n_batches, 1)
    avg_per = {k: v / max(n_batches, 1) for k, v in per_target_loss.items()}

    # Per-target metrics (lightweight — confusion matrix only at fold end)
    per_target_metrics: dict[str, dict[str, Any]] = {}
    for m in target_metas:
        p = np.concatenate(all_preds[m.name], axis=0)
        y = np.concatenate(all_targets[m.name], axis=0)
        if m.is_categorical:
            metric = categorical_metrics(p, y, m.output_dim)
            metric.pop("confusion_matrix", None)
        else:
            metric = numeric_metrics(p, y)
        per_target_metrics[m.name] = metric

    return EpochMetrics(
        epoch=0,
        split="?",
        per_target=per_target_metrics,
        loss_total=avg,
        loss_per_target=avg_per,
    )
