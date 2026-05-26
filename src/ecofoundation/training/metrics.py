"""Per-target metrics (categorical and numeric)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def categorical_metrics(
    logits: np.ndarray, y_true: np.ndarray, n_classes: int
) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro F1, AUROC (when feasible), kappa."""
    preds = logits.argmax(axis=-1)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, preds)),
    }
    # AUROC: needs probabilities; skip if a class is missing from y_true
    try:
        if n_classes == 2:
            # softmax over the last dim then take prob of class 1
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            out["auroc"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
    except (ValueError, IndexError):
        out["auroc"] = float("nan")

    cm = confusion_matrix(y_true, preds, labels=list(range(n_classes)))
    out["confusion_matrix"] = cm.tolist()
    return out


def numeric_metrics(pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    """MAE, RMSE, R2, Pearson r."""
    pred = pred.ravel()
    out: dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)) if len(y_true) >= 2 else float("nan"),
    }
    if len(y_true) >= 2:
        out["pearson_r"] = float(np.corrcoef(pred, y_true)[0, 1])
    else:
        out["pearson_r"] = float("nan")
    return out


@dataclass
class EpochMetrics:
    """Container for one epoch's per-target metrics."""

    epoch: int
    split: str  # "train" | "val" | "test"
    per_target: dict[str, dict[str, Any]]  # target_name -> {metric: value}
    loss_total: float
    loss_per_target: dict[str, float]

    def to_row(self) -> dict[str, Any]:
        flat: dict[str, Any] = {
            "epoch": self.epoch,
            "split": self.split,
            "loss_total": self.loss_total,
        }
        for name, losses in self.loss_per_target.items():
            flat[f"loss_{name}"] = losses
        for tname, mvals in self.per_target.items():
            for mname, mval in mvals.items():
                if mname == "confusion_matrix":
                    continue  # too large for flat row
                flat[f"{tname}_{mname}"] = mval
        return flat
