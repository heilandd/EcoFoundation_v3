"""Training-related matplotlib plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from ecofoundation.reporting.plots.umap import _build_palette
from ecofoundation.reporting.style import new_figure, style_axes
from ecofoundation.training.metrics import EpochMetrics


def loss_curve_figure(history: list[EpochMetrics], *, title: str = "Training curves") -> Figure:
    """Train vs val loss per epoch."""
    df = pd.DataFrame([h.to_row() for h in history])
    if df.empty:
        fig, ax = new_figure(width=4, height=2)
        ax.set_title(title + " (empty)")
        style_axes(ax)
        return fig
    fig, ax = new_figure(width=4.6, height=2.6)
    for split, color in (("train", "#2563eb"), ("val", "#dc2626")):
        sub = df[df["split"] == split]
        if not sub.empty:
            ax.plot(sub["epoch"], sub["loss_total"], color=color, label=split, linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    return fig


def metric_curve_figure(
    history: list[EpochMetrics], target: str, metric: str, *, title: str | None = None
) -> Figure:
    """Per-target metric over epochs (e.g. AUROC, accuracy, R²)."""
    df = pd.DataFrame([h.to_row() for h in history])
    col = f"{target}_{metric}"
    fig, ax = new_figure(width=4.6, height=2.6)
    if col not in df.columns:
        ax.set_title((title or f"{target} {metric}") + " (missing)")
        style_axes(ax)
        return fig
    for split, color in (("train", "#2563eb"), ("val", "#dc2626")):
        sub = df[df["split"] == split]
        if not sub.empty:
            ax.plot(sub["epoch"], sub[col], color=color, label=split, linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{target} — {metric}")
    ax.legend(loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    return fig


def confusion_matrix_figure(
    cm: np.ndarray, classes: list[str], *, title: str = "Confusion matrix", normalize: bool = True
) -> Figure:
    """Confusion matrix heatmap."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums))
        z = cm_norm
        fmt = ".2f"
    else:
        z = cm
        fmt = "d"
    fig, ax = new_figure(width=max(2.6, 0.4 * len(classes) + 1.4), height=max(2.4, 0.4 * len(classes) + 1.0))
    im = ax.imshow(z, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = z[i, j]
            ax.text(
                j, i,
                f"{val:.2f}" if normalize else f"{int(val)}",
                ha="center", va="center",
                color="white" if val > (z.max() * 0.55) else "black",
                fontsize=5,
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=5)
    style_axes(ax)
    fig.tight_layout()
    return fig


def roc_curves_figure(
    y_true: np.ndarray,
    logits: np.ndarray,
    classes: list[str],
    *,
    title: str = "ROC curves",
) -> Figure:
    """One-vs-rest ROC curves per class."""
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    fig, ax = new_figure(width=3.8, height=3.4)
    cmap = _build_palette(len(classes))
    for i, cls in enumerate(classes):
        if (y_true == i).sum() == 0:
            continue
        binary = (y_true == i).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        ax.plot(fpr, tpr, color=cmap[i], label=str(cls), linewidth=0.8)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=5, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style_axes(ax)
    fig.tight_layout()
    return fig


def per_fold_metric_bar(
    fold_metrics: list[dict[str, float]], metric: str, *, title: str | None = None
) -> Figure:
    """Bar chart of one metric across folds (with mean line)."""
    vals = [m[metric] for m in fold_metrics if metric in m and not np.isnan(m[metric])]
    fig, ax = new_figure(width=3.6, height=2.4)
    if not vals:
        ax.set_title((title or metric) + " (no data)")
        style_axes(ax)
        return fig
    x = np.arange(len(vals))
    ax.bar(x, vals, color="#a5b4fc", edgecolor="black", linewidth=0.3)
    ax.axhline(float(np.mean(vals)), color="#dc2626", linestyle="--", linewidth=0.6, label=f"mean={np.mean(vals):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i}" for i in range(len(vals))], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} per fold")
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    return fig
