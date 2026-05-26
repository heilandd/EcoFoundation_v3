"""Multi-task prediction head.

One linear head per target; categorical targets output ``len(classes)`` logits,
numeric targets output a single scalar. Combined loss = weighted sum of per-
target losses (CE / BCE / MSE / Huber / MAE).

The head returns a dict ``{target_name: tensor}`` so the training loop can
log per-target metrics without coupling to the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecofoundation.config.schemas import TargetSpec


@dataclass
class TargetMeta:
    """Resolved per-target metadata (output dim, class index).

    Built once at training-time from the target spec and the actual label set.
    """

    name: str
    type: str  # "categorical" | "numeric"
    output_dim: int
    classes: list[str] | None  # canonical class order for categorical
    loss: str
    weight: float

    @property
    def is_categorical(self) -> bool:
        return self.type == "categorical"

    def encode(self, labels) -> torch.Tensor:
        """Encode a 1-D iterable of labels into a tensor for loss computation."""
        if self.is_categorical:
            assert self.classes is not None
            lookup = {c: i for i, c in enumerate(self.classes)}
            return torch.as_tensor([lookup[str(x)] for x in labels], dtype=torch.long)
        # numeric
        return torch.as_tensor([float(x) for x in labels], dtype=torch.float32)


def resolve_target_metas(
    target_specs: list[TargetSpec],
    label_lookup: dict[str, list],
) -> list[TargetMeta]:
    """Convert TargetSpec + observed labels into TargetMeta.

    Parameters
    ----------
    target_specs
        From ``TrainingConfig.targets``.
    label_lookup
        ``{target.name: [labels per niche]}`` — the labels actually present in
        the dataset (used to infer classes when spec.classes is None).
    """
    metas: list[TargetMeta] = []
    for spec in target_specs:
        labels = label_lookup[spec.name]
        if spec.type == "categorical":
            classes = spec.classes
            if classes is None:
                classes = sorted({str(x) for x in labels})
            output_dim = max(len(classes), 2)
            metas.append(
                TargetMeta(
                    name=spec.name,
                    type="categorical",
                    output_dim=output_dim,
                    classes=classes,
                    loss=spec.loss,
                    weight=spec.weight,
                )
            )
        else:
            metas.append(
                TargetMeta(
                    name=spec.name,
                    type="numeric",
                    output_dim=1,
                    classes=None,
                    loss=spec.loss,
                    weight=spec.weight,
                )
            )
    return metas


class MultiTaskHead(nn.Module):
    """One Linear per target. Returns ``{target_name: logits/value}``."""

    def __init__(self, in_dim: int, metas: list[TargetMeta]):
        super().__init__()
        self.metas = metas
        self.heads = nn.ModuleDict(
            {m.name: nn.Linear(in_dim, m.output_dim) for m in metas}
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {m.name: self.heads[m.name](z) for m in self.metas}

    def compute_losses(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Per-target losses + weighted sum.

        Returns
        -------
        total
            Weighted total loss (scalar tensor).
        per_target
            Dict of name -> per-target loss scalar.
        """
        per: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=next(self.parameters()).device)
        for m in self.metas:
            pred = predictions[m.name]
            y = targets[m.name]
            loss = _loss_for(m.loss, pred, y, is_categorical=m.is_categorical)
            per[m.name] = loss
            total = total + m.weight * loss
        return total, per


def _loss_for(
    name: str, pred: torch.Tensor, y: torch.Tensor, *, is_categorical: bool
) -> torch.Tensor:
    name = name.lower()
    if is_categorical:
        if name in ("cross_entropy", "ce"):
            return F.cross_entropy(pred, y)
        if name == "bce":
            # binary cross-entropy with logits; expects (B, 1) or (B,)
            target_f = y.float()
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            return F.binary_cross_entropy_with_logits(pred, target_f)
        raise ValueError(f"Unsupported categorical loss: {name}")
    # numeric
    if pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if name == "mse":
        return F.mse_loss(pred, y)
    if name == "mae":
        return F.l1_loss(pred, y)
    if name == "huber":
        return F.smooth_l1_loss(pred, y)
    raise ValueError(f"Unsupported numeric loss: {name}")
