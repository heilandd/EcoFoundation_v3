"""Wrappers that expose a single-target tensor-returning model.

Both PyG's ``Explainer`` and Captum expect ``model(x, edge_index, edge_attr, batch) -> Tensor``.
Our trained models return a ``dict`` (multi-task). This wrapper selects one
target and exposes the standard signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SingleTargetWrapper(nn.Module):
    """Expose a single target's logits/value from a multi-task GNN."""

    def __init__(self, model, target_name: str):
        super().__init__()
        self.model = model
        self.target_name = target_name

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z = self.model._encode(x, edge_index, edge_attr, batch)
        out = self.model.head(z)
        return out[self.target_name]
