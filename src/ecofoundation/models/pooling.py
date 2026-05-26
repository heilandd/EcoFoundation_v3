"""Graph-level pooling helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GlobalAttention, global_max_pool, global_mean_pool


class AttentionPool(nn.Module):
    """Gated attention pooling over node embeddings.

    Reduces ``(N_nodes, hidden)`` → ``(B, hidden)`` using a learned gate per node.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        gate_nn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2 if in_dim >= 4 else in_dim),
            nn.GELU(),
            nn.Linear(in_dim // 2 if in_dim >= 4 else in_dim, 1),
        )
        self.pool = GlobalAttention(gate_nn=gate_nn)
        # Cache the most recent attention scores so callers can inspect them.
        self._last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        out = self.pool(x, batch)
        # PyG GlobalAttention exposes the gating logits internally; we recompute
        # them in eval-mode hooks elsewhere. Avoid relying on a private attribute here.
        return out


def build_pooling(kind: str, in_dim: int) -> nn.Module:
    """Factory returning a pooling module callable as ``(x, batch) → (B, hidden)``."""
    kind = kind.lower()
    if kind == "mean":
        return _GlobalPool(global_mean_pool)
    if kind == "max":
        return _GlobalPool(global_max_pool)
    if kind == "attention":
        return AttentionPool(in_dim)
    raise ValueError(f"Unknown pooling: {kind!r}")


class _GlobalPool(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.fn(x, batch)
