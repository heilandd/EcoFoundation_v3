"""GATv2 with edge-feature attention.

Alternative to GINE: GATv2Conv computes per-edge attention coefficients
conditioned on both endpoints and the edge features (when ``edge_dim`` is set).
The attention weights fall out of the model directly and are saved in
``conv._alpha`` after a forward pass — useful for Step-5 explainability
without invoking GNNExplainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from ecofoundation.models.base import GraphClassifierBase
from ecofoundation.models.heads import TargetMeta
from ecofoundation.models.pooling import build_pooling


class GATEdgeClassifier(GraphClassifierBase):
    """Stack of ``n_layers`` GATv2Conv blocks + pooling + multi-task head."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        pooling: str,
        batch_norm: bool,
        n_heads: int,
        target_metas: list[TargetMeta],
    ):
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            target_metas=target_metas,
        )
        if hidden_dim % n_heads != 0:
            # Per-head dim must be integer; nudge user to a compatible config.
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}"
            )
        per_head = hidden_dim // n_heads
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=per_head,
                    heads=n_heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity())
        self.pool = build_pooling(pooling, hidden_dim)

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns, strict=True):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h_new = bn(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new
        return self.pool(h, batch)
