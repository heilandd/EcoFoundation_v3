"""GINE graph classifier (default).

GINE = GIN + edge-feature integration. Each layer aggregates neighbour features
as ``MLP( (1+eps) * x + sum_{j ∈ N(i)} ReLU(x_j + edge_proj(e_ij)) )``,
so edge features participate in message-passing and contribute to gradients —
which is what we need for the Step-5 explainability layer.

Reference: Hu et al. (2019), "Strategies for Pre-training Graph Neural Networks".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from ecofoundation.models.base import GraphClassifierBase
from ecofoundation.models.heads import TargetMeta
from ecofoundation.models.pooling import build_pooling


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim),
    )


class GINEClassifier(GraphClassifierBase):
    """Stack of ``n_layers`` GINEConv blocks + pooling + multi-task head."""

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

        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GINEConv(_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim))
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
