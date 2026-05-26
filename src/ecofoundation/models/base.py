"""Shared base for supervised graph classifiers.

All models accept ``(x, edge_index, edge_attr, batch)`` and return a dict of
per-target predictions. They expose ``num_node_features``, ``num_edge_features``
and a ``hidden_dim`` attribute so the trainer can size the MultiTaskHead.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from ecofoundation.models.heads import MultiTaskHead, TargetMeta


class GraphClassifierBase(nn.Module):
    """Common ancestor; subclasses implement ``_encode`` returning a per-graph embedding."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        target_metas: list[TargetMeta],
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.head = MultiTaskHead(hidden_dim, target_metas)
        self.target_metas = target_metas

    @abstractmethod
    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return a ``(B, hidden_dim)`` per-graph embedding."""
        raise NotImplementedError

    def forward(
        self, data: Batch, *, return_embedding: bool = False
    ):
        """Returns ``head(z)`` by default; ``(head(z), z)`` when ``return_embedding``.

        The pooled graph embedding ``z`` is what the adversarial head consumes,
        so the trainer toggles ``return_embedding=True`` during training.
        """
        z = self._encode(data.x, data.edge_index, data.edge_attr, data.batch)
        out = self.head(z)
        if return_embedding:
            return out, z
        return out
