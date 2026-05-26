"""Deep Graph Infomax (DGI) — contrastive unsupervised GNN.

DGI maximises the mutual information between local (node-level) embeddings
and a global summary of the graph. Architecture (per Veličković et al., 2018):

  1. Encoder: GINE stack → ``H = (n_nodes, hidden)``.
  2. Summary: ``s = sigmoid(mean(H))``.
  3. Corruption: row-shuffle the node feature matrix → ``X_neg`` → ``H_neg``.
  4. Discriminator: bilinear ``D(h, s) = sigmoid(h^T W s)`` →
     positives use real ``(H, s)``; negatives use ``(H_neg, s)``.
  5. Loss: BCE forcing positives to 1 and negatives to 0.

The final niche embedding is ``pool(H_real)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from ecofoundation.models.unsup.gae import GINEEncoder


class DGI(nn.Module):
    """Deep Graph Infomax with GINE encoder."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        batch_norm: bool = True,
        corruption: str = "row_shuffle",
    ):
        super().__init__()
        self.encoder = GINEEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.hidden_dim = hidden_dim
        self.corruption = corruption
        # Bilinear discriminator
        self.bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_attr)

    def summary(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Per-graph summary vector via sigmoid(mean pooling)."""
        # to_dense_batch returns (B, max_n, hidden) with a mask
        dense, mask = to_dense_batch(h, batch)
        sums = (dense * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        means = sums / counts
        return torch.sigmoid(means)

    def discriminate(self, h: torch.Tensor, s: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Per-node discriminator score: D(h_i, s_{batch(i)})."""
        # h: (n_total_nodes, hidden); s: (B, hidden); batch: (n_total_nodes,)
        # Map each node's batch index to its graph's summary.
        s_per_node = s[batch]
        return torch.einsum("nh,hk,nk->n", h, self.bilinear, s_per_node)

    def corrupt(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Within-batch row-shuffle. Keeps batch boundaries intact."""
        if self.corruption == "row_shuffle":
            x_corr = x.clone()
            for b in batch.unique():
                mask = batch == b
                idx = torch.where(mask)[0]
                perm = idx[torch.randperm(idx.numel(), device=x.device)]
                x_corr[idx] = x[perm]
            return x_corr
        if self.corruption == "feature_shuffle":
            # Shuffle along feature dim across all rows (preserves total marginals).
            perm = torch.randperm(x.size(1), device=x.device)
            return x[:, perm]
        raise ValueError(f"Unknown corruption: {self.corruption!r}")

    def loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h_pos = self.encode(x, edge_index, edge_attr)
        s = self.summary(h_pos, batch)
        # corrupted features → corrupted embeddings on same graph
        x_neg = self.corrupt(x, batch)
        h_neg = self.encode(x_neg, edge_index, edge_attr)

        d_pos = self.discriminate(h_pos, s, batch)
        d_neg = self.discriminate(h_neg, s, batch)

        pos_loss = F.binary_cross_entropy_with_logits(d_pos, torch.ones_like(d_pos))
        neg_loss = F.binary_cross_entropy_with_logits(d_neg, torch.zeros_like(d_neg))
        return pos_loss + neg_loss
