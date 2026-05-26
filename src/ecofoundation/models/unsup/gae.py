"""Graph Auto-Encoder (GAE) and Variational GAE for unsupervised niche embedding.

Written from scratch in PyG (per project requirement — no external GAE
implementation), using GINEConv as the encoder building block so that
node + edge features participate in message passing (same explainability
guarantees as the supervised classifier).

Decoder: dot-product on node embeddings + sigmoid → predicted edge
probability. Loss: binary cross-entropy on positive edges (from the niche
graph) and ``n_neg_samples_per_edge`` negative samples drawn from the
complement of the niche-internal edge set.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import negative_sampling


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim),
    )


class GINEEncoder(nn.Module):
    """Stack of GINEConv blocks producing per-node embeddings."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        batch_norm: bool,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GINEConv(_mlp(hidden_dim, hidden_dim), edge_dim=edge_dim) for _ in range(n_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity() for _ in range(n_layers)]
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns, strict=True):
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = bn(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GAE(nn.Module):
    """Standard GAE with inner-product decoder."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        batch_norm: bool = True,
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

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_attr)

    @staticmethod
    def decode(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Inner-product logits for the given edge pairs."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def reconstruction_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        n_neg_per_edge: int,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """BCE on positive edges + ``n_neg_per_edge`` negative samples per positive.

        Negative samples are drawn per-graph (within-batch) so we don't
        accidentally create cross-niche negative edges.
        """
        pos_logits = self.decode(z, pos_edge_index)
        # negative_sampling honours the batch index when ``num_nodes`` is given.
        n_nodes = z.size(0)
        num_neg = pos_edge_index.size(1) * n_neg_per_edge
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=n_nodes,
            num_neg_samples=num_neg,
            method="sparse",
        )
        neg_logits = self.decode(z, neg_edge_index)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )
        return pos_loss + neg_loss


class VGAE(GAE):
    """Variational GAE: encoder outputs μ + log σ; sample z = μ + σ·ε."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add a separate head for log_sigma. The base encoder's last linear of
        # the MLP becomes μ; we project μ to log_sigma.
        self.logsigma_head = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        mu = self.encoder(x, edge_index, edge_attr)
        if not self.training:
            return mu
        log_sigma = self.logsigma_head(mu)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * log_sigma) * eps
        # Stash params for the KL term.
        self._mu = mu
        self._log_sigma = log_sigma
        return z

    def kl_term(self) -> torch.Tensor:
        mu = self._mu
        log_sigma = self._log_sigma
        return -0.5 * torch.mean(1 + log_sigma - mu.pow(2) - log_sigma.exp())
