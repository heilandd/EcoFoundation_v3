"""Supervised graph classifiers."""

from ecofoundation.models.sup.gat_edge import GATEdgeClassifier
from ecofoundation.models.sup.gine import GINEClassifier

__all__ = ["GINEClassifier", "GATEdgeClassifier"]


def build_model(
    architecture: str,
    *,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float,
    pooling: str,
    batch_norm: bool,
    n_heads: int,
    target_metas,
):
    """Factory: build a supervised classifier by name."""
    if architecture == "gine":
        return GINEClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            batch_norm=batch_norm,
            target_metas=target_metas,
        )
    if architecture == "gat_edge":
        return GATEdgeClassifier(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling,
            batch_norm=batch_norm,
            n_heads=n_heads,
            target_metas=target_metas,
        )
    raise ValueError(f"Unknown architecture: {architecture!r}")
