"""Unsupervised GNN models for spatial-niche embedding.

All three (GAE, VGAE, DGI) are implemented from scratch on top of GINEConv
so that node + edge features participate in message passing — same
explainability guarantees as the supervised path.
"""

from ecofoundation.models.unsup.dgi import DGI
from ecofoundation.models.unsup.gae import GAE, VGAE, GINEEncoder

__all__ = ["GAE", "VGAE", "DGI", "GINEEncoder", "build_unsup_model"]


def build_unsup_model(
    architecture: str,
    *,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float,
    batch_norm: bool,
    dgi_corruption: str = "row_shuffle",
):
    kwargs = {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "batch_norm": batch_norm,
    }
    if architecture == "gae":
        return GAE(**kwargs)
    if architecture == "vgae":
        return VGAE(**kwargs)
    if architecture == "dgi":
        return DGI(**kwargs, corruption=dgi_corruption)
    raise ValueError(f"Unknown unsupervised architecture: {architecture!r}")
