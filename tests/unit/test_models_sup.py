"""GINE and GAT-edge classifiers — forward pass + shapes."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from ecofoundation.models import build_model
from ecofoundation.models.heads import TargetMeta


def _toy_batch(n_graphs=3, n_nodes=10, node_dim=8, edge_dim=2):
    graphs = []
    for g in range(n_graphs):
        x = torch.randn(n_nodes, node_dim)
        edges = torch.tensor(
            [[i, (i + 1) % n_nodes] for i in range(n_nodes)] +
            [[(i + 1) % n_nodes, i] for i in range(n_nodes)],
            dtype=torch.long,
        ).T
        ea = torch.randn(edges.shape[1], edge_dim)
        graphs.append(Data(x=x, edge_index=edges, edge_attr=ea))
    return Batch.from_data_list(graphs)


def _metas():
    return [
        TargetMeta(name="y", type="categorical", output_dim=2, classes=["a", "b"], loss="cross_entropy", weight=1.0),
    ]


def test_gine_forward_shapes():
    batch = _toy_batch()
    model = build_model(
        "gine",
        node_dim=8, edge_dim=2, hidden_dim=16, n_layers=2,
        dropout=0.0, pooling="attention", batch_norm=True, n_heads=4,
        target_metas=_metas(),
    )
    model.eval()
    out = model(batch)
    assert out["y"].shape == (3, 2)  # B graphs × n_classes


def test_gat_edge_forward_shapes():
    batch = _toy_batch()
    model = build_model(
        "gat_edge",
        node_dim=8, edge_dim=2, hidden_dim=16, n_layers=2,
        dropout=0.0, pooling="mean", batch_norm=True, n_heads=4,
        target_metas=_metas(),
    )
    model.eval()
    out = model(batch)
    assert out["y"].shape == (3, 2)


def test_gine_backward_flows_through_edges():
    """Verify edges DO influence the loss — needed for explainability later."""
    batch = _toy_batch()
    model = build_model(
        "gine",
        node_dim=8, edge_dim=2, hidden_dim=16, n_layers=2,
        dropout=0.0, pooling="mean", batch_norm=False, n_heads=4,
        target_metas=_metas(),
    )
    batch.edge_attr.requires_grad_(True)
    out = model(batch)
    out["y"].sum().backward()
    assert batch.edge_attr.grad is not None
    assert (batch.edge_attr.grad.abs().sum() > 0).item()
