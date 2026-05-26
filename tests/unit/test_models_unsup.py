"""GAE / VGAE / DGI: forward + loss smoke tests."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from ecofoundation.models.unsup import build_unsup_model


def _toy_batch(n_graphs=3, n_nodes=8, node_dim=6, edge_dim=2):
    graphs = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, node_dim)
        edges = torch.tensor(
            [[i, (i + 1) % n_nodes] for i in range(n_nodes)]
            + [[(i + 1) % n_nodes, i] for i in range(n_nodes)],
            dtype=torch.long,
        ).T
        ea = torch.randn(edges.shape[1], edge_dim)
        graphs.append(Data(x=x, edge_index=edges, edge_attr=ea))
    return Batch.from_data_list(graphs)


def test_gae_forward_shape():
    m = build_unsup_model(
        "gae", node_dim=6, edge_dim=2, hidden_dim=8, n_layers=2,
        dropout=0.0, batch_norm=False,
    )
    m.eval()
    b = _toy_batch()
    z = m.encode(b.x, b.edge_index, b.edge_attr)
    assert z.shape == (b.num_nodes, 8)


def test_gae_reconstruction_loss_finite():
    m = build_unsup_model(
        "gae", node_dim=6, edge_dim=2, hidden_dim=8, n_layers=2,
        dropout=0.0, batch_norm=False,
    )
    b = _toy_batch()
    z = m.encode(b.x, b.edge_index, b.edge_attr)
    loss = m.reconstruction_loss(z, b.edge_index, n_neg_per_edge=1, batch=b.batch)
    assert torch.isfinite(loss).item()


def test_vgae_kl_finite():
    m = build_unsup_model(
        "vgae", node_dim=6, edge_dim=2, hidden_dim=8, n_layers=2,
        dropout=0.0, batch_norm=False,
    )
    m.train()
    b = _toy_batch()
    z = m.encode(b.x, b.edge_index, b.edge_attr)
    recon = m.reconstruction_loss(z, b.edge_index, n_neg_per_edge=1, batch=b.batch)
    kl = m.kl_term()
    assert torch.isfinite(recon).item()
    assert torch.isfinite(kl).item()


def test_dgi_loss_finite():
    m = build_unsup_model(
        "dgi", node_dim=6, edge_dim=2, hidden_dim=8, n_layers=2,
        dropout=0.0, batch_norm=False,
    )
    b = _toy_batch()
    loss = m.loss(b.x, b.edge_index, b.edge_attr, b.batch)
    assert torch.isfinite(loss).item()


def test_dgi_corruption_shape_preserved():
    m = build_unsup_model(
        "dgi", node_dim=6, edge_dim=2, hidden_dim=8, n_layers=1,
        dropout=0.0, batch_norm=False,
    )
    b = _toy_batch()
    x_corr = m.corrupt(b.x, b.batch)
    assert x_corr.shape == b.x.shape
