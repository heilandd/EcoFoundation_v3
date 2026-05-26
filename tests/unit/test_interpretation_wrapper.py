"""SingleTargetWrapper."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from ecofoundation.interpretation import SingleTargetWrapper
from ecofoundation.models import build_model
from ecofoundation.models.heads import TargetMeta


def _toy_data(n_nodes=8, node_dim=6, edge_dim=2):
    x = torch.randn(n_nodes, node_dim)
    ei = torch.tensor(
        [[i, (i + 1) % n_nodes] for i in range(n_nodes)] +
        [[(i + 1) % n_nodes, i] for i in range(n_nodes)],
        dtype=torch.long,
    ).T
    ea = torch.randn(ei.shape[1], edge_dim)
    return Data(x=x, edge_index=ei, edge_attr=ea)


def test_wrapper_returns_tensor_for_one_target():
    metas = [
        TargetMeta(name="y", type="categorical", output_dim=2, classes=["a", "b"], loss="cross_entropy", weight=1.0),
        TargetMeta(name="z", type="numeric", output_dim=1, classes=None, loss="mse", weight=0.5),
    ]
    model = build_model(
        "gine", node_dim=6, edge_dim=2, hidden_dim=8, n_layers=1,
        dropout=0.0, pooling="mean", batch_norm=False, n_heads=2, target_metas=metas,
    )
    model.eval()
    wrapper_y = SingleTargetWrapper(model, "y")
    wrapper_z = SingleTargetWrapper(model, "z")

    batch = Batch.from_data_list([_toy_data(), _toy_data()])
    y_out = wrapper_y(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    z_out = wrapper_z(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert y_out.shape == (2, 2)
    assert z_out.shape == (2, 1)


def test_wrapper_infers_batch_for_single_graph():
    metas = [TargetMeta(name="y", type="categorical", output_dim=2, classes=["a","b"], loss="cross_entropy", weight=1.0)]
    model = build_model(
        "gine", node_dim=6, edge_dim=2, hidden_dim=8, n_layers=1,
        dropout=0.0, pooling="mean", batch_norm=False, n_heads=2, target_metas=metas,
    )
    model.eval()
    wrapper = SingleTargetWrapper(model, "y")
    d = _toy_data()
    out = wrapper(d.x, d.edge_index, d.edge_attr, batch=None)
    assert out.shape == (1, 2)
