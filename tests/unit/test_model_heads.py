"""MultiTaskHead + loss computation."""

from __future__ import annotations

import torch

from ecofoundation.config.schemas import TargetSpec
from ecofoundation.models.heads import MultiTaskHead, TargetMeta, resolve_target_metas


def test_resolve_categorical_infers_classes():
    spec = TargetSpec(name="t", obs_column="t", type="categorical")
    metas = resolve_target_metas([spec], {"t": ["a", "b", "a", "c"]})
    assert metas[0].output_dim == 3
    assert metas[0].classes == ["a", "b", "c"]


def test_resolve_numeric():
    spec = TargetSpec(name="x", obs_column="x", type="numeric", loss="mse")
    metas = resolve_target_metas([spec], {"x": [1.0, 2.0, 3.0]})
    assert metas[0].output_dim == 1
    assert metas[0].classes is None


def test_multitask_head_shape_and_loss():
    metas = [
        TargetMeta(name="ct", type="categorical", output_dim=3, classes=["a", "b", "c"], loss="cross_entropy", weight=1.0),
        TargetMeta(name="x", type="numeric", output_dim=1, classes=None, loss="mse", weight=0.5),
    ]
    head = MultiTaskHead(in_dim=8, metas=metas)
    z = torch.randn(4, 8)
    out = head(z)
    assert out["ct"].shape == (4, 3)
    assert out["x"].shape == (4, 1)
    y = {
        "ct": torch.tensor([0, 1, 2, 1]),
        "x": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }
    total, per = head.compute_losses(out, y)
    assert total.ndim == 0
    assert set(per.keys()) == {"ct", "x"}
