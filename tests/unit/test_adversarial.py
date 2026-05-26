"""Gradient Reversal + Adversarial Batch Head smoke tests."""

from __future__ import annotations

import torch

from ecofoundation.models.adversarial import (
    AdversarialBatchHead,
    grad_reverse,
    lambda_schedule,
)


def test_grad_reverse_forward_is_identity():
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, 0.5)
    assert torch.allclose(y, x)


def test_grad_reverse_negates_gradient_with_lambda():
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, 0.5)
    y.sum().backward()
    # Forward grad on y is 1; reversed: dL/dx = -0.5 * 1 → -0.5 per element.
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))


def test_adv_head_output_shape():
    head = AdversarialBatchHead(in_dim=16, n_batches=5, hidden_dim=8, dropout=0.0)
    z = torch.randn(7, 16)
    logits = head(z, lambda_=1.0)
    assert logits.shape == (7, 5)


def test_lambda_schedule_warmup():
    assert lambda_schedule(0, lambda_max=1.0, warmup_epochs=5) == 0.0
    assert lambda_schedule(3, lambda_max=1.0, warmup_epochs=5) == 0.6
    assert lambda_schedule(5, lambda_max=1.0, warmup_epochs=5) == 1.0
    assert lambda_schedule(99, lambda_max=2.0, warmup_epochs=5) == 2.0
    # Zero warmup -> immediately at max
    assert lambda_schedule(0, lambda_max=0.7, warmup_epochs=0) == 0.7


def test_adv_head_gradient_flow_through_encoder():
    """End-to-end: encoder weights should receive a *negative-pointing* push
    from the adversarial loss (vs. just adv head alone)."""
    import torch.nn as nn
    import torch.nn.functional as F

    enc = nn.Linear(8, 16, bias=False)
    head = AdversarialBatchHead(in_dim=16, n_batches=3, hidden_dim=8, dropout=0.0)

    x = torch.randn(6, 8)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    z = enc(x)
    logits = head(z, lambda_=1.0)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    # Encoder weight must have a non-zero gradient (GRL flipped it).
    assert enc.weight.grad is not None
    assert enc.weight.grad.abs().sum().item() > 0
