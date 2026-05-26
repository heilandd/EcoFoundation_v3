"""Domain-Adversarial debiasing — Gradient Reversal + Batch Discriminator.

Implements the DANN trick (Ganin et al. 2016) so that the encoder learns
features that are *invariant* to a nuisance variable (sample / patient ID).

Setup
-----
- ``GradReverse`` — autograd function that flips the gradient sign during
  backprop (forward is identity).
- ``AdversarialBatchHead`` — small MLP that predicts the nuisance label from
  the (graph-level) embedding ``z``. In the forward, ``z`` goes through
  ``grad_reverse(z, lambda_)`` first; the head is trained normally on
  cross-entropy, but the encoder receives ``-lambda * dL_adv/dz``, pushing
  it AWAY from batch-discriminating features.

Loss composition (trainer side)::

    total = task_loss + adv_loss

Because of the reverse-layer the encoder's gradient ends up
``dL_task/dz - lambda * dL_adv/dz``, while the adv head's gradient is the
usual ``+dL_adv/dW_adv``.

Lambda warmup
-------------
``lambda`` is ramped from 0 → ``lambda_max`` over ``warmup_epochs`` so the
adversary has time to learn before it starts pushing the encoder around.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    """Identity in the forward pass; negated * lambda in the backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # noqa: D401
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # noqa: D401
        return grad.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradReverse.apply(x, lambda_)


class AdversarialBatchHead(nn.Module):
    """Small classifier predicting batch / nuisance labels from ``z``.

    Parameters
    ----------
    in_dim
        Input embedding dimension (matches ``encoder hidden_dim``).
    n_batches
        Number of nuisance classes (e.g. number of patients).
    hidden_dim
        Width of the MLP hidden layer.
    dropout
        Dropout on the MLP hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        n_batches: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_batches = n_batches
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_batches),
        )

    def forward(self, z: torch.Tensor, lambda_: float) -> torch.Tensor:
        z_rev = grad_reverse(z, lambda_)
        return self.net(z_rev)


def lambda_schedule(epoch: int, *, lambda_max: float, warmup_epochs: int) -> float:
    """Linear warm-up from 0 to ``lambda_max`` over ``warmup_epochs`` epochs."""
    if warmup_epochs <= 0:
        return float(lambda_max)
    return float(lambda_max) * min(1.0, max(0.0, epoch) / float(warmup_epochs))
