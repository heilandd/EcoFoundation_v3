"""GNN architectures — supervised (sup/) and unsupervised (unsup/).

Constraint: both node-features and edge-features must be attributable
for downstream explainability (Step 5). GINEConv and GATv2Conv with
edge_dim satisfy this constraint.

Adversarial debiasing (Step 6.5): :mod:`.adversarial` provides a Gradient
Reversal Layer + AdversarialBatchHead used by both trainers to push the
encoder toward sample/patient-invariant features.
"""

from ecofoundation.models.adversarial import (
    AdversarialBatchHead,
    grad_reverse,
    lambda_schedule,
)
from ecofoundation.models.heads import MultiTaskHead, TargetMeta, resolve_target_metas
from ecofoundation.models.pooling import build_pooling
from ecofoundation.models.sup import GATEdgeClassifier, GINEClassifier, build_model

__all__ = [
    "MultiTaskHead",
    "TargetMeta",
    "resolve_target_metas",
    "build_pooling",
    "GINEClassifier",
    "GATEdgeClassifier",
    "build_model",
    "AdversarialBatchHead",
    "grad_reverse",
    "lambda_schedule",
]
