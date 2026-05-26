"""GNN training: supervised trainer, unsupervised trainer, CV/splits, metrics."""

from ecofoundation.training.cv import SplitResult, build_splits
from ecofoundation.training.metrics import (
    EpochMetrics,
    categorical_metrics,
    numeric_metrics,
)
from ecofoundation.training.trainer import FoldResult, train_one_fold
from ecofoundation.training.unsup_trainer import (
    UnsupTrainResult,
    attach_adv_labels,
    encode_niche_embeddings,
    train_unsup,
)

__all__ = [
    "SplitResult",
    "build_splits",
    "EpochMetrics",
    "categorical_metrics",
    "numeric_metrics",
    "FoldResult",
    "train_one_fold",
    "UnsupTrainResult",
    "train_unsup",
    "encode_niche_embeddings",
    "attach_adv_labels",
]
