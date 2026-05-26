"""Train/test split logic.

Adaptive strategy:

  1. **patient_kfold** — if ``n_unique_patients >= min_patients_for_kfold``
     (default 5): GroupKFold by patient_id. For categorical targets we use
     StratifiedGroupKFold so class balance is preserved fold by fold.
  2. **subgraph_split** — otherwise: 70/30 split on the niche level, stratified
     by the target when categorical.

The split is on **niche indices** — the caller maps those into the underlying
PyG graph list / labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

from ecofoundation.config.schemas import TrainingConfig
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class SplitResult:
    """One or more train/test splits."""

    strategy: str
    folds: list[tuple[np.ndarray, np.ndarray]]  # list of (train_idx, test_idx)
    rationale: str


def build_splits(
    *,
    patients: np.ndarray,
    primary_label: np.ndarray | None,
    cfg: TrainingConfig,
    seed: int = 0,
) -> SplitResult:
    """Decide and produce CV splits.

    Parameters
    ----------
    patients
        Per-niche patient id, shape ``(n_niches,)``. Used for GroupKFold.
    primary_label
        Per-niche label for the *first* categorical target, used for
        stratification. Pass None to disable stratification.
    cfg
        :class:`TrainingConfig` controlling strategy + thresholds.
    seed
        RNG seed for stratified subgraph splits.
    """
    n_patients = int(len(np.unique(patients)))
    n_niches = len(patients)

    strategy = cfg.cv_strategy
    if strategy == "auto":
        strategy = (
            "patient_kfold" if n_patients >= cfg.min_patients_for_kfold else "subgraph_split"
        )

    if strategy == "patient_kfold":
        n_folds = min(cfg.n_folds, n_patients)
        if primary_label is not None and cfg.stratify_categorical:
            try:
                splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                folds = list(splitter.split(np.zeros(n_niches), primary_label, groups=patients))
                rationale = (
                    f"StratifiedGroupKFold(n_splits={n_folds}, groups=patient_id, "
                    f"stratify={primary_label is not None})"
                )
            except ValueError as e:
                _log.warning(f"StratifiedGroupKFold failed ({e}); falling back to GroupKFold.")
                splitter = GroupKFold(n_splits=n_folds)
                folds = list(splitter.split(np.zeros(n_niches), groups=patients))
                rationale = f"GroupKFold(n_splits={n_folds}, groups=patient_id)"
        else:
            splitter = GroupKFold(n_splits=n_folds)
            folds = list(splitter.split(np.zeros(n_niches), groups=patients))
            rationale = f"GroupKFold(n_splits={n_folds}, groups=patient_id)"
    elif strategy == "subgraph_split":
        idx = np.arange(n_niches)
        test_size = 1.0 - cfg.train_test_ratio
        if primary_label is not None and cfg.stratify_categorical:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(sss.split(idx, primary_label))
        else:
            train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
        folds = [(train_idx, test_idx)]
        rationale = (
            f"Subgraph {cfg.train_test_ratio:.0%}/{(1-cfg.train_test_ratio):.0%} split"
            f" (only {n_patients} patients — below min_patients_for_kfold={cfg.min_patients_for_kfold})"
        )
    else:
        raise ValueError(f"Unknown cv_strategy: {strategy!r}")

    _log.info(
        f"CV: strategy='{strategy}', folds={len(folds)}, n_niches={n_niches}, n_patients={n_patients}"
    )
    return SplitResult(strategy=strategy, folds=folds, rationale=rationale)
