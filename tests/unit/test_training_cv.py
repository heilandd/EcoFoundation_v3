"""Adaptive CV strategy."""

from __future__ import annotations

import numpy as np

from ecofoundation.config.schemas import TargetSpec, TrainingConfig
from ecofoundation.training.cv import build_splits


def _cfg(strategy="auto", n_folds=3, min_patients=5):
    return TrainingConfig(
        targets=[TargetSpec(name="t", obs_column="t")],
        cv_strategy=strategy,
        n_folds=n_folds,
        min_patients_for_kfold=min_patients,
    )


def test_patient_kfold_when_enough_patients():
    patients = np.repeat(np.arange(5), 20)  # 5 patients
    label = np.tile([0, 1], 50)
    splits = build_splits(patients=patients, primary_label=label, cfg=_cfg(n_folds=5), seed=0)
    assert splits.strategy == "patient_kfold"
    assert len(splits.folds) == 5
    # No patient appears in both train and test of any fold
    for train_idx, test_idx in splits.folds:
        train_pats = set(patients[train_idx])
        test_pats = set(patients[test_idx])
        assert train_pats.isdisjoint(test_pats)


def test_subgraph_split_when_few_patients():
    patients = np.repeat(np.arange(3), 30)  # only 3 patients
    label = np.tile([0, 1], 45)
    splits = build_splits(patients=patients, primary_label=label, cfg=_cfg(min_patients=5), seed=0)
    assert splits.strategy == "subgraph_split"
    assert len(splits.folds) == 1
    train_idx, test_idx = splits.folds[0]
    assert abs(len(train_idx) / 90 - 0.7) < 0.05


def test_stratified_subgraph_preserves_balance():
    patients = np.repeat(np.arange(2), 50)
    label = np.array([0] * 80 + [1] * 20)  # imbalanced
    splits = build_splits(patients=patients, primary_label=label, cfg=_cfg(min_patients=5), seed=0)
    assert splits.strategy == "subgraph_split"
    _, test_idx = splits.folds[0]
    test_balance = label[test_idx].mean()
    assert 0.10 < test_balance < 0.30  # ≈ 20% of class 1
