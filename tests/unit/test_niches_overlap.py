"""Overlap controller — enforces the Jaccard cap."""

from __future__ import annotations

import numpy as np
import pytest

from ecofoundation.niches.base import NicheAssignment
from ecofoundation.niches.overlap import enforce_overlap_limit


def _make_synthetic(cells_per_niche: list[list[int]], groups: list[str]) -> NicheAssignment:
    n = len(cells_per_niche)
    return NicheAssignment(
        cells_per_niche=[np.asarray(c, dtype=np.int64) for c in cells_per_niche],
        ego_cell=np.array([c[0] for c in cells_per_niche], dtype=np.int64),
        group_label=np.array(groups, dtype=object),
        sample_label=None,
        centroid=np.array([[0.0, 0.0]] * n),
        strategy_name="synthetic",
        params={},
        n_source_cells=1000,
    )


def test_disjoint_partition_passes_through_unchanged():
    n = _make_synthetic([[0, 1, 2], [3, 4, 5], [6, 7, 8]], ["A", "A", "A"])
    filtered, info = enforce_overlap_limit(n, max_overlap_fraction=0.2)
    assert filtered.n_niches == 3
    assert info.dropped_indices.size == 0


def test_full_overlap_drops_smaller():
    # Two identical niches → keep one (the first by stable sort on size).
    n = _make_synthetic([[0, 1, 2, 3], [0, 1, 2, 3]], ["A", "A"])
    filtered, info = enforce_overlap_limit(n, max_overlap_fraction=0.2)
    assert filtered.n_niches == 1
    assert info.dropped_indices.size == 1


def test_max_overlap_zero_yields_disjoint():
    # Two niches share 1 cell out of 4 → Jaccard = 1/7 ≈ 0.143
    n = _make_synthetic([[0, 1, 2, 3], [3, 4, 5, 6]], ["A", "A"])
    filtered, _ = enforce_overlap_limit(n, max_overlap_fraction=0.0)
    assert filtered.n_niches == 1, "overlap=0 must drop any niche sharing cells"


def test_cross_group_overlap_ignored():
    # Same cell indices, but different groups → not compared
    n = _make_synthetic([[0, 1, 2, 3], [0, 1, 2, 3]], ["A", "B"])
    filtered, _ = enforce_overlap_limit(n, max_overlap_fraction=0.0)
    assert filtered.n_niches == 2, "niches in different groups must never be conflict-filtered"


def test_keeps_larger_niche_first():
    n = _make_synthetic([[0, 1], [0, 1, 2, 3, 4, 5]], ["A", "A"])
    filtered, info = enforce_overlap_limit(n, max_overlap_fraction=0.0)
    # The 6-cell niche wins; the 2-cell niche is dropped.
    assert filtered.n_niches == 1
    assert filtered.sizes()[0] == 6


@pytest.mark.parametrize("cap", [0.0, 0.1, 0.3, 0.5, 0.9, 1.0])
def test_invariant_pairwise_jaccard_under_cap(cap):
    rng = np.random.default_rng(0)
    niches = [sorted(rng.choice(50, size=20, replace=False).tolist()) for _ in range(15)]
    na = _make_synthetic(niches, ["A"] * 15)
    filtered, _ = enforce_overlap_limit(na, max_overlap_fraction=cap)
    # Verify the invariant directly on the kept set.
    for i in range(filtered.n_niches):
        for j in range(i + 1, filtered.n_niches):
            assert filtered.jaccard(i, j) <= cap + 1e-9, (
                f"violation at ({i},{j}): {filtered.jaccard(i, j):.3f} > {cap}"
            )
