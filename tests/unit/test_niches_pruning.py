"""Edge pruning utilities."""

from __future__ import annotations

import numpy as np

from ecofoundation.niches.pruning import (
    apply_edge_pruning,
    prune_edges_absolute,
    prune_edges_quantile,
)


def test_absolute_cutoff():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    pruned, lengths = apply_edge_pruning(edges, coords, max_length=2.0)
    assert pruned.shape[0] == 1
    assert np.allclose(lengths, [1.0])


def test_quantile_cutoff():
    coords = np.array([[float(i), 0.0] for i in range(10)])
    edges = np.array([[0, i] for i in range(1, 10)])  # lengths 1..9
    pruned, lengths = apply_edge_pruning(edges, coords, quantile=0.5)
    # Median = 5 → keep edges with length ≤ 5 → 5 edges
    assert pruned.shape[0] == 5
    assert lengths.max() <= 5.0


def test_no_prune_when_neither_set():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    edges = np.array([[0, 1]])
    pruned, _ = apply_edge_pruning(edges, coords)
    assert pruned.shape[0] == 1


def test_empty_input():
    coords = np.empty((0, 2))
    edges = np.empty((0, 2), dtype=int)
    pruned, lengths = apply_edge_pruning(edges, coords, max_length=5.0)
    assert pruned.shape == (0, 2)
    assert lengths.shape == (0,)


def test_mask_helpers():
    edges = np.array([[0, 1], [1, 2]])
    lengths = np.array([1.0, 5.0])
    assert prune_edges_absolute(edges, lengths, 3.0).tolist() == [True, False]
    # Median of [1, 5] is 3.0 → only the length-1 edge survives.
    assert prune_edges_quantile(edges, lengths, 0.5).tolist() == [True, False]
