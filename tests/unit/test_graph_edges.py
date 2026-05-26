"""Intra-niche edge construction."""

from __future__ import annotations

import numpy as np

from ecofoundation.graph.edges import (
    DelaunayCache,
    build_niche_edges,
)


def _grid(n=10, spacing=5.0):
    g = np.arange(n) * spacing
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(float)


def test_delaunay_intra_niche_returns_sub_edges():
    coords = _grid(8)
    cache = DelaunayCache(coords)
    cache.global_to_patient_local = {i: i for i in range(coords.shape[0])}

    # Niche = first 3x3 block of the grid → 9 cells
    cells = np.array([r * 8 + c for r in range(3) for c in range(3)], dtype=np.int64)
    ne = build_niche_edges(cells, coords=coords, delaunay_cache=cache)
    assert ne.edges.shape[0] > 0
    # All edges live within the niche (local indices 0..8)
    assert ne.edges.max() < 9
    # Lengths match euclidean distances
    sub_coords = coords[cells]
    for k in range(ne.edges.shape[0]):
        a, b = ne.edges[k]
        expected = np.linalg.norm(sub_coords[a] - sub_coords[b])
        assert np.isclose(ne.distances[k], expected, atol=1e-6)


def test_knn_intra_niche():
    coords = _grid(6)
    cells = np.arange(coords.shape[0], dtype=np.int64)
    ne = build_niche_edges(
        cells, coords=coords, topology="knn_intra_niche", knn_k=4
    )
    assert ne.edges.shape[0] > 0
    # No duplicate edges
    pairs = {tuple(e) for e in ne.edges.tolist()}
    assert len(pairs) == ne.edges.shape[0]


def test_edge_pruning_max_length():
    # Place cells on a line: 0, 1, 2, 10  → edges to "10" should be pruned at max=3
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    cells = np.arange(4, dtype=np.int64)
    ne = build_niche_edges(
        cells,
        coords=coords,
        topology="knn_intra_niche",
        knn_k=3,
        max_edge_length=3.0,
    )
    # Cells 0,1,2 cluster — edges to 3 (at x=10) must be gone.
    assert all(ne.distances <= 3.0)


def test_delaunay_cache_reused():
    coords = _grid(8)
    cache = DelaunayCache(coords)
    cache.global_to_patient_local = {i: i for i in range(coords.shape[0])}
    e1 = cache.edges()
    e2 = cache.edges()
    assert e1 is e2  # cached
