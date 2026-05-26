"""Niche strategies — unit tests against synthetic coordinate fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from ecofoundation.niches.base import NicheAssignment
from ecofoundation.niches.strategies.delaunay import DelaunayKHopStrategy
from ecofoundation.niches.strategies.knn import KNNStrategy
from ecofoundation.niches.strategies.radius import RadiusStrategy
from ecofoundation.niches.strategies.tiling import VoronoiTilingStrategy


def _grid_coords(n_per_side: int = 12, spacing: float = 10.0) -> np.ndarray:
    g = np.arange(n_per_side) * spacing
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(float)


def _two_group_coords(n_each: int = 100, sep: float = 1000.0):
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, size=(n_each, 2))
    b = rng.uniform(0, 100, size=(n_each, 2)) + sep
    coords = np.vstack([a, b])
    groups = np.array(["A"] * n_each + ["B"] * n_each)
    return coords, groups


# ----- shape ----------------------------------------------------------------


def test_delaunay_basic_shape():
    coords = _grid_coords(8)
    s = DelaunayKHopStrategy(k_hop=1, min_cells_per_niche=2)
    na = s.assign(coords, np.array(["P"] * coords.shape[0]))
    assert isinstance(na, NicheAssignment)
    assert na.n_niches > 0
    # k=1 hop on a 8x8 grid: interior cells have 4 Delaunay neighbours → niche size 5
    assert na.sizes().max() >= 5


def test_knn_each_cell_yields_niche_of_k_plus_one():
    coords = _grid_coords(10)
    k = 6
    s = KNNStrategy(k=k, min_cells_per_niche=2)
    na = s.assign(coords, np.array(["P"] * coords.shape[0]))
    # kNN returns k+1 (self included), all of equal size by construction
    sizes = na.sizes()
    assert sizes.size == coords.shape[0]
    assert np.all(sizes == k + 1)


def test_radius_excludes_isolated():
    rng = np.random.default_rng(0)
    cluster = rng.uniform(0, 5, size=(40, 2))
    # Only 3 isolated cells in a tight clump far away — under the min_cells_per_niche cap.
    isolated = rng.uniform(1000, 1005, size=(3, 2))
    coords = np.vstack([cluster, isolated])
    s = RadiusStrategy(radius=10.0, min_cells_per_niche=5)
    na = s.assign(coords, np.array(["P"] * coords.shape[0]))
    # isolated cells should not yield niches (< min_cells)
    assert na.n_niches == 40


def test_tiling_partitions_disjointly():
    coords = _grid_coords(10, spacing=10)
    s = VoronoiTilingStrategy(target_spacing=30.0, min_cells_per_niche=2)
    na = s.assign(coords, np.array(["P"] * coords.shape[0]))
    assert na.n_niches >= 2
    # union of all cell sets equals (subset of) input indices, no duplicates
    flat = np.concatenate(na.cells_per_niche)
    assert len(np.unique(flat)) == len(flat), "tiling produced overlapping cells"


# ----- patient separation ---------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        DelaunayKHopStrategy(k_hop=2, min_cells_per_niche=4),
        KNNStrategy(k=8, min_cells_per_niche=4),
        RadiusStrategy(radius=30.0, min_cells_per_niche=4),
        VoronoiTilingStrategy(target_spacing=40.0, min_cells_per_niche=4),
    ],
)
def test_no_cross_group_niche(strategy):
    coords, groups = _two_group_coords(n_each=80, sep=5000.0)
    na = strategy.assign(coords, groups)
    assert na.n_niches > 0, "expected niches to be produced for both groups"
    for niche_id in range(na.n_niches):
        cells = na.cells_per_niche[niche_id]
        unique_groups = set(groups[cells])
        assert len(unique_groups) == 1, f"niche {niche_id} spans groups: {unique_groups}"


# ----- min/max sizes --------------------------------------------------------


def test_min_cells_filter():
    coords = _grid_coords(8)
    s = DelaunayKHopStrategy(k_hop=1, min_cells_per_niche=100)  # impossible
    na = s.assign(coords, np.array(["P"] * coords.shape[0]))
    assert na.n_niches == 0


def test_max_cells_cap():
    coords = _grid_coords(15)
    s = DelaunayKHopStrategy(k_hop=4, min_cells_per_niche=2, max_cells_per_niche=10)
    na = s.assign(coords, np.array(["P"] * coords.shape[0]))
    if na.n_niches:
        assert na.sizes().max() <= 10


# ----- ego cell validity ----------------------------------------------------


def test_ego_cell_in_niche():
    coords = _grid_coords(8)
    s = DelaunayKHopStrategy(k_hop=1, min_cells_per_niche=2)
    na = s.assign(coords, np.array(["P"] * coords.shape[0]))
    for nid in range(na.n_niches):
        assert int(na.ego_cell[nid]) in na.cells_per_niche[nid].tolist()
