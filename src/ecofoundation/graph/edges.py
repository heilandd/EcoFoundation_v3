"""Intra-niche edge construction.

Two topologies share the same interface:

  - ``delaunay_intra_niche`` (default): edges are the subset of the patient-wide
    Delaunay triangulation that lies entirely within a niche. Cached per patient
    so we don't re-triangulate for every niche.
  - ``knn_intra_niche``: each cell within the niche connects to its k nearest
    intra-niche neighbours.

Both return a directed (symmetric) edge list in **local niche indices** —
i.e. indexed 0..n_niche_cells-1, not into the global AnnData.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from ecofoundation.niches.pruning import apply_edge_pruning
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class NicheEdges:
    """Edges of one niche, in local indices."""

    edges: np.ndarray  # (E, 2) int64, undirected pairs (i, j) with i < j
    distances: np.ndarray  # (E,)
    topology: str


def build_niche_edges(
    cell_indices: np.ndarray,
    *,
    coords: np.ndarray,
    delaunay_cache: "DelaunayCache | None" = None,
    topology: str = "delaunay_intra_niche",
    knn_k: int = 8,
    max_edge_length: float | None = None,
    edge_length_quantile_cutoff: float | None = None,
) -> NicheEdges:
    """Build the intra-niche edge list.

    Parameters
    ----------
    cell_indices
        Global indices (into the patient-level coords) of the niche cells.
    coords
        Patient-level coords (``n_patient_cells, 2``).
    delaunay_cache
        Cache holding the patient's full Delaunay edges. Required when topology
        is ``"delaunay_intra_niche"``.
    """
    local_coords = coords[cell_indices]

    if topology == "delaunay_intra_niche":
        if delaunay_cache is None:
            raise ValueError("delaunay_intra_niche requires a DelaunayCache")
        edges_local, lengths = _delaunay_sub_edges(cell_indices, delaunay_cache)
    elif topology == "knn_intra_niche":
        edges_local, lengths = _knn_edges(local_coords, knn_k)
    else:
        raise ValueError(f"Unknown topology: {topology!r}")

    edges_local, lengths = apply_edge_pruning(
        edges_local,
        local_coords,
        max_length=max_edge_length,
        quantile=edge_length_quantile_cutoff,
    )
    return NicheEdges(edges=edges_local, distances=lengths, topology=topology)


# ---------------------------------------------------------------------------
# Delaunay cache (per patient, computed once)
# ---------------------------------------------------------------------------


class DelaunayCache:
    """Holds the Delaunay edge list for one patient's coords.

    The edges are stored as a sorted (E, 2) int64 array; cell membership lookup
    uses a hash set per query. For ~50k cells per patient this is fast enough
    (Delaunay on 50k pts: <2 s; sub-edge extraction is O(E)).
    """

    def __init__(self, coords: np.ndarray):
        self.coords = coords
        self._global_edges: np.ndarray | None = None

    def edges(self) -> np.ndarray:
        if self._global_edges is None:
            self._global_edges = _delaunay_edge_list(self.coords)
        return self._global_edges


def _delaunay_edge_list(coords: np.ndarray) -> np.ndarray:
    """Compute the deduplicated undirected edge list of a Delaunay triangulation."""
    if coords.shape[0] < 4:
        return np.empty((0, 2), dtype=np.int64)
    tri = Delaunay(coords)
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(a, c), max(a, c)))
        edge_set.add((min(b, c), max(b, c)))
    return np.asarray(sorted(edge_set), dtype=np.int64)


def _delaunay_sub_edges(
    cell_indices: np.ndarray, cache: DelaunayCache
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only edges of the global Delaunay where both endpoints are in the niche.

    Vectorised: builds a boolean ``in_niche`` mask over patient-cell indices and
    indexes the edge list with it. This is critical — the Python-loop version
    was O(E_patient) per niche in pure Python and stalled at ~10k niches.
    """
    global_edges = cache.edges()
    if global_edges.shape[0] == 0 or cell_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    n_patient = cache.coords.shape[0]
    in_niche = np.zeros(n_patient, dtype=bool)
    in_niche[cell_indices] = True

    # Vectorised membership check.
    keep = in_niche[global_edges[:, 0]] & in_niche[global_edges[:, 1]]
    sub_global = global_edges[keep]

    if sub_global.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    # Remap patient-local indices → niche-local via a vector lookup table.
    remap = np.full(n_patient, -1, dtype=np.int64)
    remap[cell_indices] = np.arange(cell_indices.size, dtype=np.int64)
    edges_local = remap[sub_global]

    coords = cache.coords[cell_indices]
    diffs = coords[edges_local[:, 0]] - coords[edges_local[:, 1]]
    lengths = np.sqrt((diffs * diffs).sum(axis=1))
    return edges_local, lengths


# ---------------------------------------------------------------------------
# kNN fallback topology
# ---------------------------------------------------------------------------


def _knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    dists, idx = nn.kneighbors(coords)

    rows: list[tuple[int, int]] = []
    lengths: list[float] = []
    for i in range(n):
        for j_pos in range(1, k + 1):  # skip self at position 0
            j = int(idx[i, j_pos])
            if i < j:
                rows.append((i, j))
                lengths.append(float(dists[i, j_pos]))
            elif j < i:
                rows.append((j, i))
                lengths.append(float(dists[i, j_pos]))
    if not rows:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    # Deduplicate (i, j) pairs (kNN is asymmetric)
    arr = np.asarray(rows, dtype=np.int64)
    lens = np.asarray(lengths, dtype=np.float64)
    # unique pairs, keep min length
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr_sorted = arr[order]
    lens_sorted = lens[order]
    unique_mask = np.ones(arr_sorted.shape[0], dtype=bool)
    if arr_sorted.shape[0] > 1:
        same = (arr_sorted[1:] == arr_sorted[:-1]).all(axis=1)
        unique_mask[1:] = ~same
    return arr_sorted[unique_mask], lens_sorted[unique_mask]
