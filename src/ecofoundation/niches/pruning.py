"""Edge-length pruning for niche graphs.

Long edges in a Delaunay triangulation or radius graph often connect cells
across tissue gaps (e.g. across vessels or sectioning artefacts). Pruning
keeps niches biologically coherent.

Two modes:
  - absolute cutoff in µm (``max_edge_length``)
  - per-niche quantile cutoff (``edge_length_quantile_cutoff``, e.g. 0.95
    drops the top 5 % longest edges within the niche's edges)
"""

from __future__ import annotations

import numpy as np


def prune_edges_absolute(
    edges: np.ndarray, lengths: np.ndarray, max_length: float
) -> np.ndarray:
    """Keep edges with ``length <= max_length``. Returns boolean mask."""
    return lengths <= max_length


def prune_edges_quantile(
    edges: np.ndarray, lengths: np.ndarray, quantile: float
) -> np.ndarray:
    """Drop edges above the ``quantile``-th length percentile.

    quantile=0.95 → drop the longest 5 % of edges.
    """
    if edges.shape[0] == 0:
        return np.ones(0, dtype=bool)
    cutoff = np.quantile(lengths, quantile)
    return lengths <= cutoff


def apply_edge_pruning(
    edges: np.ndarray,
    coords: np.ndarray,
    *,
    max_length: float | None = None,
    quantile: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply distance-based pruning to an edge list.

    Parameters
    ----------
    edges
        ``(E, 2)`` int array of cell indices.
    coords
        ``(N, 2)`` cell coordinates.
    max_length
        If set, drop edges longer than this in coordinate units.
    quantile
        If set (and ``max_length`` is None), drop edges above this quantile.

    Returns
    -------
    pruned_edges
        Surviving edges, shape ``(E', 2)``.
    pruned_lengths
        Their euclidean lengths, shape ``(E',)``.
    """
    if edges.shape[0] == 0:
        return edges, np.empty(0, dtype=np.float64)

    deltas = coords[edges[:, 0]] - coords[edges[:, 1]]
    lengths = np.sqrt((deltas * deltas).sum(axis=1))

    if max_length is not None:
        mask = prune_edges_absolute(edges, lengths, max_length)
    elif quantile is not None:
        mask = prune_edges_quantile(edges, lengths, quantile)
    else:
        mask = np.ones(edges.shape[0], dtype=bool)

    return edges[mask], lengths[mask]
