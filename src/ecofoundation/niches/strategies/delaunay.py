"""Delaunay-triangulation-based ego niches with k-hop neighbourhoods.

Default strategy in EcoFoundation. For each cell:
  1. Build a Delaunay triangulation of all cells in this group (patient).
  2. Optionally prune edges that are too long (gaps in tissue, sectioning artefacts).
  3. BFS the ego cell up to ``k_hop`` hops along the pruned graph.
  4. The visited set is the niche.

This gives biologically meaningful local neighborhoods that respect cell
density (a 3-hop neighborhood in a dense region is smaller in µm than in a
sparse one).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from ecofoundation.niches.base import NicheStrategy
from ecofoundation.niches.pruning import apply_edge_pruning


class DelaunayKHopStrategy(NicheStrategy):
    """Default niche strategy: k-hop BFS on a (pruned) Delaunay graph."""

    name = "delaunay"

    def __init__(
        self,
        *,
        k_hop: int = 3,
        max_edge_length: float | None = None,
        edge_length_quantile_cutoff: float | None = 0.95,
        min_cells_per_niche: int = 5,
        max_cells_per_niche: int | None = 500,
        random_state: int = 0,
    ):
        super().__init__(
            min_cells_per_niche=min_cells_per_niche,
            max_cells_per_niche=max_cells_per_niche,
            random_state=random_state,
        )
        if not (1 <= k_hop <= 8):
            raise ValueError(f"k_hop must be in [1,8]; got {k_hop}")
        if max_edge_length is not None and edge_length_quantile_cutoff is not None:
            # absolute wins; we ignore quantile in that case
            edge_length_quantile_cutoff = None
        self.k_hop = k_hop
        self.max_edge_length = max_edge_length
        self.edge_length_quantile_cutoff = edge_length_quantile_cutoff

    def params(self) -> dict[str, Any]:
        return {
            **super().params(),
            "k_hop": self.k_hop,
            "max_edge_length": self.max_edge_length,
            "edge_length_quantile_cutoff": self.edge_length_quantile_cutoff,
        }

    def _assign_group(
        self, coords: np.ndarray
    ) -> tuple[list[np.ndarray], list[int]]:
        n = coords.shape[0]
        if n < 4:  # Delaunay needs ≥ 3 non-collinear points
            return [], []

        try:
            tri = Delaunay(coords)
        except Exception:  # collinear etc.
            return [], []

        # Edge list from simplices (deduplicated)
        edge_set: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
            edge_set.add((min(a, b), max(a, b)))
            edge_set.add((min(a, c), max(a, c)))
            edge_set.add((min(b, c), max(b, c)))
        edges = np.asarray(sorted(edge_set), dtype=np.int64)

        edges, _ = apply_edge_pruning(
            edges,
            coords,
            max_length=self.max_edge_length,
            quantile=self.edge_length_quantile_cutoff,
        )

        # Build adjacency
        adj: dict[int, list[int]] = defaultdict(list)
        for a, b in edges:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))

        # BFS k-hop from every cell
        cells_per_niche: list[np.ndarray] = []
        ego_cells: list[int] = []
        for ego in range(n):
            niche = _bfs_k_hop(ego, adj, self.k_hop)
            if len(niche) < self.min_cells_per_niche:
                continue
            cells_per_niche.append(np.fromiter(sorted(niche), dtype=np.int64))
            ego_cells.append(ego)

        return cells_per_niche, ego_cells


def _bfs_k_hop(start: int, adj: dict[int, list[int]], k: int) -> set[int]:
    """Iterative BFS up to depth ``k`` from ``start``."""
    visited: set[int] = {start}
    frontier: set[int] = {start}
    for _ in range(k):
        next_frontier: set[int] = set()
        for node in frontier:
            for nb in adj.get(node, ()):
                if nb not in visited:
                    next_frontier.add(nb)
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier
    return visited
