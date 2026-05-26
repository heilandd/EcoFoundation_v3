"""Voronoi tiling: non-overlapping niche partition via Farthest-Point-Sampling.

Algorithm:
  1. Pick a seed cell uniformly at random.
  2. Iteratively pick the cell **farthest** from the current set of seeds (FPS).
  3. Continue until the next farthest distance falls below ``target_spacing``,
     or we have ``n_seeds`` seeds.
  4. Assign every cell to its nearest seed (Voronoi partition).

This produces a strict disjoint partition (no overlap by construction) with
seeds spread roughly evenly in proportion to cell density.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ecofoundation.niches.base import NicheStrategy


class VoronoiTilingStrategy(NicheStrategy):
    """Non-overlapping spatial partition.

    Pick one of ``target_spacing`` (in coordinate units) or ``n_seeds_per_group``
    to control niche density. ``target_spacing`` is preferred — it scales with
    tissue size.
    """

    name = "tiling"

    def __init__(
        self,
        *,
        target_spacing: float | None = None,
        n_seeds_per_group: int | None = None,
        min_cells_per_niche: int = 5,
        max_cells_per_niche: int | None = None,
        random_state: int = 0,
    ):
        super().__init__(
            min_cells_per_niche=min_cells_per_niche,
            max_cells_per_niche=max_cells_per_niche,
            random_state=random_state,
        )
        if target_spacing is None and n_seeds_per_group is None:
            raise ValueError("Provide target_spacing or n_seeds_per_group")
        self.target_spacing = target_spacing
        self.n_seeds_per_group = n_seeds_per_group

    def params(self) -> dict[str, Any]:
        return {
            **super().params(),
            "target_spacing": self.target_spacing,
            "n_seeds_per_group": self.n_seeds_per_group,
        }

    def _assign_group(
        self, coords: np.ndarray
    ) -> tuple[list[np.ndarray], list[int]]:
        n = coords.shape[0]
        if n < self.min_cells_per_niche:
            return [], []

        seeds = _farthest_point_sampling(
            coords,
            target_spacing=self.target_spacing,
            n_seeds=self.n_seeds_per_group,
            seed=self.random_state,
        )
        if len(seeds) == 0:
            return [], []

        # Voronoi assignment: each cell to its nearest seed.
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(coords[seeds])
        _, nearest_seed = nn.kneighbors(coords)
        nearest_seed = nearest_seed.ravel()

        cells_per_niche: list[np.ndarray] = []
        ego_cells: list[int] = []
        for sid, seed_global in enumerate(seeds):
            members = np.flatnonzero(nearest_seed == sid).astype(np.int64)
            if members.size < self.min_cells_per_niche:
                continue
            cells_per_niche.append(np.sort(members))
            ego_cells.append(int(seed_global))
        return cells_per_niche, ego_cells


def _farthest_point_sampling(
    coords: np.ndarray,
    *,
    target_spacing: float | None,
    n_seeds: int | None,
    seed: int = 0,
) -> list[int]:
    """Greedy FPS. Stops at ``n_seeds`` or when min-distance < ``target_spacing``."""
    n = coords.shape[0]
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    seeds: list[int] = [first]

    # Distance from every point to the closest seed so far.
    diffs = coords - coords[first]
    min_d2 = np.sum(diffs * diffs, axis=1)

    max_seeds = n_seeds if n_seeds is not None else n
    target_d2 = (target_spacing ** 2) if target_spacing is not None else 0.0

    while len(seeds) < max_seeds:
        next_idx = int(np.argmax(min_d2))
        next_d2 = float(min_d2[next_idx])
        if target_spacing is not None and next_d2 < target_d2:
            break
        if next_d2 == 0.0:  # all points already coincide with a seed
            break
        seeds.append(next_idx)
        diffs = coords - coords[next_idx]
        d2 = np.sum(diffs * diffs, axis=1)
        np.minimum(min_d2, d2, out=min_d2)

    return seeds
