"""Radius niche strategy: ego cell + all cells within R µm."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ecofoundation.niches.base import NicheStrategy


class RadiusStrategy(NicheStrategy):
    """Niches defined by a fixed spatial radius around each ego cell."""

    name = "radius"

    def __init__(
        self,
        *,
        radius: float,
        min_cells_per_niche: int = 5,
        max_cells_per_niche: int | None = 500,
        random_state: int = 0,
    ):
        super().__init__(
            min_cells_per_niche=min_cells_per_niche,
            max_cells_per_niche=max_cells_per_niche,
            random_state=random_state,
        )
        if radius <= 0:
            raise ValueError("radius must be > 0")
        self.radius = radius

    def params(self) -> dict[str, Any]:
        return {**super().params(), "radius": self.radius}

    def _assign_group(
        self, coords: np.ndarray
    ) -> tuple[list[np.ndarray], list[int]]:
        n = coords.shape[0]
        if n < self.min_cells_per_niche:
            return [], []

        nn = NearestNeighbors(radius=self.radius)
        nn.fit(coords)
        indices = nn.radius_neighbors(coords, return_distance=False)

        cells_per_niche: list[np.ndarray] = []
        ego_cells: list[int] = []
        for ego in range(n):
            members = indices[ego]
            if members.size < self.min_cells_per_niche:
                continue
            cells_per_niche.append(np.sort(members.astype(np.int64)))
            ego_cells.append(ego)
        return cells_per_niche, ego_cells
