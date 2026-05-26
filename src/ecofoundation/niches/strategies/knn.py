"""kNN niche strategy: each cell + its k nearest neighbours."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ecofoundation.niches.base import NicheStrategy


class KNNStrategy(NicheStrategy):
    """Each ego cell's niche = the ego plus its k nearest spatial neighbours."""

    name = "knn"

    def __init__(
        self,
        *,
        k: int = 15,
        max_edge_length: float | None = None,
        min_cells_per_niche: int = 5,
        max_cells_per_niche: int | None = None,
        random_state: int = 0,
    ):
        super().__init__(
            min_cells_per_niche=min_cells_per_niche,
            max_cells_per_niche=max_cells_per_niche,
            random_state=random_state,
        )
        if k < 2:
            raise ValueError("k must be ≥ 2")
        self.k = k
        self.max_edge_length = max_edge_length

    def params(self) -> dict[str, Any]:
        return {**super().params(), "k": self.k, "max_edge_length": self.max_edge_length}

    def _assign_group(
        self, coords: np.ndarray
    ) -> tuple[list[np.ndarray], list[int]]:
        n = coords.shape[0]
        if n <= self.k:
            return [], []

        nn = NearestNeighbors(n_neighbors=self.k + 1)  # +1 to include self
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

        cells_per_niche: list[np.ndarray] = []
        ego_cells: list[int] = []
        for ego in range(n):
            members = indices[ego]
            dists = distances[ego]
            if self.max_edge_length is not None:
                keep = dists <= self.max_edge_length
                members = members[keep]
            if len(members) < self.min_cells_per_niche:
                continue
            cells_per_niche.append(np.sort(members.astype(np.int64)))
            ego_cells.append(ego)
        return cells_per_niche, ego_cells
