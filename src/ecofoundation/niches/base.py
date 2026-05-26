"""Niche datamodel + strategy interface.

A **niche** is a set of cells that lie close to each other in tissue space.
EcoFoundation supports four construction strategies (Delaunay k-hop, kNN,
radius, tiling) — all conform to the same :class:`NicheStrategy` interface
and produce a :class:`NicheAssignment`.

Key invariants of every NicheAssignment:
  - Each niche is fully contained in one patient (and typically one sample).
  - Pairwise Jaccard overlap is capped by ``max_overlap_fraction``
    (default 0.2; enforced by the overlap controller, not the strategies).
  - Niches with fewer than ``min_cells_per_niche`` cells are dropped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NicheAssignment:
    """A collection of niches produced by a :class:`NicheStrategy`.

    Parallel arrays/lists indexed by niche id (0..n_niches-1).

    Attributes
    ----------
    cells_per_niche
        For each niche, the global cell indices (into the source AnnData) that
        belong to it. Length ``n_niches``.
    ego_cell
        Index of the "center" cell of each niche, where the concept applies
        (Delaunay/kNN/radius). For tiling-style niches this is the cell closest
        to the centroid. Length ``n_niches``.
    group_label
        Patient (or other group) label per niche. Length ``n_niches``.
    sample_label
        Sample id per niche, if available. Length ``n_niches`` or None.
    centroid
        2D centroid of each niche in spatial coordinates. Shape ``(n_niches, 2)``.
    strategy_name
        Which strategy produced this assignment (e.g. ``"delaunay"``).
    params
        Serializable dict of the strategy params actually used.
    n_source_cells
        Total cell count in the source AnnData (for sanity checks).
    """

    cells_per_niche: list[np.ndarray]
    ego_cell: np.ndarray
    group_label: np.ndarray
    sample_label: np.ndarray | None
    centroid: np.ndarray
    strategy_name: str
    params: dict[str, Any] = field(default_factory=dict)
    n_source_cells: int = 0

    # ----- size helpers ------------------------------------------------------

    @property
    def n_niches(self) -> int:
        return len(self.cells_per_niche)

    def sizes(self) -> np.ndarray:
        """Cells-per-niche, shape ``(n_niches,)``."""
        return np.fromiter((len(c) for c in self.cells_per_niche), dtype=np.int64)

    # ----- overlap -----------------------------------------------------------

    def jaccard(self, i: int, j: int) -> float:
        """Pairwise Jaccard overlap of niches i and j (1.0 = identical)."""
        a = self.cells_per_niche[i]
        b = self.cells_per_niche[j]
        if len(a) == 0 or len(b) == 0:
            return 0.0
        set_a = set(a.tolist())
        set_b = set(b.tolist())
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 0.0

    def cell_to_niches(self) -> dict[int, list[int]]:
        """Reverse mapping: cell index → niches it belongs to."""
        rev: dict[int, list[int]] = {}
        for niche_id, cells in enumerate(self.cells_per_niche):
            for c in cells.tolist():
                rev.setdefault(c, []).append(niche_id)
        return rev

    # ----- summary -----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Lightweight dict for the HTML report."""
        sizes = self.sizes()
        groups, group_counts = np.unique(self.group_label, return_counts=True)
        return {
            "strategy": self.strategy_name,
            "n_niches": self.n_niches,
            "n_source_cells": self.n_source_cells,
            "median_cells_per_niche": int(np.median(sizes)) if sizes.size else 0,
            "min_cells_per_niche": int(sizes.min()) if sizes.size else 0,
            "max_cells_per_niche": int(sizes.max()) if sizes.size else 0,
            "n_groups": int(groups.size),
            "niches_per_group": dict(zip(map(str, groups), map(int, group_counts), strict=True)),
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class NicheStrategy(ABC):
    """Abstract base for niche-construction strategies.

    Strategies operate on spatial coordinates + group labels (patient_id).
    They MUST NOT create niches that span multiple groups.

    Subclasses implement :meth:`_assign_group`, which produces ego-centric or
    tile-style niches for a single group's cell coordinates. The base class
    handles the per-group dispatch and re-indexing back to global indices.
    """

    name: str = "abstract"

    def __init__(
        self,
        *,
        min_cells_per_niche: int = 5,
        max_cells_per_niche: int | None = None,
        random_state: int = 0,
    ):
        self.min_cells_per_niche = min_cells_per_niche
        self.max_cells_per_niche = max_cells_per_niche
        self.random_state = random_state

    # ----- core API ----------------------------------------------------------

    def assign(
        self,
        coords: np.ndarray,
        group_labels: np.ndarray,
        *,
        sample_labels: np.ndarray | None = None,
    ) -> NicheAssignment:
        """Build niches across the dataset, dispatching one group at a time.

        Parameters
        ----------
        coords
            2D coordinates of every cell. Shape ``(n_cells, 2)``.
        group_labels
            Per-cell group label (typically patient id). Shape ``(n_cells,)``.
        sample_labels
            Optional per-cell sample id. Shape ``(n_cells,)``.
        """
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(f"coords must be (n,2+); got shape {coords.shape}")
        if len(group_labels) != coords.shape[0]:
            raise ValueError("group_labels length must match coords")
        if sample_labels is not None and len(sample_labels) != coords.shape[0]:
            raise ValueError("sample_labels length must match coords")

        groups = self._unique_preserve_order(group_labels)
        all_cells_per_niche: list[np.ndarray] = []
        all_ego: list[int] = []
        all_group: list[Any] = []
        all_sample: list[Any] = []
        all_centroid: list[np.ndarray] = []

        for g in groups:
            mask = group_labels == g
            local_indices = np.flatnonzero(mask)
            if local_indices.size < self.min_cells_per_niche:
                continue
            local_coords = coords[local_indices, :2]

            cells_per_niche, ego_local = self._assign_group(local_coords)

            for niche_cells_local, ego_idx_local in zip(cells_per_niche, ego_local, strict=True):
                if niche_cells_local.size < self.min_cells_per_niche:
                    continue
                if self.max_cells_per_niche is not None and niche_cells_local.size > self.max_cells_per_niche:
                    # cap by closest-to-ego (or random if no ego concept)
                    niche_cells_local = self._cap_niche(
                        niche_cells_local, ego_idx_local, local_coords
                    )

                global_cells = local_indices[niche_cells_local]
                global_ego = int(local_indices[ego_idx_local])
                all_cells_per_niche.append(global_cells)
                all_ego.append(global_ego)
                all_group.append(g)
                if sample_labels is not None:
                    # majority sample within the niche (usually unique)
                    samples_in_niche = sample_labels[global_cells]
                    vals, counts = np.unique(samples_in_niche, return_counts=True)
                    all_sample.append(vals[counts.argmax()])
                all_centroid.append(coords[global_cells, :2].mean(axis=0))

        return NicheAssignment(
            cells_per_niche=all_cells_per_niche,
            ego_cell=np.asarray(all_ego, dtype=np.int64),
            group_label=np.asarray(all_group, dtype=object),
            sample_label=(np.asarray(all_sample, dtype=object) if sample_labels is not None else None),
            centroid=np.asarray(all_centroid) if all_centroid else np.empty((0, 2)),
            strategy_name=self.name,
            params=self.params(),
            n_source_cells=int(coords.shape[0]),
        )

    @abstractmethod
    def _assign_group(
        self, coords: np.ndarray
    ) -> tuple[list[np.ndarray], list[int]]:
        """Build niches for a single group.

        Parameters
        ----------
        coords
            Coordinates of the cells in this group. Shape ``(n_group, 2)``.

        Returns
        -------
        cells_per_niche
            For each niche, local cell indices (relative to ``coords``).
        ego
            Local index of the ego cell of each niche.
        """
        raise NotImplementedError

    def params(self) -> dict[str, Any]:
        """Strategy parameters for the manifest."""
        return {
            "min_cells_per_niche": self.min_cells_per_niche,
            "max_cells_per_niche": self.max_cells_per_niche,
            "random_state": self.random_state,
        }

    # ----- helpers -----------------------------------------------------------

    def _cap_niche(
        self, niche_cells: np.ndarray, ego_idx: int, coords: np.ndarray
    ) -> np.ndarray:
        """Cap a niche to ``max_cells_per_niche`` by keeping cells closest to ego."""
        assert self.max_cells_per_niche is not None
        ego_xy = coords[ego_idx]
        d2 = np.sum((coords[niche_cells] - ego_xy) ** 2, axis=1)
        keep_local = np.argsort(d2)[: self.max_cells_per_niche]
        return niche_cells[keep_local]

    @staticmethod
    def _unique_preserve_order(arr: np.ndarray) -> list[Any]:
        seen: set[Any] = set()
        out: list[Any] = []
        for v in arr.tolist():
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out
