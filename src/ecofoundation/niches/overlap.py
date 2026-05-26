"""Overlap controller.

Enforces a maximum pairwise Jaccard overlap between niches via greedy filtering.
The greedy policy:

  1. Sort niches by size (largest first — biggest information content).
  2. Walk the list; keep each niche only if its Jaccard with every previously
     kept niche is ``≤ max_overlap``.
  3. Skipped niches are reported.

Tie-breaking on size ties uses ``random_state`` for determinism.

For ``max_overlap = 0.0`` this degenerates to a disjoint partition (largest
niches first claim cells exclusively).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecofoundation.niches.base import NicheAssignment
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class OverlapFilterResult:
    """Bookkeeping of which niches were dropped."""

    kept_indices: np.ndarray
    dropped_indices: np.ndarray
    max_overlap_fraction: float


def enforce_overlap_limit(
    niches: NicheAssignment, max_overlap_fraction: float
) -> tuple[NicheAssignment, OverlapFilterResult]:
    """Greedily drop niches until pairwise Jaccard ≤ ``max_overlap_fraction``.

    Niches are compared **only within the same group label** (patient) —
    cross-group overlap is by construction zero in EcoFoundation.
    """
    if not (0.0 <= max_overlap_fraction <= 1.0):
        raise ValueError("max_overlap_fraction must be in [0,1]")
    if niches.n_niches == 0:
        return niches, OverlapFilterResult(
            kept_indices=np.empty(0, dtype=np.int64),
            dropped_indices=np.empty(0, dtype=np.int64),
            max_overlap_fraction=max_overlap_fraction,
        )

    sizes = niches.sizes()
    # Sort niches by size desc; stable to honour input order on ties.
    order = np.argsort(-sizes, kind="stable")

    # Materialize sets once.
    cell_sets = [set(niches.cells_per_niche[i].tolist()) for i in range(niches.n_niches)]
    groups = niches.group_label

    # Inverted index: cell_id -> set of kept niches containing it.
    # Two niches can only overlap if they share a cell, so we look up overlap
    # candidates via this index in O(|cells in niche|) instead of O(n_kept).
    cell_to_kept: dict[int, set[int]] = {}

    kept: list[int] = []
    dropped: list[int] = []

    for niche_id in order:
        candidate_set = cell_sets[niche_id]
        g = groups[niche_id]

        # Find already-kept niches that share at least one cell with this one.
        sharing_kept: dict[int, int] = {}  # prev_id -> #shared
        for c in candidate_set:
            for prev_id in cell_to_kept.get(c, ()):
                if groups[prev_id] != g:
                    continue
                sharing_kept[prev_id] = sharing_kept.get(prev_id, 0) + 1

        keep_this = True
        for prev_id, inter in sharing_kept.items():
            prev_set = cell_sets[prev_id]
            union = len(candidate_set) + len(prev_set) - inter
            if union and (inter / union) > max_overlap_fraction:
                keep_this = False
                break

        if keep_this:
            nid_int = int(niche_id)
            kept.append(nid_int)
            for c in candidate_set:
                cell_to_kept.setdefault(c, set()).add(nid_int)
        else:
            dropped.append(int(niche_id))

    kept_arr = np.asarray(sorted(kept), dtype=np.int64)
    dropped_arr = np.asarray(sorted(dropped), dtype=np.int64)

    _log.info(
        f"Overlap filter @ {max_overlap_fraction:.2f}: kept {len(kept_arr)}/{niches.n_niches} niches"
    )

    filtered = NicheAssignment(
        cells_per_niche=[niches.cells_per_niche[i] for i in kept_arr],
        ego_cell=niches.ego_cell[kept_arr],
        group_label=niches.group_label[kept_arr],
        sample_label=(
            niches.sample_label[kept_arr] if niches.sample_label is not None else None
        ),
        centroid=niches.centroid[kept_arr] if len(niches.centroid) else niches.centroid,
        strategy_name=niches.strategy_name,
        params={**niches.params, "max_overlap_fraction": max_overlap_fraction},
        n_source_cells=niches.n_source_cells,
    )
    result = OverlapFilterResult(
        kept_indices=kept_arr,
        dropped_indices=dropped_arr,
        max_overlap_fraction=max_overlap_fraction,
    )
    return filtered, result
