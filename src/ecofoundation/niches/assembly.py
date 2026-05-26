"""High-level niche assignment orchestrator.

Takes an AnnData + NicheConfig and produces a :class:`NicheAssignment`
that respects the patient column from :class:`DataConfig`. Wraps:

  1. strategy selection
  2. patient-aware dispatch (handled inside ``NicheStrategy.assign``)
  3. overlap enforcement
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from ecofoundation.config.schemas import DataConfig, NicheConfig
from ecofoundation.niches.base import NicheAssignment, NicheStrategy
from ecofoundation.niches.overlap import OverlapFilterResult, enforce_overlap_limit
from ecofoundation.niches.strategies.delaunay import DelaunayKHopStrategy
from ecofoundation.niches.strategies.knn import KNNStrategy
from ecofoundation.niches.strategies.radius import RadiusStrategy
from ecofoundation.niches.strategies.tiling import VoronoiTilingStrategy
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


def _build_strategy(cfg: NicheConfig) -> NicheStrategy:
    """Instantiate the strategy named in ``cfg.strategy``."""
    if cfg.strategy == "delaunay":
        return DelaunayKHopStrategy(
            k_hop=cfg.k_hop,
            max_edge_length=cfg.max_edge_length,
            edge_length_quantile_cutoff=cfg.edge_length_quantile_cutoff,
            min_cells_per_niche=cfg.min_cells_per_niche,
            max_cells_per_niche=cfg.max_cells_per_niche,
        )
    if cfg.strategy == "knn":
        return KNNStrategy(
            k=cfg.knn_k,
            max_edge_length=cfg.max_edge_length,
            min_cells_per_niche=cfg.min_cells_per_niche,
            max_cells_per_niche=cfg.max_cells_per_niche,
        )
    if cfg.strategy == "radius":
        if cfg.radius is None:
            raise ValueError("NicheConfig.radius is required for strategy='radius'")
        return RadiusStrategy(
            radius=cfg.radius,
            min_cells_per_niche=cfg.min_cells_per_niche,
            max_cells_per_niche=cfg.max_cells_per_niche,
        )
    if cfg.strategy == "tiling":
        # Default ``target_spacing`` is taken from radius if not set explicitly.
        # We expose this via NicheConfig.radius (reusing the slot to keep the
        # schema compact). Users can override per-pipeline.
        if cfg.radius is None:
            raise ValueError(
                "NicheConfig.radius is required for strategy='tiling' "
                "(re-used as target_spacing)."
            )
        return VoronoiTilingStrategy(
            target_spacing=cfg.radius,
            min_cells_per_niche=cfg.min_cells_per_niche,
            max_cells_per_niche=cfg.max_cells_per_niche,
        )
    raise ValueError(f"Unknown strategy: {cfg.strategy!r}")


def assign_niches(
    adata: ad.AnnData,
    data_cfg: DataConfig,
    niche_cfg: NicheConfig,
) -> tuple[NicheAssignment, OverlapFilterResult]:
    """Build niches across an AnnData, respecting ``data_cfg.patient_id_col``.

    The patient column is *required* (otherwise we'd build one giant niche set
    that mixes patients — never what we want for downstream classification).
    If the user truly wants a single-group dataset, they can set every cell's
    patient_id to the same value.
    """
    if data_cfg.patient_id_col is None or data_cfg.patient_id_col not in adata.obs.columns:
        raise ValueError(
            f"patient_id_col '{data_cfg.patient_id_col}' missing from obs — "
            "niches must be patient-scoped."
        )
    if data_cfg.spatial_key not in adata.obsm:
        raise ValueError(f"obsm['{data_cfg.spatial_key}'] missing")

    coords = np.asarray(adata.obsm[data_cfg.spatial_key])[:, :2]
    groups = adata.obs[data_cfg.patient_id_col].astype(str).to_numpy()
    samples = (
        adata.obs[data_cfg.sample_id_col].astype(str).to_numpy()
        if data_cfg.sample_id_col in adata.obs.columns
        else None
    )

    strategy = _build_strategy(niche_cfg)
    _log.info(
        f"Building niches: strategy={strategy.name} "
        f"params={strategy.params()} | patients={len(set(groups))} cells={len(coords)}"
    )
    raw = strategy.assign(coords, groups, sample_labels=samples)
    _log.info(f"Raw niches: {raw.n_niches}")

    if not niche_cfg.overlap_filter_enabled:
        # Unsupervised mode: keep ALL niches so every cell has a niche representation.
        _log.info("Overlap filter disabled — keeping all niches (unsupervised mode).")
        passthrough = OverlapFilterResult(
            kept_indices=np.arange(raw.n_niches, dtype=np.int64),
            dropped_indices=np.empty(0, dtype=np.int64),
            max_overlap_fraction=1.0,
        )
        return raw, passthrough

    filtered, overlap_info = enforce_overlap_limit(raw, niche_cfg.max_overlap_fraction)
    return filtered, overlap_info
