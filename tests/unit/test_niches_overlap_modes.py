"""Overlap-filter mode: enabled vs disabled."""

from __future__ import annotations

from ecofoundation.config.schemas import NicheConfig
from ecofoundation.niches.assembly import assign_niches


def test_overlap_filter_disabled_keeps_all(tiny_run_config, tiny_adata):
    """Unsupervised mode: every ego cell yields a kept niche."""
    cfg = NicheConfig(
        strategy="delaunay",
        k_hop=2,
        min_cells_per_niche=3,
        overlap_filter_enabled=False,
    )
    niches, info = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    assert info.dropped_indices.size == 0
    assert info.kept_indices.size == niches.n_niches
    # Many overlapping niches expected; we just check we kept more than the filtered run.

    # Compare against the filtered version with overlap=0.2
    cfg_filt = NicheConfig(
        strategy="delaunay",
        k_hop=2,
        min_cells_per_niche=3,
        overlap_filter_enabled=True,
        max_overlap_fraction=0.2,
    )
    niches_filt, _ = assign_niches(tiny_adata, tiny_run_config.data, cfg_filt)
    assert niches.n_niches >= niches_filt.n_niches, (
        "Unfiltered mode must keep at least as many niches as filtered mode"
    )


def test_overlap_filter_enabled_default(tiny_run_config, tiny_adata):
    cfg = NicheConfig(strategy="delaunay", k_hop=2, min_cells_per_niche=3)
    assert cfg.overlap_filter_enabled is True  # default
    niches, info = assign_niches(tiny_adata, tiny_run_config.data, cfg)
    # Cap is enforced.
    for i in range(niches.n_niches):
        for j in range(i + 1, niches.n_niches):
            if niches.group_label[i] == niches.group_label[j]:
                assert niches.jaccard(i, j) <= cfg.max_overlap_fraction + 1e-9
