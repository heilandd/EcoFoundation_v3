"""Niche construction.

Strategies (kNN, radius, Delaunay-default, tiling) all share the
:class:`NicheStrategy` interface and produce :class:`NicheAssignment`.

Invariants enforced at this layer:
  - niches are patient-scoped (no cross-patient cells)
  - niches respect ``max_overlap_fraction`` (default 0.2)
  - niches with fewer than ``min_cells_per_niche`` cells are dropped
"""

from ecofoundation.niches.assembly import assign_niches
from ecofoundation.niches.base import NicheAssignment, NicheStrategy
from ecofoundation.niches.characterization import NicheStats, compute_niche_stats
from ecofoundation.niches.cluster_biology import (
    ClusterBiology,
    compute_cluster_biology,
    compute_cluster_lr_interactions,
    compute_cluster_pathway_enrichment,
    pick_example_niches_per_cluster,
)
from ecofoundation.niches.cluster_characterization import (
    NicheClusterStats,
    characterize_niche_clusters,
)
from ecofoundation.niches.overlap import OverlapFilterResult, enforce_overlap_limit
from ecofoundation.niches.pruning import (
    apply_edge_pruning,
    prune_edges_absolute,
    prune_edges_quantile,
)
from ecofoundation.niches.strategies import (
    DelaunayKHopStrategy,
    KNNStrategy,
    RadiusStrategy,
    VoronoiTilingStrategy,
)

__all__ = [
    "NicheAssignment",
    "NicheStats",
    "NicheClusterStats",
    "NicheStrategy",
    "assign_niches",
    "compute_niche_stats",
    "characterize_niche_clusters",
    "ClusterBiology",
    "compute_cluster_biology",
    "compute_cluster_pathway_enrichment",
    "compute_cluster_lr_interactions",
    "pick_example_niches_per_cluster",
    "DelaunayKHopStrategy",
    "KNNStrategy",
    "RadiusStrategy",
    "VoronoiTilingStrategy",
    "OverlapFilterResult",
    "enforce_overlap_limit",
    "apply_edge_pruning",
    "prune_edges_absolute",
    "prune_edges_quantile",
]
