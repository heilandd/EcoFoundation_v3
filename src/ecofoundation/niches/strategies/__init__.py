"""Niche-construction strategies.

All four strategies share the :class:`~ecofoundation.niches.base.NicheStrategy`
interface and can be selected via ``NicheConfig.strategy``.
"""

from ecofoundation.niches.strategies.delaunay import DelaunayKHopStrategy
from ecofoundation.niches.strategies.knn import KNNStrategy
from ecofoundation.niches.strategies.radius import RadiusStrategy
from ecofoundation.niches.strategies.tiling import VoronoiTilingStrategy

__all__ = [
    "DelaunayKHopStrategy",
    "KNNStrategy",
    "RadiusStrategy",
    "VoronoiTilingStrategy",
]
