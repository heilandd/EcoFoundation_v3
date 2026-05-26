"""Graph construction: per-niche PyG ``Data`` objects with rich edge features.

Step 3 of the EcoFoundation pipeline. Inputs are an AnnData and a
:class:`~ecofoundation.niches.base.NicheAssignment`; output is a list of
``torch_geometric.data.Data`` ready for training the supervised GNN (Step 4)
or the unsupervised GNN clusterer (Step 6).
"""

from ecofoundation.graph.construction import (
    GraphConstructionResult,
    build_niche_graphs,
)
from ecofoundation.graph.edge_features import EdgeFeatures, build_edge_features
from ecofoundation.graph.edges import (
    DelaunayCache,
    NicheEdges,
    build_niche_edges,
)
from ecofoundation.graph.lr_scoring import (
    LRResource,
    load_lr_resource,
    score_edges_lr,
)
from ecofoundation.graph.node_features import (
    NodeFeatureResolver,
    build_node_features,
)

__all__ = [
    "GraphConstructionResult",
    "build_niche_graphs",
    "EdgeFeatures",
    "build_edge_features",
    "DelaunayCache",
    "NicheEdges",
    "build_niche_edges",
    "LRResource",
    "load_lr_resource",
    "score_edges_lr",
    "NodeFeatureResolver",
    "build_node_features",
]
