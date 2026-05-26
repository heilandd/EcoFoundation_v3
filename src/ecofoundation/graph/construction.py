"""High-level graph construction: NicheAssignment + AnnData -> list of PyG ``Data``.

Pipeline per niche:
  1. Look up the patient → cached Delaunay edges (or kNN topology).
  2. Slice intra-niche edges (in local indices) + edge lengths.
  3. Slice node features (expression / embedding / concat).
  4. Compute tier-1 LR score per edge.
  5. Pack into ``torch_geometric.data.Data`` with metadata attached.

The Delaunay cache is built once per patient and re-used across all niches of
that patient — critical to keep construction sub-minute on Xenium-scale data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import torch
from torch_geometric.data import Data

from ecofoundation.config.schemas import DataConfig, GraphConfig
from ecofoundation.graph.edge_features import build_edge_features
from ecofoundation.graph.edges import DelaunayCache, build_niche_edges
from ecofoundation.graph.lr_scoring import LRResource, load_lr_resource, score_edges_lr
from ecofoundation.graph.node_features import NodeFeatureResolver, build_node_features
from ecofoundation.niches.base import NicheAssignment
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class GraphConstructionResult:
    """Output of :func:`build_niche_graphs`."""

    graphs: list[Data]
    node_feature_names: list[str]
    edge_feature_names: list[str]
    lr_resource: LRResource
    summary: dict[str, Any]


def build_niche_graphs(
    adata: ad.AnnData,
    niches: NicheAssignment,
    data_cfg: DataConfig,
    graph_cfg: GraphConfig,
) -> GraphConstructionResult:
    """Build one PyG ``Data`` object per niche."""
    if niches.n_niches == 0:
        return GraphConstructionResult(
            graphs=[],
            node_feature_names=[],
            edge_feature_names=[],
            lr_resource=load_lr_resource(adata, graph_cfg.lr_scoring),
            summary={"n_graphs": 0},
        )

    coords = np.asarray(adata.obsm[data_cfg.spatial_key])[:, :2]
    patient_col = adata.obs[data_cfg.patient_id_col].astype(str).to_numpy()

    node_resolver = build_node_features(adata, graph_cfg)
    lr = (
        load_lr_resource(adata, graph_cfg.lr_scoring)
        if graph_cfg.lr_scoring.enabled
        else LRResource(
            ligand_gene_idx=np.empty(0, dtype=np.int64),
            receptor_gene_idx=np.empty(0, dtype=np.int64),
            ligand_names=[],
            receptor_names=[],
            n_pairs_total=0,
            n_pairs_kept=0,
        )
    )
    _log.info(f"LR resource: {lr.n_pairs_kept} / {lr.n_pairs_total} usable pairs")

    # Build a Delaunay cache per patient (once).
    delaunay_caches: dict[str, DelaunayCache] = {}

    # For LR scoring we need expression in its FULL gene space (raw layer),
    # independent of any HVG subsetting on node features.
    if graph_cfg.lr_scoring.enabled and lr.n_pairs_kept > 0:
        expr_full = adata.layers.get(graph_cfg.node_expression_layer, adata.X)
    else:
        expr_full = None

    graphs: list[Data] = []
    n_edges_total = 0
    for niche_id in range(niches.n_niches):
        global_cells = niches.cells_per_niche[niche_id]
        patient = str(niches.group_label[niche_id])

        if patient not in delaunay_caches:
            mask = patient_col == patient
            local_idx_for_patient = np.flatnonzero(mask)
            # Build cache in patient-local coords; map global → patient-local.
            patient_coords = coords[local_idx_for_patient]
            cache = DelaunayCache(patient_coords)
            # Stash the mapping so we can translate global indices to patient-local.
            cache.global_to_patient_local = {
                int(g): i for i, g in enumerate(local_idx_for_patient.tolist())
            }
            delaunay_caches[patient] = cache

        cache = delaunay_caches[patient]
        local_indices = np.asarray(
            [cache.global_to_patient_local[int(c)] for c in global_cells.tolist()],
            dtype=np.int64,
        )

        niche_edges = build_niche_edges(
            local_indices,
            coords=cache.coords,
            delaunay_cache=cache,
            topology=graph_cfg.edge_topology,
            knn_k=graph_cfg.edge_knn_k,
        )

        # Node features (sliced from global AnnData, in the niche's local order).
        x = node_resolver.slice(global_cells)
        x_tensor = torch.from_numpy(x)

        # Edge features
        if expr_full is not None and niche_edges.edges.shape[0] > 0:
            niche_expr = expr_full[global_cells, :]
            lr_scores = score_edges_lr(niche_expr, niche_edges.edges, lr)
        else:
            lr_scores = np.zeros(niche_edges.edges.shape[0], dtype=np.float32)

        ef = build_edge_features(
            distances=niche_edges.distances, lr_scores=lr_scores, cfg=graph_cfg
        )

        # PyG expects edge_index of shape (2, E*2) for undirected (both directions).
        if niche_edges.edges.shape[0] == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, ef.n_channels), dtype=torch.float32)
        else:
            both = np.concatenate(
                [niche_edges.edges, niche_edges.edges[:, ::-1]], axis=0
            ).astype(np.int64)
            edge_index = torch.from_numpy(both.T)
            edge_attr = torch.from_numpy(np.concatenate([ef.matrix, ef.matrix], axis=0))

        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
        # Attach per-niche metadata. PyG passes through unknown attrs.
        data.niche_id = niche_id
        data.patient = patient
        data.sample = (
            str(niches.sample_label[niche_id]) if niches.sample_label is not None else None
        )
        data.global_cell_indices = torch.from_numpy(global_cells.astype(np.int64))
        data.num_nodes = x_tensor.shape[0]

        graphs.append(data)
        n_edges_total += niche_edges.edges.shape[0]

    summary = {
        "n_graphs": len(graphs),
        "n_node_features": node_resolver.n_features,
        "n_edge_features": graphs[0].edge_attr.shape[1] if graphs else 0,
        "median_nodes": int(np.median([g.num_nodes for g in graphs])) if graphs else 0,
        "median_edges": int(np.median([g.edge_index.shape[1] // 2 for g in graphs])) if graphs else 0,
        "total_undirected_edges": int(n_edges_total),
        "lr_pairs_used": lr.n_pairs_kept,
        "node_feature_source": graph_cfg.node_feature_source,
        "edge_topology": graph_cfg.edge_topology,
    }
    return GraphConstructionResult(
        graphs=graphs,
        node_feature_names=node_resolver.feature_names,
        edge_feature_names=_channel_names(graph_cfg, lr),
        lr_resource=lr,
        summary=summary,
    )


def _channel_names(cfg: GraphConfig, lr: LRResource) -> list[str]:
    names = []
    if cfg.edge_feature_distance:
        names.append("distance_norm" if cfg.edge_feature_normalize_distance else "distance")
    if cfg.lr_scoring.enabled and lr.n_pairs_kept > 0:
        names.append("lr_score_tier1")
    if not names:
        names.append("constant")
    return names
