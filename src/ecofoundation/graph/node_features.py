"""Per-niche node features.

User default: normalized expression from ``layers['X_exp']``. Subsetting via
HVG list, custom gene list, or scVI embedding is opt-in via GraphConfig.

The returned object is a thin :class:`NodeFeatureResolver` that the graph
builder uses to slice features per niche without re-densifying the entire
expression matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import scipy.sparse as sp

from ecofoundation.config.schemas import GraphConfig
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class NodeFeatureResolver:
    """Densifies a slice of node features on demand.

    Holds either a dense (n_cells, n_features) matrix or a sparse one and
    materialises only the rows requested per niche.
    """

    matrix: np.ndarray | sp.spmatrix
    feature_names: list[str]
    source: str  # "expression" | "embedding" | "concat"

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]

    def slice(self, cell_indices: np.ndarray) -> np.ndarray:
        """Return ``(len(cell_indices), n_features)`` dense float32 array."""
        sub = self.matrix[cell_indices, :]
        if sp.issparse(sub):
            sub = sub.toarray()
        return np.ascontiguousarray(sub, dtype=np.float32)


def build_node_features(adata: ad.AnnData, cfg: GraphConfig) -> NodeFeatureResolver:
    """Resolve node features per the GraphConfig (no slicing yet, just selection)."""
    src = cfg.node_feature_source
    if src == "expression":
        return _expression_resolver(adata, cfg)
    if src == "embedding":
        return _embedding_resolver(adata, cfg)
    if src == "concat":
        expr = _expression_resolver(adata, cfg)
        emb = _embedding_resolver(adata, cfg)
        if sp.issparse(expr.matrix):
            expr_mat = expr.matrix.toarray().astype(np.float32)
        else:
            expr_mat = np.asarray(expr.matrix, dtype=np.float32)
        emb_mat = np.asarray(emb.matrix, dtype=np.float32)
        return NodeFeatureResolver(
            matrix=np.concatenate([expr_mat, emb_mat], axis=1),
            feature_names=expr.feature_names + emb.feature_names,
            source="concat",
        )
    raise ValueError(f"Unknown node_feature_source: {src!r}")


def _expression_resolver(adata: ad.AnnData, cfg: GraphConfig) -> NodeFeatureResolver:
    layer = cfg.node_expression_layer
    if layer in adata.layers:
        mat = adata.layers[layer]
    else:
        _log.warning(f"layer '{layer}' missing — falling back to adata.X")
        mat = adata.X

    var_names = list(adata.var_names)
    keep_cols, kept_names = _select_genes(adata, cfg, var_names)
    mat = mat[:, keep_cols]

    _log.info(
        f"Node features: source=expression layer='{layer}' "
        f"n_genes={len(kept_names)} sparse={sp.issparse(mat)}"
    )
    return NodeFeatureResolver(matrix=mat, feature_names=kept_names, source="expression")


def _embedding_resolver(adata: ad.AnnData, cfg: GraphConfig) -> NodeFeatureResolver:
    key = cfg.node_embedding_key
    if key not in adata.obsm:
        raise KeyError(f"obsm['{key}'] missing — required when node_feature_source='embedding'.")
    mat = np.asarray(adata.obsm[key], dtype=np.float32)
    names = [f"{key}_{i}" for i in range(mat.shape[1])]
    _log.info(f"Node features: source=embedding key='{key}' dim={mat.shape[1]}")
    return NodeFeatureResolver(matrix=mat, feature_names=names, source="embedding")


def _select_genes(
    adata: ad.AnnData, cfg: GraphConfig, var_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Translate gene_subset config into a column-index array."""
    if cfg.gene_subset == "all":
        return np.arange(len(var_names)), var_names
    if cfg.gene_subset == "custom":
        if not cfg.custom_genes:
            raise ValueError("gene_subset='custom' requires custom_genes")
        idx = [var_names.index(g) for g in cfg.custom_genes if g in var_names]
        kept = [var_names[i] for i in idx]
        return np.asarray(idx, dtype=np.int64), kept
    if cfg.gene_subset == "hvg":
        return _hvg_indices(adata, cfg.n_hvg, var_names)
    raise ValueError(f"Unknown gene_subset: {cfg.gene_subset!r}")


def _hvg_indices(adata: ad.AnnData, n_top: int, var_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Pick the top-``n_top`` variable genes by mean-variance ratio.

    We do not call ``scanpy.pp.highly_variable_genes`` here because it mutates
    ``adata.var`` in ways the user may not want. Lightweight inline scoring is
    sufficient for downstream subsetting.
    """
    X = adata.layers.get("X_exp", adata.X)
    if sp.issparse(X):
        X = X.tocsc()
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq_mean = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = np.clip(sq_mean - mean * mean, 0, None)
    else:
        mean = X.mean(axis=0)
        var = X.var(axis=0)
    # mean-variance-ratio (Seurat style approximation)
    mvr = var / np.clip(mean, 1e-8, None)
    n_top = min(n_top, len(var_names))
    idx = np.argsort(-mvr)[:n_top]
    kept = [var_names[i] for i in idx]
    return np.sort(idx), [var_names[i] for i in np.sort(idx)]
