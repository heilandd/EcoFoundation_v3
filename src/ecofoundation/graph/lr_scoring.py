"""Ligand-receptor scoring on intra-niche edges.

Tier-1 (default): a single aggregated LR-match score per edge.

For an edge (a, b) the score is

    score(a,b) = mean over LR pairs of  ( expr[a, L] * expr[b, R]
                                       + expr[b, L] * expr[a, R] )

The resource defaults to the OmniPath consensus shipped by LIANA. Pairs whose
ligand OR receptor gene is absent from ``adata.var_names`` are silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ecofoundation.config.schemas import LRScoringConfig
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class LRResource:
    """Resolved LR pairs whose genes exist in the AnnData."""

    ligand_gene_idx: np.ndarray  # (n_pairs,) int into adata.var_names
    receptor_gene_idx: np.ndarray  # (n_pairs,) int
    ligand_names: list[str]
    receptor_names: list[str]
    n_pairs_total: int
    n_pairs_kept: int


def load_lr_resource(adata: ad.AnnData, cfg: LRScoringConfig) -> LRResource:
    """Load the LR resource and map names → var-indices into the AnnData."""
    pairs = _load_resource_dataframe(cfg)
    var_lookup = {g: i for i, g in enumerate(adata.var_names)}

    L_idx: list[int] = []
    R_idx: list[int] = []
    L_name: list[str] = []
    R_name: list[str] = []
    for _, row in pairs.iterrows():
        lig = str(row["ligand"])
        rec = str(row["receptor"])
        # Some resources encode complexes as "GENE_A_GENE_B" — split on '_' isn't
        # robust here, so for now we only accept exact-match single genes.
        if lig in var_lookup and rec in var_lookup:
            L_idx.append(var_lookup[lig])
            R_idx.append(var_lookup[rec])
            L_name.append(lig)
            R_name.append(rec)

    if not L_idx:
        _log.warning(
            f"LR resource '{cfg.resource}' produced 0 usable pairs against "
            f"{len(var_lookup)} genes. Edge LR-scores will be 0."
        )

    return LRResource(
        ligand_gene_idx=np.asarray(L_idx, dtype=np.int64),
        receptor_gene_idx=np.asarray(R_idx, dtype=np.int64),
        ligand_names=L_name,
        receptor_names=R_name,
        n_pairs_total=int(len(pairs)),
        n_pairs_kept=len(L_idx),
    )


def _load_resource_dataframe(cfg: LRScoringConfig) -> pd.DataFrame:
    if cfg.resource == "custom":
        if cfg.custom_resource_path is None:
            raise ValueError("custom LR resource requires custom_resource_path")
        df = pd.read_csv(cfg.custom_resource_path)
        if not {"ligand", "receptor"}.issubset(df.columns):
            raise ValueError("custom LR resource must contain 'ligand' and 'receptor' columns")
        return df[["ligand", "receptor"]]

    # OmniPath consensus via LIANA
    import liana as li

    try:
        df = li.resource.select_resource("consensus")
    except Exception as e:  # network/path issues etc.
        _log.warning(f"LIANA consensus resource unavailable ({e}); falling back to empty resource.")
        return pd.DataFrame(columns=["ligand", "receptor"])

    # LIANA returns columns 'ligand', 'receptor' (lower-case) in recent versions.
    cols = {c.lower(): c for c in df.columns}
    if "ligand" not in cols or "receptor" not in cols:
        raise RuntimeError(f"Unexpected LIANA columns: {df.columns.tolist()}")
    df = df[[cols["ligand"], cols["receptor"]]].rename(
        columns={cols["ligand"]: "ligand", cols["receptor"]: "receptor"}
    )
    df = df.drop_duplicates().reset_index(drop=True)
    _log.info(f"Loaded LIANA '{cfg.resource}' resource: {len(df)} LR pairs")
    return df


def score_edges_lr(
    expr_subset: np.ndarray | sp.spmatrix,
    edges: np.ndarray,
    lr: LRResource,
) -> np.ndarray:
    """Tier-1 LR score per undirected edge.

    Parameters
    ----------
    expr_subset
        ``(n_niche_cells, n_genes_full)`` expression matrix for the niche.
    edges
        ``(E, 2)`` int array of local cell indices into ``expr_subset``.
    lr
        Resolved LR resource.

    Returns
    -------
    scores
        ``(E,)`` float32 array. Zero if the resource is empty.
    """
    if edges.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if lr.n_pairs_kept == 0:
        return np.zeros(edges.shape[0], dtype=np.float32)

    # Densify only the ligand- and receptor-gene columns for this niche.
    if sp.issparse(expr_subset):
        L_mat = expr_subset[:, lr.ligand_gene_idx].toarray()
        R_mat = expr_subset[:, lr.receptor_gene_idx].toarray()
    else:
        L_mat = expr_subset[:, lr.ligand_gene_idx]
        R_mat = expr_subset[:, lr.receptor_gene_idx]
    L_mat = np.asarray(L_mat, dtype=np.float32)
    R_mat = np.asarray(R_mat, dtype=np.float32)

    a = edges[:, 0]
    b = edges[:, 1]
    # bidirectional aggregate
    fwd = (L_mat[a] * R_mat[b]).mean(axis=1)
    bwd = (L_mat[b] * R_mat[a]).mean(axis=1)
    return (fwd + bwd).astype(np.float32)
