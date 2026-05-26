"""Write EcoFoundation analysis results back into an AnnData (.h5ad).

Goal: after a pipeline run, the user gets a self-contained ``adata`` with all
measurements attached, ready to load in scanpy / squidpy / R / ... and combine
with downstream tools.

Conventions:

  - ``obs['ecof_niche_id']``       — niche id of the niche this cell is the ego of
  - ``obs['ecof_niche_cluster']``  — categorical niche-cluster label (from Step 6)
  - ``obs['ecof_predicted_label']``— categorical prediction (Step 4) for ego cells
  - ``obsm['ecof_niche_embedding']``— (n_cells, hidden_dim) per-cell niche embedding
  - ``obsm['ecof_niche_umap']``    — (n_cells, 2) UMAP of the niche embedding
  - ``uns['ecof_run']``            — run-level metadata (run_id, config hash, ...)

Cells that are not the ego of any niche (e.g. dropped by min_cells_per_niche)
are filled with ``-1`` / ``NaN`` in the relevant columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ecofoundation.niches.base import NicheAssignment
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class UnsupExportInputs:
    """Inputs to :func:`export_unsup_results_to_anndata`."""

    niches: NicheAssignment
    niche_embeddings: np.ndarray  # (n_niches, hidden_dim)
    niche_umap_2d: np.ndarray | None  # (n_niches, 2) or None
    niche_cluster_labels: np.ndarray  # (n_niches,) cluster ids (int)
    run_id: str
    config_hash: str
    cluster_labels_str: np.ndarray | None = None  # optional human-readable labels


def export_unsup_results_to_anndata(
    adata_source: ad.AnnData,
    inputs: UnsupExportInputs,
    *,
    out_path: Path | str,
    write_compressed: bool = True,
) -> Path:
    """Copy the source AnnData, attach the unsupervised clustering results, persist."""
    adata = adata_source.copy()
    n_cells = adata.shape[0]
    n_hidden = inputs.niche_embeddings.shape[1]

    # ---- per-cell mappings ------------------------------------------------
    cell_cluster_int = np.full(n_cells, -1, dtype=np.int64)
    cell_niche_id = np.full(n_cells, -1, dtype=np.int64)
    emb_per_cell = np.full((n_cells, n_hidden), np.nan, dtype=np.float32)
    umap_per_cell = (
        np.full((n_cells, 2), np.nan, dtype=np.float32)
        if inputs.niche_umap_2d is not None
        else None
    )

    for nid in range(inputs.niches.n_niches):
        ego = int(inputs.niches.ego_cell[nid])
        cell_cluster_int[ego] = int(inputs.niche_cluster_labels[nid])
        cell_niche_id[ego] = nid
        emb_per_cell[ego] = inputs.niche_embeddings[nid]
        if umap_per_cell is not None:
            umap_per_cell[ego] = inputs.niche_umap_2d[nid]

    # Categorical column (string-based, scanpy-friendly).
    if inputs.cluster_labels_str is not None:
        cell_cluster_str = np.array(["unassigned"] * n_cells, dtype=object)
        for nid in range(inputs.niches.n_niches):
            ego = int(inputs.niches.ego_cell[nid])
            cell_cluster_str[ego] = str(inputs.cluster_labels_str[nid])
    else:
        cell_cluster_str = np.where(
            cell_cluster_int >= 0,
            np.char.add("nc_", cell_cluster_int.astype(str)),
            "unassigned",
        )

    adata.obs["ecof_niche_cluster"] = pd.Categorical(cell_cluster_str)
    adata.obs["ecof_niche_id"] = cell_niche_id
    adata.obsm["ecof_niche_embedding"] = emb_per_cell
    if umap_per_cell is not None:
        adata.obsm["ecof_niche_umap"] = umap_per_cell

    # Niche-level frame in .uns for downstream lookup
    niche_df = pd.DataFrame(
        {
            "niche_id": np.arange(inputs.niches.n_niches),
            "ego_cell_idx": inputs.niches.ego_cell,
            "patient": inputs.niches.group_label.astype(str),
            "sample": (
                inputs.niches.sample_label.astype(str)
                if inputs.niches.sample_label is not None
                else np.array([""] * inputs.niches.n_niches)
            ),
            "size": inputs.niches.sizes(),
            "centroid_x": inputs.niches.centroid[:, 0],
            "centroid_y": inputs.niches.centroid[:, 1],
            "niche_cluster": inputs.niche_cluster_labels,
        }
    )
    adata.uns["ecof_niche_table"] = niche_df.to_dict(orient="list")
    adata.uns["ecof_run"] = {
        "run_id": inputs.run_id,
        "config_hash": inputs.config_hash,
        "n_niches": int(inputs.niches.n_niches),
        "niche_strategy": inputs.niches.strategy_name,
        "niche_params": {str(k): str(v) for k, v in inputs.niches.params.items()},
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if write_compressed else None
    adata.write_h5ad(out, compression=compression)
    _log.info(
        f"Exported annotated AnnData to {out} "
        f"({out.stat().st_size / 1024**2:.1f} MB, "
        f"{int((cell_cluster_int >= 0).sum())} cells annotated)"
    )
    return out
