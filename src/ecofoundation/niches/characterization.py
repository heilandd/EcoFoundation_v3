"""Standardised niche-characterisation statistics.

For each niche, compute:

  - ``size`` — number of cells
  - ``center_celltype`` — annotation of the ego cell (if available)
  - ``cell_type_composition`` — fraction of each cell type
  - ``shannon_entropy`` — diversity of cell-type composition
  - ``center_purity`` — fraction of niche cells matching the center type
  - ``mean_nn_distance`` — mean nearest-neighbour distance inside the niche
    (proxy for local cellular density: small distances → dense tissue)
  - ``median_pairwise_distance`` — median of all intra-niche cell-cell distances
  - ``radius`` — max distance from centroid to any niche member (niche extent)
  - ``n_unique_celltypes`` — distinct cell types represented

Aggregated outputs (across all niches):

  - ``co_occurrence`` — (center_type × neighbor_type) frequency matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ecofoundation.config.schemas import DataConfig
from ecofoundation.niches.base import NicheAssignment
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class NicheStats:
    """Per-niche and aggregated statistics."""

    per_niche: pd.DataFrame  # one row per niche
    co_occurrence: pd.DataFrame  # center × neighbor cell type → count
    cell_types: list[str]

    def summary(self) -> dict:
        d = self.per_niche
        out: dict = {
            "n_niches": int(len(d)),
            "median_size": int(d["size"].median()) if len(d) else 0,
            "median_shannon_entropy": float(d["shannon_entropy"].median()) if len(d) else 0.0,
            "median_center_purity": float(d["center_purity"].median()) if len(d) else 0.0,
            "median_mean_nn_distance": float(d["mean_nn_distance"].median()) if len(d) else 0.0,
            "median_radius": float(d["radius"].median()) if len(d) else 0.0,
            "median_n_unique_celltypes": float(d["n_unique_celltypes"].median()) if len(d) else 0.0,
            "n_cell_types": len(self.cell_types),
        }
        return out


def compute_niche_stats(
    adata: ad.AnnData,
    niches: NicheAssignment,
    data_cfg: DataConfig,
) -> NicheStats:
    """Compute per-niche characterisation statistics."""
    if niches.n_niches == 0:
        return NicheStats(
            per_niche=pd.DataFrame(),
            co_occurrence=pd.DataFrame(),
            cell_types=[],
        )

    coords = np.asarray(adata.obsm[data_cfg.spatial_key])[:, :2]

    ct_col = data_cfg.celltype_col
    if ct_col and ct_col in adata.obs.columns:
        celltype = adata.obs[ct_col].astype(str).to_numpy()
        cell_types = sorted(set(celltype.tolist()))
    else:
        celltype = np.array(["unknown"] * adata.shape[0])
        cell_types = ["unknown"]

    ct_to_idx = {c: i for i, c in enumerate(cell_types)}
    co_occ = np.zeros((len(cell_types), len(cell_types)), dtype=np.int64)

    rows = []
    for nid in range(niches.n_niches):
        cells = niches.cells_per_niche[nid]
        n = len(cells)
        ego = int(niches.ego_cell[nid])

        # Center cell type
        center_ct = celltype[ego]
        center_idx = ct_to_idx[center_ct]

        # Composition
        types, counts = np.unique(celltype[cells], return_counts=True)
        comp = counts / n
        ct_present = sorted(types.tolist())
        n_unique = len(ct_present)

        # Entropy (natural log)
        p = comp[comp > 0]
        entropy = float(-(p * np.log(p)).sum())

        # Purity
        purity = float(counts[types == center_ct].sum() / n) if center_ct in types else 0.0

        # Distance statistics
        niche_coords = coords[cells]
        if n >= 2:
            # mean NN distance: per cell, distance to its closest *other* niche cell
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(niche_coords)
            dists, _ = nn.kneighbors(niche_coords)
            mean_nn = float(dists[:, 1].mean())  # column 0 is self
            # pairwise median (sample to keep it fast for large niches)
            if n > 100:
                rng = np.random.default_rng(nid)
                pick = rng.choice(n, size=100, replace=False)
                sub = niche_coords[pick]
            else:
                sub = niche_coords
            diffs = sub[:, None, :] - sub[None, :, :]
            pairwise = np.sqrt((diffs * diffs).sum(axis=2))
            iu = np.triu_indices(len(sub), k=1)
            median_pw = float(np.median(pairwise[iu]))
            # radius from centroid
            centroid = niche_coords.mean(axis=0)
            radius = float(np.linalg.norm(niche_coords - centroid, axis=1).max())
        else:
            mean_nn = 0.0
            median_pw = 0.0
            radius = 0.0

        # Co-occurrence: center → each neighbor type weighted by its count
        for t, c in zip(types, counts, strict=True):
            co_occ[center_idx, ct_to_idx[t]] += int(c)

        rows.append(
            {
                "niche_id": nid,
                "patient": str(niches.group_label[nid]),
                "sample": (
                    str(niches.sample_label[nid]) if niches.sample_label is not None else None
                ),
                "size": n,
                "center_celltype": center_ct,
                "n_unique_celltypes": n_unique,
                "shannon_entropy": entropy,
                "center_purity": purity,
                "mean_nn_distance": mean_nn,
                "median_pairwise_distance": median_pw,
                "radius": radius,
            }
        )

    per_niche = pd.DataFrame(rows)
    co_df = pd.DataFrame(co_occ, index=cell_types, columns=cell_types)
    _log.info(
        f"Niche characterisation: {len(per_niche)} niches, "
        f"{len(cell_types)} cell types, median entropy={per_niche['shannon_entropy'].median():.2f}"
    )
    return NicheStats(per_niche=per_niche, co_occurrence=co_df, cell_types=cell_types)
