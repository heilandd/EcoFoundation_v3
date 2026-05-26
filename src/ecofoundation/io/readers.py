"""AnnData loading with schema validation.

Always uses ``backed='r'`` first for big files so we can inspect the structure
without paying the full RAM cost. Callers can convert to in-memory via
``adata.to_memory()`` once they know what they need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad

from ecofoundation.config.schemas import DataConfig
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class SchemaReport:
    """Summary of what was found vs expected in an AnnData."""

    n_cells: int
    n_genes: int
    has_spatial: bool
    sample_id_present: bool
    patient_id_present: bool
    condition_present: bool
    celltype_present: bool
    counts_layer_present: bool
    normalized_layer_present: bool
    embedding_present: bool
    n_samples: int | None
    n_patients: int | None

    def to_dict(self) -> dict:
        return {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "has_spatial": self.has_spatial,
            "sample_id_present": self.sample_id_present,
            "patient_id_present": self.patient_id_present,
            "condition_present": self.condition_present,
            "celltype_present": self.celltype_present,
            "counts_layer_present": self.counts_layer_present,
            "normalized_layer_present": self.normalized_layer_present,
            "embedding_present": self.embedding_present,
            "n_samples": self.n_samples,
            "n_patients": self.n_patients,
        }


def load_anndata(cfg: DataConfig, *, backed: bool = False) -> ad.AnnData:
    """Load an AnnData from disk and validate that the expected columns are present.

    Parameters
    ----------
    cfg
        :class:`DataConfig` with path + column-name mapping.
    backed
        If True, open in backed mode (memory-mapped). Use this for inspection /
        per-sample workflows on Xenium-scale data. Default False (loads to RAM).
    """
    path = Path(cfg.path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")
    _log.info(f"Loading AnnData from {path} (backed={backed})")
    adata = ad.read_h5ad(path, backed="r" if backed else None)
    _log.info(f"Loaded AnnData: {adata.shape[0]} cells x {adata.shape[1]} genes")
    report = validate_schema(adata, cfg)
    _log.info(
        f"Schema OK | samples={report.n_samples} patients={report.n_patients} "
        f"spatial={report.has_spatial}"
    )
    return adata


def validate_schema(adata: ad.AnnData, cfg: DataConfig) -> SchemaReport:
    """Check that columns/layers/keys referenced in ``cfg`` actually exist.

    Raises
    ------
    ValueError
        If a required field (sample_id_col, spatial_key) is missing.
        Optional fields (condition_col, embedding_key) only log a warning.
    """
    missing_required: list[str] = []

    if cfg.sample_id_col not in adata.obs.columns:
        missing_required.append(f"obs['{cfg.sample_id_col}'] (sample_id_col)")
    if cfg.spatial_key not in adata.obsm:
        missing_required.append(f"obsm['{cfg.spatial_key}'] (spatial_key)")

    patient_present = (
        cfg.patient_id_col is not None and cfg.patient_id_col in adata.obs.columns
    )
    condition_present = (
        cfg.condition_col is not None and cfg.condition_col in adata.obs.columns
    )
    celltype_present = (
        cfg.celltype_col is not None and cfg.celltype_col in adata.obs.columns
    )

    if cfg.patient_id_col is not None and not patient_present:
        _log.warning(f"obs['{cfg.patient_id_col}'] not found — patient grouping disabled.")

    counts_present = cfg.counts_layer is None or cfg.counts_layer in adata.layers
    normalized_present = cfg.normalized_layer is None or cfg.normalized_layer in adata.layers
    embedding_present = cfg.embedding_key is None or cfg.embedding_key in adata.obsm

    if not counts_present:
        _log.warning(f"layers['{cfg.counts_layer}'] not found — will use adata.X if needed.")
    if not normalized_present:
        _log.warning(f"layers['{cfg.normalized_layer}'] not found — normalization may run.")
    if not embedding_present:
        _log.warning(f"obsm['{cfg.embedding_key}'] not found — will recompute if needed.")

    if missing_required:
        raise ValueError("AnnData missing required fields: " + "; ".join(missing_required))

    n_samples = int(adata.obs[cfg.sample_id_col].nunique()) if cfg.sample_id_col in adata.obs else None
    n_patients = (
        int(adata.obs[cfg.patient_id_col].nunique()) if patient_present else None
    )

    return SchemaReport(
        n_cells=int(adata.shape[0]),
        n_genes=int(adata.shape[1]),
        has_spatial=cfg.spatial_key in adata.obsm,
        sample_id_present=True,
        patient_id_present=patient_present,
        condition_present=condition_present,
        celltype_present=celltype_present,
        counts_layer_present=counts_present,
        normalized_layer_present=normalized_present,
        embedding_present=embedding_present,
        n_samples=n_samples,
        n_patients=n_patients,
    )
