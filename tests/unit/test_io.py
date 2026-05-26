"""IO: AnnData loading + schema validation + run-folder persistence."""

from __future__ import annotations

import json

import pytest

from ecofoundation.config.schemas import DataConfig
from ecofoundation.io.readers import load_anndata, validate_schema
from ecofoundation.io.writers import create_run_folder


def test_load_and_validate_tiny(tiny_run_config):
    adata = load_anndata(tiny_run_config.data)
    assert adata.shape == (200, 50)
    rep = validate_schema(adata, tiny_run_config.data)
    assert rep.n_samples == 2
    assert rep.n_patients == 2
    assert rep.has_spatial
    assert rep.embedding_present


def test_missing_spatial_key_raises(tiny_adata_path, tmp_path):
    # Same path but wrong spatial key
    cfg = DataConfig(path=tiny_adata_path, spatial_key="not_there")
    with pytest.raises(ValueError, match="not_there"):
        load_anndata(cfg)


def test_create_run_folder_writes_manifest(tiny_run_config):
    folder = create_run_folder(tiny_run_config)
    assert folder.root.exists()
    assert folder.config_path.exists()
    assert folder.manifest_path.exists()
    m = json.loads(folder.manifest_path.read_text())
    assert m["run_id"] == folder.run_id
    assert "versions" in m
    assert m["seed"] == tiny_run_config.seed
