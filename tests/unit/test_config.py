"""Config schema and YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ecofoundation.config.loader import dump_config, load_config
from ecofoundation.config.schemas import DataConfig, NicheConfig, RunConfig


def _minimal_cfg_dict(data_path: Path) -> dict:
    return {
        "run_name": "x",
        "data": {"path": str(data_path)},
    }


def test_minimal_config_loads(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    data_path = tmp_path / "fake.h5ad"
    data_path.touch()
    cfg_path.write_text(yaml.safe_dump(_minimal_cfg_dict(data_path)))
    cfg = load_config(cfg_path)
    assert isinstance(cfg, RunConfig)
    assert cfg.run_name == "x"
    assert cfg.data.sample_id_col == "samples"  # default
    assert cfg.niches.k_hop == 3  # default
    assert cfg.niches.max_overlap_fraction == 0.2  # default


def test_unknown_field_rejected(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    data_path = tmp_path / "fake.h5ad"
    data_path.touch()
    bad = _minimal_cfg_dict(data_path)
    bad["not_a_field"] = 1
    cfg_path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_niche_overlap_bounds():
    with pytest.raises(ValidationError):
        NicheConfig(max_overlap_fraction=1.5)
    with pytest.raises(ValidationError):
        NicheConfig(max_overlap_fraction=-0.1)


def test_round_trip_yaml(tmp_path):
    data_path = tmp_path / "fake.h5ad"
    data_path.touch()
    cfg = RunConfig(data=DataConfig(path=data_path))
    out = tmp_path / "out.yaml"
    dump_config(cfg, out)
    cfg2 = load_config(out)
    assert cfg2.model_dump(mode="json") == cfg.model_dump(mode="json")
