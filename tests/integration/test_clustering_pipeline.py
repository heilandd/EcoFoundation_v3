"""Integration test: full clustering pipeline on the tiny fixture dataset."""

from __future__ import annotations

import json

import pytest

from ecofoundation.pipelines.clustering import run_clustering_pipeline


@pytest.mark.slow
def test_clustering_pipeline_end_to_end(tiny_run_config):
    """The pipeline runs to completion and produces a report + manifest.

    The fixture has only 200 cells; we use a very low Leiden resolution so the
    test stays fast.
    """
    tiny_run_config.leiden.resolution = 0.3
    tiny_run_config.leiden.n_neighbors = 10
    # Tiny dataset has 50 generic genes; rank_genes_groups still works
    tiny_run_config.markers.n_top = 5

    folder = run_clustering_pipeline(tiny_run_config)

    # Run folder artifacts
    assert folder.root.exists()
    assert folder.report_path.exists()
    assert folder.config_path.exists()
    assert folder.manifest_path.exists()
    assert (folder.artifacts_dir / "marker_genes.csv").exists()

    # Report content sanity
    html = folder.report_path.read_text()
    assert "<html" in html.lower()
    assert "Leiden" in html
    assert "Spatial" in html
    assert "Marker" in html

    # Manifest sanity
    m = json.loads(folder.manifest_path.read_text())
    assert m["seed"] == tiny_run_config.seed
    assert "torch" in m["versions"]
