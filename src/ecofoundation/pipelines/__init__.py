"""End-to-end pipelines that orchestrate single steps and emit one report per run.

Pipelines are thin: they sequence steps from other modules and feed everything
into a ReportBuilder. No numerical logic lives here.
"""

from ecofoundation.pipelines.clustering import run_clustering_pipeline, run_from_yaml
from ecofoundation.pipelines.explain import (
    ExplainPipelineInputs,
    run_explainability_pipeline,
)
from ecofoundation.pipelines.graphs import run_graph_construction_pipeline
from ecofoundation.pipelines.niches import run_niche_pipeline
from ecofoundation.pipelines.train import run_training_pipeline
from ecofoundation.pipelines.unsup_cluster import run_unsup_clustering_pipeline

__all__ = [
    "run_clustering_pipeline",
    "run_from_yaml",
    "run_niche_pipeline",
    "run_graph_construction_pipeline",
    "run_training_pipeline",
    "run_explainability_pipeline",
    "ExplainPipelineInputs",
    "run_unsup_clustering_pipeline",
]
