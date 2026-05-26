"""Matplotlib figure builders for the HTML report."""

from ecofoundation.reporting.plots.cluster_biology import (
    cluster_lr_heatmap,
    cluster_pathway_dotplot,
    example_niche_celltypes_figure,
)
from ecofoundation.reporting.plots.composition import composition_bar
from ecofoundation.reporting.plots.explanation import (
    edge_channel_importance_bar,
    niche_explanation_figure,
    top_gene_importance_bar,
)
from ecofoundation.reporting.plots.explanation_celltypes import (
    cell_type_importance_bar,
    cell_type_importance_heatmap,
)
from ecofoundation.reporting.plots.explanation_embedding import (
    niche_embedding_umap_figure,
)
from ecofoundation.reporting.plots.explanation_lr import (
    lr_celltype_pair_heatmap,
    top_lr_interactions_table,
)
from ecofoundation.reporting.plots.explanation_pathways import pathway_dotplot
from ecofoundation.reporting.plots.graphs import (
    edge_feature_distributions,
    graph_size_distributions,
    niche_graph_figure,
)
from ecofoundation.reporting.plots.markers import marker_dotplot, top_markers_heatmap
from ecofoundation.reporting.plots.niche_characterization import (
    center_purity_by_celltype,
    co_occurrence_heatmap,
    heterogeneity_histogram,
    n_unique_celltypes_histogram,
    niche_density_histogram,
    size_vs_density_scatter,
)
from ecofoundation.reporting.plots.niches import (
    niche_centroids_spatial,
    niche_size_distribution,
    niches_per_group_bar,
)
from ecofoundation.reporting.plots.qc import qc_distributions_figure
from ecofoundation.reporting.plots.spatial import spatial_figure
from ecofoundation.reporting.plots.training import (
    confusion_matrix_figure,
    loss_curve_figure,
    metric_curve_figure,
    per_fold_metric_bar,
    roc_curves_figure,
)
from ecofoundation.reporting.plots.umap import umap_figure
from ecofoundation.reporting.plots.unsup_clustering import (
    niche_cluster_composition_bar,
    niche_cluster_embedding_umap,
    niche_cluster_marker_heatmap,
    niche_cluster_spatial,
    unsup_training_loss,
)

__all__ = [
    "qc_distributions_figure",
    "umap_figure",
    "spatial_figure",
    "marker_dotplot",
    "top_markers_heatmap",
    "composition_bar",
    "niche_size_distribution",
    "niches_per_group_bar",
    "niche_centroids_spatial",
    "niche_graph_figure",
    "edge_feature_distributions",
    "graph_size_distributions",
    "co_occurrence_heatmap",
    "niche_density_histogram",
    "heterogeneity_histogram",
    "center_purity_by_celltype",
    "size_vs_density_scatter",
    "n_unique_celltypes_histogram",
    "loss_curve_figure",
    "metric_curve_figure",
    "confusion_matrix_figure",
    "roc_curves_figure",
    "per_fold_metric_bar",
    "niche_explanation_figure",
    "top_gene_importance_bar",
    "edge_channel_importance_bar",
    "cell_type_importance_heatmap",
    "cell_type_importance_bar",
    "lr_celltype_pair_heatmap",
    "top_lr_interactions_table",
    "niche_embedding_umap_figure",
    "pathway_dotplot",
    "unsup_training_loss",
    "niche_cluster_embedding_umap",
    "niche_cluster_spatial",
    "niche_cluster_composition_bar",
    "niche_cluster_marker_heatmap",
    "cluster_pathway_dotplot",
    "cluster_lr_heatmap",
    "example_niche_celltypes_figure",
]
