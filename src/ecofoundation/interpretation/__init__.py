"""Explainability for the supervised GNN.

Three core attribution signals plus four biology-rich extensions:

  - **GNNExplainer** (PyG) — learns soft masks over nodes that preserve the
    model's prediction. Good at localising "which cells matter" for one niche.
  - **Integrated Gradients** (Captum) — gradient-based path attribution to
    node features (genes) and edge features (distance, LR-score).
  - **Aggregation** — per-class summary of gene/edge importance.

Biology-rich extensions (Step 5.5):

  - **Cell-Type Attribution** — per (class × cell-type) mean attribution,
    revealing which cell types drive each class.
  - **LR-Interaction Attribution** — post-hoc decomposition of important
    edges into ligand-receptor pairs annotated by sender/receiver cell type.
  - **Niche Embeddings** — extract the model's pooled embedding per niche
    and UMAP-project to inspect class structure in latent space.
  - **Pathway Enrichment** — per (class × cell-type) gene-set enrichment
    of top-attributed genes (Enrichr / MSigDB Hallmark by default).
"""

from ecofoundation.interpretation.aggregation import (
    ClassAttribution,
    aggregate_explanations,
)
from ecofoundation.interpretation.cell_type_attribution import (
    CellTypeAttribution,
    compute_cell_type_attribution,
)
from ecofoundation.interpretation.embeddings import (
    NicheEmbeddings,
    compute_niche_umap,
    extract_niche_embeddings,
)
from ecofoundation.interpretation.gnn_explainer import NicheExplanation, explain_niche
from ecofoundation.interpretation.integrated_gradients import (
    IGResult,
    ig_attribute_niche,
)
from ecofoundation.interpretation.lr_interaction_attribution import (
    LRInteractionAttribution,
    compute_lr_interaction_attribution,
)
from ecofoundation.interpretation.pathway_enrichment import (
    PathwayEnrichmentResult,
    compute_pathway_enrichment,
)
from ecofoundation.interpretation.wrapper import SingleTargetWrapper

__all__ = [
    "SingleTargetWrapper",
    "NicheExplanation",
    "explain_niche",
    "IGResult",
    "ig_attribute_niche",
    "ClassAttribution",
    "aggregate_explanations",
    "CellTypeAttribution",
    "compute_cell_type_attribution",
    "LRInteractionAttribution",
    "compute_lr_interaction_attribution",
    "NicheEmbeddings",
    "extract_niche_embeddings",
    "compute_niche_umap",
    "PathwayEnrichmentResult",
    "compute_pathway_enrichment",
]
