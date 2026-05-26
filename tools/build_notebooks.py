"""Generate Jupyter notebooks under ``examples/notebooks/``.

Run from the repo root::

    .venv/bin/python tools/build_notebooks.py

Each notebook walks through one part of the EcoFoundation pipeline with
explanatory markdown cells, runnable code cells, and per-plot examples.

The notebooks intentionally **import** the high-level pipeline functions for
the heavy lifting and break out individual plot calls for inspection — so
the user can both reproduce the report and modify individual plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nbformat as nbf

OUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH_DEFAULT = "../../configs/unsup_clustering.yaml"  # relative to notebook


def md(*lines: str) -> dict:
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> dict:
    return nbf.v4.new_code_cell("\n".join(lines))


def save(name: str, cells: Iterable) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = list(cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ecofoundation)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    path = OUT_DIR / name
    nbf.write(nb, path)
    return path


# ============================================================================
# 00_overview.ipynb
# ============================================================================


def build_overview() -> Path:
    cells = [
        md(
            "# EcoFoundation — Notebook Overview",
            "",
            "This folder holds step-by-step notebooks that mirror the EcoFoundation",
            "pipeline. Each notebook is self-contained and shows both the high-level",
            "pipeline call and the underlying building blocks, so you can:",
            "",
            "1. Reproduce the report end-to-end.",
            "2. Inspect / re-render individual plots.",
            "3. Tweak parameters and re-run a step in isolation.",
            "",
            "## Recommended order",
            "",
            "| Notebook | Pipeline step | What it covers |",
            "|---|---|---|",
            "| `01_niches_and_characterization.ipynb` | Step 2 / 2.5 | Niche construction (4 strategies) + standardised characterisation |",
            "| `02_graphs_and_lr.ipynb` | Step 3 | Per-niche PyG graphs + ligand-receptor scoring |",
            "| `03_supervised_training_and_explain.ipynb` | Step 4 + 5 / 5.5 | GINE/GAT training + GNNExplainer + IG + cell-type / LR / pathway / embedding attribution |",
            "| `04_unsupervised_clustering.ipynb` | Step 6 + 6.5 + 7 | GAE/DGI + Leiden + adversarial debiasing + cluster biology + AnnData export |",
            "",
            "Each notebook expects the demo dataset at `../scVI_adata_annotated.h5ad`",
            "and a working `.venv` (run `uv sync --extra dev` once from the repo root).",
            "",
            "## Kernel",
            "",
            "Activate the kernel pointing at the project's `.venv`:",
            "",
            "```bash",
            ".venv/bin/python -m ipykernel install --user --name ecofoundation --display-name 'Python 3 (ecofoundation)'",
            "```",
        ),
    ]
    return save("00_overview.ipynb", cells)


# ============================================================================
# 01_niches_and_characterization.ipynb
# ============================================================================


def build_niches_notebook() -> Path:
    cells = [
        md(
            "# 01 — Niche construction & characterisation",
            "",
            "EcoFoundation analyses spatial transcriptomics data through **niches** —",
            "local neighbourhoods of cells in tissue. This notebook covers Step 2 + 2.5:",
            "",
            "1. Load the AnnData",
            "2. Build niches with each of the four strategies (kNN, Delaunay, radius, tiling)",
            "3. Compare niche-size distributions",
            "4. Compute standardised per-niche characterisation",
            "5. Render the report-style plots interactively",
            "",
            "## Why niches?",
            "",
            "Single-cell methods describe each cell in isolation; spatial methods need a",
            "scale at which biological context emerges. A niche bundles a cell with its",
            "immediate microenvironment — the GNN downstream treats each niche as one graph.",
        ),
        code(
            "import os, sys",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib",
            "matplotlib.use('agg')  # in-notebook display still works",
            "",
            "# Ensure the repo root is on the path (works whether kernel is project or system)",
            "REPO = Path.cwd().parent.parent",
            "if str(REPO / 'src') not in sys.path:",
            "    sys.path.insert(0, str(REPO / 'src'))",
            "",
            "from ecofoundation.config.schemas import DataConfig, NicheConfig",
            "from ecofoundation.io.readers import load_anndata, validate_schema",
            "from ecofoundation.niches.assembly import assign_niches",
            "from ecofoundation.niches.characterization import compute_niche_stats",
        ),
        md(
            "## 1. Load the AnnData",
            "",
            "We use the demo dataset shipped at the repo root. `DataConfig` is a",
            "Pydantic schema that holds all column-name mappings — change `path` and",
            "the relevant column names for your own data.",
        ),
        code(
            "data_cfg = DataConfig(",
            "    path=REPO / 'scVI_adata_annotated.h5ad',",
            "    sample_id_col='samples',",
            "    patient_id_col='patient',",
            "    condition_col='sonicated',",
            "    celltype_col='celltype_level_1',",
            "    spatial_key='spatial',",
            "    counts_layer='counts',",
            "    normalized_layer='X_exp',",
            "    embedding_key='X_scVI',",
            ")",
            "adata = load_anndata(data_cfg)",
            "schema = validate_schema(adata, data_cfg)",
            "schema",
        ),
        md(
            "## 2. Niche construction — four strategies",
            "",
            "`NicheConfig.strategy` controls which strategy is used. EcoFoundation",
            "supports:",
            "",
            "- `knn` (default) — fixed-size niches; every ego cell has its 50 nearest spatial neighbours.",
            "- `delaunay` — density-aware k-hop niches on the Delaunay triangulation.",
            "- `radius` — all cells within a µm radius.",
            "- `tiling` — Voronoi partition seeded via Farthest-Point-Sampling (strictly disjoint).",
            "",
            "Always patient-aware: a niche cannot span two patients. The overlap controller",
            "caps pairwise Jaccard overlap (default 0.2) for supervised setups; for the",
            "unsupervised pipeline we disable it so every cell gets its own niche.",
        ),
        code(
            "knn_niches, _ = assign_niches(",
            "    adata, data_cfg,",
            "    NicheConfig(strategy='knn', knn_k=50, min_cells_per_niche=10,",
            "                overlap_filter_enabled=True, max_overlap_fraction=0.2),",
            ")",
            "print('kNN-50 niches:', knn_niches.n_niches,",
            "      '| median size:', int(np.median(knn_niches.sizes())))",
        ),
        code(
            "delaunay_niches, _ = assign_niches(",
            "    adata, data_cfg,",
            "    NicheConfig(strategy='delaunay', k_hop=3,",
            "                edge_length_quantile_cutoff=0.95,",
            "                min_cells_per_niche=8, max_cells_per_niche=300,",
            "                overlap_filter_enabled=True, max_overlap_fraction=0.2),",
            ")",
            "print('Delaunay-3hop niches:', delaunay_niches.n_niches,",
            "      '| size range:', int(delaunay_niches.sizes().min()),",
            "      '..', int(delaunay_niches.sizes().max()))",
        ),
        md(
            "## 3. Niche characterisation",
            "",
            "For every niche we compute:",
            "",
            "- `size`, `radius`, `mean_nn_distance` (cellular density proxy)",
            "- `shannon_entropy` of cell-type composition",
            "- `center_purity` (fraction of niche cells matching the ego's cell type)",
            "- `n_unique_celltypes`",
            "- `co_occurrence` matrix (center cell-type × neighbour cell-type)",
        ),
        code(
            "stats = compute_niche_stats(adata, knn_niches, data_cfg)",
            "stats.summary()",
        ),
        code(
            "stats.per_niche.describe().round(2)",
        ),
        md(
            "## 4. Visualisations",
            "",
            "Every plot in EcoFoundation returns a matplotlib Figure styled with",
            "Helvetica + fonttype=42 (Illustrator-editable). The function below saves",
            "the figure as PDF so you can edit it offline.",
        ),
        code(
            "from ecofoundation.reporting.plots import (",
            "    niche_size_distribution, niches_per_group_bar, niche_centroids_spatial,",
            "    co_occurrence_heatmap, niche_density_histogram, heterogeneity_histogram,",
            ")",
            "from ecofoundation.reporting.style import save_pdf",
            "",
            "fig = niche_size_distribution(knn_niches)",
            "save_pdf(fig, REPO / 'examples/notebooks/_out/01_niche_sizes.pdf')",
            "fig",
        ),
        code(
            "fig = niches_per_group_bar(knn_niches)",
            "fig",
        ),
        code(
            "fig = niche_centroids_spatial(",
            "    adata, knn_niches,",
            "    sample_key=data_cfg.sample_id_col,",
            "    spatial_key=data_cfg.spatial_key,",
            "    cell_sample=6000,",
            ")",
            "fig",
        ),
        code(
            "fig = co_occurrence_heatmap(stats)",
            "fig",
        ),
        code(
            "fig = niche_density_histogram(stats)",
            "fig",
        ),
        code(
            "fig = heterogeneity_histogram(stats)",
            "fig",
        ),
        md(
            "## 5. Saving the niche assignment",
            "",
            "For downstream steps (graph construction, GNN training, unsupervised",
            "clustering) we persist the long-form niche → cell membership table.",
        ),
        code(
            "long_rows = []",
            "for nid in range(knn_niches.n_niches):",
            "    for c in knn_niches.cells_per_niche[nid].tolist():",
            "        long_rows.append({",
            "            'niche_id': nid,",
            "            'cell_index': c,",
            "            'ego_cell': int(knn_niches.ego_cell[nid]),",
            "            'patient': str(knn_niches.group_label[nid]),",
            "        })",
            "df = pd.DataFrame(long_rows)",
            "df.head()",
        ),
    ]
    return save("01_niches_and_characterization.ipynb", cells)


# ============================================================================
# 02_graphs_and_lr.ipynb
# ============================================================================


def build_graphs_notebook() -> Path:
    cells = [
        md(
            "# 02 — Per-niche graphs + ligand-receptor scoring",
            "",
            "Step 3 of the pipeline. Each niche becomes one PyG `Data` object:",
            "",
            "- **Nodes** = cells of the niche. Features = expression (HVG subset by default).",
            "- **Edges** = intra-niche Delaunay edges (cached per patient).",
            "- **Edge features** = euclidean distance + tier-1 LR-consensus score.",
            "",
            "The LR score per edge aggregates ligand × receptor expression over all pairs",
            "in the OmniPath consensus database (loaded via LIANA).",
        ),
        code(
            "import sys; from pathlib import Path",
            "REPO = Path.cwd().parent.parent",
            "if str(REPO / 'src') not in sys.path: sys.path.insert(0, str(REPO / 'src'))",
            "",
            "from ecofoundation.config.schemas import DataConfig, GraphConfig, LRScoringConfig, NicheConfig",
            "from ecofoundation.io.readers import load_anndata",
            "from ecofoundation.niches.assembly import assign_niches",
            "from ecofoundation.graph.construction import build_niche_graphs",
            "import numpy as np",
        ),
        code(
            "data_cfg = DataConfig(path=REPO / 'scVI_adata_annotated.h5ad')",
            "adata = load_anndata(data_cfg)",
            "niches, _ = assign_niches(adata, data_cfg, NicheConfig(strategy='knn', knn_k=50))",
            "print('Built', niches.n_niches, 'niches')",
        ),
        md(
            "## Graph construction",
            "",
            "Node features come from `layers['X_exp']`. With `gene_subset='hvg', n_hvg=500`",
            "we use the top-500 highly variable genes — a balance between expressiveness",
            "and tractability for the GNN. Edge features get distance + tier-1 LR score.",
        ),
        code(
            "graph_cfg = GraphConfig(",
            "    node_feature_source='expression',",
            "    node_expression_layer='X_exp',",
            "    gene_subset='hvg',",
            "    n_hvg=500,",
            "    edge_topology='delaunay_intra_niche',",
            "    edge_feature_distance=True,",
            "    edge_feature_normalize_distance=True,",
            "    lr_scoring=LRScoringConfig(enabled=True, resource='omnipath_consensus'),",
            ")",
            "result = build_niche_graphs(adata, niches, data_cfg, graph_cfg)",
            "result.summary",
        ),
        md(
            "## Inspect one graph",
            "",
            "Each `Data` object has the usual PyG attributes plus EcoFoundation metadata",
            "(`niche_id`, `patient`, `sample`, `global_cell_indices`).",
        ),
        code(
            "g = result.graphs[0]",
            "print('Niche', g.niche_id, '| patient', g.patient, '| sample', g.sample)",
            "print('nodes:', g.num_nodes, '| edges:', g.edge_index.shape[1] // 2)",
            "print('node feature dim:', g.x.shape[1])",
            "print('edge feature dim:', g.edge_attr.shape[1])",
            "print('edge feature names:', result.edge_feature_names)",
        ),
        md(
            "## Diagnostic plots",
        ),
        code(
            "from ecofoundation.reporting.plots import (",
            "    graph_size_distributions, edge_feature_distributions, niche_graph_figure,",
            ")",
            "import torch",
            "fig = graph_size_distributions(result.graphs)",
            "fig",
        ),
        code(
            "all_attrs = torch.cat([g.edge_attr for g in result.graphs], dim=0).cpu().numpy()",
            "fig = edge_feature_distributions(all_attrs, result.edge_feature_names)",
            "fig",
        ),
        md(
            "## Plot one niche-graph",
            "",
            "Edge width encodes the chosen edge feature channel (here: LR score).",
        ),
        code(
            "coords = np.asarray(adata.obsm['spatial'])[:, :2]",
            "g = result.graphs[0].clone()",
            "g.pos = torch.from_numpy(coords[g.global_cell_indices.cpu().numpy()].astype(np.float32))",
            "fig = niche_graph_figure(g, edge_feature_index=1, edge_feature_name='lr_score_tier1')",
            "fig",
        ),
    ]
    return save("02_graphs_and_lr.ipynb", cells)


# ============================================================================
# 03_supervised_training_and_explain.ipynb
# ============================================================================


def build_supervised_notebook() -> Path:
    cells = [
        md(
            "# 03 — Supervised GNN + Explainability",
            "",
            "Step 4 (training) + Step 5/5.5 (explainability) in one notebook.",
            "",
            "**What we predict.** We use the binary `sonicated` annotation as a demo",
            "target. The GINE classifier (with adversarial debiasing against patient ID)",
            "is trained with 5-fold patient-level CV on the niches.",
            "",
            "**What we explain.** After training we run GNNExplainer + Integrated Gradients,",
            "and turn the per-niche attributions into class-level summaries: top genes,",
            "edge-feature channels, cell-type importance, LR-pair × cell-type-pair",
            "interactions, pathway enrichment per cell type, and a niche-embedding UMAP.",
            "",
            "Because the full run can take 15–20 min, this notebook drives the **prebuilt",
            "pipelines** rather than redoing everything cell-by-cell.",
        ),
        code(
            "import sys; from pathlib import Path",
            "REPO = Path.cwd().parent.parent",
            "if str(REPO / 'src') not in sys.path: sys.path.insert(0, str(REPO / 'src'))",
            "",
            "from ecofoundation.config.loader import load_config",
            "from ecofoundation.pipelines.train import run_training_pipeline",
            "from ecofoundation.pipelines.explain import (",
            "    ExplainPipelineInputs, run_explainability_pipeline,",
            ")",
        ),
        md(
            "## 1. Train",
            "",
            "Loads `configs/train_sonicated.yaml` (adversarial debiasing enabled by",
            "default). Set `cfg.training.adversarial.enabled = False` if you want to",
            "compare against the baseline.",
        ),
        code(
            "cfg = load_config(REPO / 'configs/train_sonicated.yaml')",
            "# Uncomment to disable adversarial debiasing:",
            "# cfg.training.adversarial.enabled = False",
            "train_folder = run_training_pipeline(cfg)",
            "print('Trained run:', train_folder.run_id)",
            "print('Report     :', train_folder.report_path)",
        ),
        md(
            "## 2. Explain",
            "",
            "Loads the model checkpoint from the training run and produces the",
            "explainability report (gene importance, LR × cell-type, pathway, embedding UMAP).",
        ),
        code(
            "cfg.run_name = 'explain_sonicated_nb'",
            "inputs = ExplainPipelineInputs(",
            "    source_model_path=train_folder.artifacts_dir / 'model_fold0.pt',",
            "    source_predictions_path=train_folder.artifacts_dir / 'test_predictions.parquet',",
            "    target_name='sonicated',",
            "    top_k_per_outcome=8,",
            "    gnn_explainer_epochs=80,",
            "    ig_steps=16,",
            ")",
            "explain_folder = run_explainability_pipeline(cfg, inputs)",
            "print('Explainability report:', explain_folder.report_path)",
        ),
        md(
            "## 3. Inspect the artefacts",
            "",
            "All outputs are persisted as parquet in `runs/<run_id>/artifacts/` and",
            "as standalone PDFs in `runs/<run_id>/pdf/`.",
        ),
        code(
            "import pandas as pd",
            "gene_imp_c0 = pd.read_parquet(explain_folder.artifacts_dir / 'gene_importance_class0.parquet')",
            "gene_imp_c1 = pd.read_parquet(explain_folder.artifacts_dir / 'gene_importance_class1.parquet')",
            "ct_imp     = pd.read_parquet(explain_folder.artifacts_dir / 'cell_type_importance.parquet')",
            "lr_imp     = pd.read_parquet(explain_folder.artifacts_dir / 'lr_interaction_attribution.parquet')",
            "pathways   = pd.read_parquet(explain_folder.artifacts_dir / 'pathway_enrichment.parquet')",
            "gene_imp_c0.head(10)",
        ),
        md(
            "## 4. Render plots individually",
            "",
            "Every plot in the report has a corresponding factory function in",
            "`ecofoundation.reporting.plots`. You can rebuild them with custom",
            "parameters (top_n, colourmap, ...).",
        ),
        code(
            "from ecofoundation.reporting.plots import (",
            "    top_gene_importance_bar, edge_channel_importance_bar,",
            "    cell_type_importance_heatmap, lr_celltype_pair_heatmap,",
            "    pathway_dotplot,",
            ")",
            "from ecofoundation.interpretation.aggregation import ClassAttribution",
            "",
            "attr_c1 = ClassAttribution(",
            "    target_class=1,",
            "    n_niches_explained=int(gene_imp_c1['mean_abs_attr'].notna().sum()),",
            "    gene_importance=gene_imp_c1,",
            "    edge_channel_importance=pd.DataFrame(columns=['channel','mean_abs_attr']),",
            "    node_mask_summary=pd.DataFrame(columns=['gene','mean_node_mask']),",
            ")",
            "fig = top_gene_importance_bar(attr_c1, top_n=15, class_label='Sonicated')",
            "fig",
        ),
    ]
    return save("03_supervised_training_and_explain.ipynb", cells)


# ============================================================================
# 04_unsupervised_clustering.ipynb
# ============================================================================


def build_unsupervised_notebook() -> Path:
    cells = [
        md(
            "# 04 — Unsupervised niche clustering + biology",
            "",
            "Step 6 + 6.5 + 7. Trains a Graph Auto-Encoder (or DGI) on every cell's",
            "niche, projects all 259k niches into a 64-dim embedding space, runs Leiden",
            "to discover niche-clusters, then enriches each cluster with:",
            "",
            "- pathway enrichment on its top marker genes",
            "- ligand-receptor interactions across its niches",
            "- example niche visualisations with cell-type colouring",
            "",
            "Adversarial debiasing against `patient_id` pushes the encoder toward",
            "patient-invariant representations.",
            "",
            "Outputs are persisted as `ecof_annotated.h5ad` — a copy of the input",
            "AnnData with `obs['ecof_niche_cluster']`, `obsm['ecof_niche_embedding']`",
            "and `obsm['ecof_niche_umap']` attached.",
        ),
        code(
            "import sys; from pathlib import Path",
            "REPO = Path.cwd().parent.parent",
            "if str(REPO / 'src') not in sys.path: sys.path.insert(0, str(REPO / 'src'))",
            "",
            "from ecofoundation.config.loader import load_config",
            "from ecofoundation.pipelines.unsup_cluster import run_unsup_clustering_pipeline",
        ),
        md(
            "## 1. Run the pipeline",
            "",
            "Defaults: GAE, 30% subsample for training, adversarial debiasing on",
            "`patient_id`, AnnData export enabled.",
            "",
            "Switch to DGI by setting `cfg.unsup.model.architecture = 'dgi'`.",
        ),
        code(
            "cfg = load_config(REPO / 'configs/unsup_clustering.yaml')",
            "# cfg.unsup.model.architecture = 'dgi'   # alternative",
            "# cfg.unsup.adversarial.lambda_max = 0.3  # gentler debias",
            "folder = run_unsup_clustering_pipeline(cfg)",
            "print('Report :', folder.report_path)",
            "print('AnnData:', folder.artifacts_dir / 'ecof_annotated.h5ad')",
        ),
        md(
            "## 2. Load the annotated AnnData",
            "",
            "Standard scanpy-style usage from here on.",
        ),
        code(
            "import anndata as ad",
            "a = ad.read_h5ad(folder.artifacts_dir / 'ecof_annotated.h5ad', backed='r')",
            "print(a)",
            "print('Niche-cluster counts:')",
            "print(a.obs['ecof_niche_cluster'].value_counts().head())",
        ),
        md(
            "## 3. Cluster-biology artefacts",
            "",
            "The Step-7 deep-dive writes three parquet tables.",
        ),
        code(
            "import pandas as pd",
            "summary = pd.read_parquet(folder.artifacts_dir / 'niche_cluster_summary.parquet')",
            "markers = pd.read_parquet(folder.artifacts_dir / 'niche_cluster_markers.parquet')",
            "summary",
        ),
        code(
            "markers[markers['rank'] <= 3].head(20)",
        ),
        code(
            "pathways_path = folder.artifacts_dir / 'cluster_pathways.parquet'",
            "if pathways_path.exists():",
            "    pathways = pd.read_parquet(pathways_path)",
            "    display(pathways.groupby('cluster').head(2)[['cluster','term','adjusted_p_value','combined_score']])",
        ),
        code(
            "lr_path = folder.artifacts_dir / 'cluster_lr_interactions.parquet'",
            "if lr_path.exists():",
            "    lr = pd.read_parquet(lr_path)",
            "    display(lr.groupby('cluster').head(3)[['cluster','ct_pair_a','ct_pair_b','ligand','receptor','score_sum']])",
        ),
        md(
            "## 4. Plots, on demand",
            "",
            "All factory functions live in `ecofoundation.reporting.plots`; pass the",
            "cluster_stats / biology objects directly.",
        ),
        code(
            "from ecofoundation.reporting.plots import (",
            "    niche_cluster_composition_bar, niche_cluster_marker_heatmap,",
            "    cluster_pathway_dotplot, cluster_lr_heatmap,",
            ")",
            "from ecofoundation.niches.cluster_characterization import NicheClusterStats",
            "from ecofoundation.niches.cluster_biology import ClusterBiology",
            "",
            "comp = pd.read_parquet(folder.artifacts_dir / 'niche_cluster_composition.parquet')",
            "stats_lite = NicheClusterStats(",
            "    composition=comp,",
            "    size_distribution=pd.DataFrame(),",
            "    sample_distribution=pd.DataFrame(),",
            "    markers=markers,",
            "    cluster_summary=summary,",
            ")",
            "fig = niche_cluster_composition_bar(stats_lite)",
            "fig",
        ),
        code(
            "fig = niche_cluster_marker_heatmap(stats_lite, n_top=5)",
            "fig",
        ),
        code(
            "if pathways_path.exists() and lr_path.exists():",
            "    biology = ClusterBiology(",
            "        pathways=pd.read_parquet(pathways_path),",
            "        lr_interactions=pd.read_parquet(lr_path),",
            "        example_niches={},",
            "    )",
            "    display(cluster_pathway_dotplot(biology))",
            "    display(cluster_lr_heatmap(biology))",
        ),
        md(
            "## 5. Cell-level niche cluster on the spatial map",
            "",
            "`scanpy.pl.spatial` works directly on the exported AnnData.",
        ),
        code(
            "# Load fully into memory for plotting",
            "import anndata as ad",
            "a_full = ad.read_h5ad(folder.artifacts_dir / 'ecof_annotated.h5ad')",
            "# Drop cells without a niche assignment for cleaner plots",
            "sub = a_full[a_full.obs['ecof_niche_cluster'].astype(str) != 'unassigned']",
            "print(sub.obs['ecof_niche_cluster'].value_counts().head())",
            "",
            "# Optional: render via the project's matplotlib helpers",
            "from ecofoundation.reporting.plots import niche_cluster_spatial",
            "import numpy as np",
            "fig = niche_cluster_spatial(",
            "    sub,",
            "    sample_key='samples',",
            "    spatial_key='spatial',",
            "    cluster_per_cell=np.array(sub.obs['ecof_niche_cluster'].astype(str)),",
            "    max_points_per_sample=4000,",
            ")",
            "fig",
        ),
    ]
    return save("04_unsupervised_clustering.ipynb", cells)


def main() -> None:
    paths = [
        build_overview(),
        build_niches_notebook(),
        build_graphs_notebook(),
        build_supervised_notebook(),
        build_unsupervised_notebook(),
    ]
    print("Wrote notebooks:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
