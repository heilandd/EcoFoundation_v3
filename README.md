# EcoFoundation

Modular Python package for spatial transcriptomics analysis (Xenium / Visium / MERFISH / CosMx).
Two pipelines, both producing self-contained interactive HTML reports plus
Illustrator-editable PDFs:

1. **Unsupervised spatial-niche clustering** — custom GAE/VGAE/DGI on per-cell niches → niche-cluster labels written back into AnnData.
2. **Supervised GNN niche classification** — GINE/GAT-edge on per-niche graphs → graph-level predictions with full node + edge + cell-type + ligand-receptor + pathway explainability.

## Quickstart

```bash
uv sync --extra dev
.venv/bin/python -m pytest tests/                   # 110 tests
.venv/bin/python examples/01_hello_world.py         # demo run on the shipped dataset
```

Every run writes to `runs/<timestamp>__<run_name>__<config_hash>/`:
- `report.html` — self-contained, all plots inline as SVG
- `pdf/*.pdf` — same plots as Illustrator-editable PDFs
- `artifacts/*.parquet` — all measurements as tidy tables
- `artifacts/ecof_annotated.h5ad` — your AnnData with all results added (unsupervised pipeline)

## Notebooks

Step-by-step walkthroughs in [`examples/notebooks/`](examples/notebooks/):

| Notebook | Pipeline step | Coverage |
|---|---|---|
| `00_overview.ipynb` | — | Project map |
| `01_niches_and_characterization.ipynb` | 2 / 2.5 | Niche construction + characterisation |
| `02_graphs_and_lr.ipynb` | 3 | PyG graphs + LR scoring |
| `03_supervised_training_and_explain.ipynb` | 4 + 5 + 5.5 | GINE training + GNNExplainer/IG + biology |
| `04_unsupervised_clustering.ipynb` | 6 + 6.5 + 7 | GAE/DGI + adversarial + cluster biology + AnnData export |

Run them with the project's `.venv` kernel:

```bash
.venv/bin/python -m ipykernel install --user --name ecofoundation --display-name 'Python 3 (ecofoundation)'
.venv/bin/jupyter lab examples/notebooks/
```

## Pipeline overview

```
                    DataConfig + AnnData
                            │
                ┌───────────┴───────────┐
                │                       │
        Niches (kNN/Delaunay/         (skip — supervised quick path
        radius/tiling) +              uses all cells)
        characterisation              │
                │                     │
        PyG niche-graphs              │
        (intra-niche Delaunay,        │
         distance + LR features)      │
                │                     │
       ┌────────┴─────────┐           │
       │                  │           │
   GAE / VGAE          GINE /         │
   DGI training        GAT-edge       │
   (Step 6)            (Step 4)       │
       │                  │           │
   Niche embeddings    Supervised     │
   → Leiden            predictions    │
   → niche clusters    + metrics      │
       │                  │           │
   Cluster biology     Explainability │
   (pathways +         (GNNExplainer  │
    LR per cluster     + IG + cell    │
    + examples,        type + LR      │
    Step 7)            + pathway +    │
                       embedding UMAP,│
                       Step 5/5.5)    │
       │                  │           │
   AnnData export      HTML report
   (Step 6 + 7)        per CV fold
```

## Design choices

- **Patient-aware niches.** A niche never spans two patients. Configurable via `DataConfig.patient_id_col`.
- **KNN-50 default niches.** Standardised size (51 cells incl. ego) for stable downstream stats; Delaunay-3-hop available for density-aware analyses.
- **Overlap-filter is mode-dependent.**
  - Supervised → `overlap_filter_enabled=True` (default) prevents train/test leakage.
  - Unsupervised → `overlap_filter_enabled=False` so every cell gets one niche.
- **Adversarial debiasing** (Step 6.5). DANN-style Gradient Reversal + batch head, default debias against `patient_id`. Active by default in both pipelines.
- **Subsampling for training**. The unsupervised pipeline trains on a 30 % subsample by default; inference and AnnData export always cover all niches.
- **Matplotlib only, Illustrator-editable.** PDF backend, Helvetica, `fonttype=42`, white bg + black frame, 6 pt default. Plots embed as inline SVG in the report and as separate PDFs in `runs/<id>/pdf/`.

## Persisted artefacts per run

| File | Pipeline | Contents |
|---|---|---|
| `niche_assignment.parquet` | all | long-form cell → niche membership |
| `niche_characterization.parquet` | all | per-niche size, density, entropy, purity, radius |
| `co_occurrence_matrix.csv` | all | center × neighbour cell-type frequency |
| `niche_graphs.pt` | Step 3 | list of PyG `Data` objects |
| `niche_graphs_metadata.parquet` | Step 3 | per-niche graph metadata |
| `model_fold0.pt` | Step 4 | trained GINE/GAT checkpoint |
| `test_predictions.parquet` | Step 4 | per-fold test predictions |
| `gene_importance_class*.parquet` | Step 5 | IG-derived gene attribution per class |
| `edge_channel_importance_class*.parquet` | Step 5 | distance vs LR importance per class |
| `cell_type_importance.parquet` | Step 5.5 | per (class × cell-type) attribution |
| `lr_interaction_attribution.parquet` | Step 5.5 | LR-pair × cell-type-pair attribution per class |
| `pathway_enrichment.parquet` | Step 5.5 | Enrichr terms per (class × cell-type) |
| `niche_embeddings.npz` | Step 6 | (n_niches, hidden_dim) GAE embedding + UMAP + cluster labels |
| `unsup_gae.pt` | Step 6 | trained GAE/VGAE/DGI checkpoint |
| `niche_cluster_*.parquet` | Step 6 | per-cluster composition / markers / summary |
| `cluster_pathways.parquet` | Step 7 | Enrichr terms per niche cluster |
| `cluster_lr_interactions.parquet` | Step 7 | LR pair × cell-type pair per niche cluster |
| `ecof_annotated.h5ad` | Step 6 | input AnnData with all results attached |

## Environment

- Python 3.11
- macOS x86_64-compatible pins on `torch<2.3`, `numba<0.61`, `llvmlite<0.45`, `gseapy<1.1` (these versions are the last with x86_64 macOS wheels; relax on ARM Mac / Linux / Windows).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically in `ecofoundation/__init__.py` so PyG ops that lack MPS implementations silently fall back to CPU.

## License

MIT.
