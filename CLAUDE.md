# CLAUDE.md — Project context for AI agents

This file orients Claude / Codex / similar coding agents working in this repo.

## What this codebase is

EcoFoundation is a modular Python package for spatial transcriptomics analysis
(primary modality: 10x Xenium). It ships two end-to-end pipelines:

1. **Unsupervised niche clustering** (`pipelines/unsup_cluster.py`) — custom
   GAE/VGAE/DGI (PyG, written from scratch) over per-cell niches; produces
   niche-cluster labels written back into AnnData.
2. **Supervised niche classification** (`pipelines/train.py`) — GINE/GAT-edge
   on per-niche graphs; graph-level prediction with full explainability
   (`pipelines/explain.py`).

Each run produces a self-contained HTML report (matplotlib SVG inline +
editable PDFs on disk) plus all measurements as tidy parquet tables.

## Hard architectural constraints

These were chosen with the user and should not be changed casually:

1. **Patient-aware niches** — a niche never crosses patients
   (`niches/assembly.py`). Patient column is configurable via
   `DataConfig.patient_id_col`.
2. **KNN-50 is the default niche strategy** — produces uniform-size niches;
   Delaunay-3-hop is opt-in for density-aware analyses
   (`config.NicheConfig.strategy`).
3. **Overlap filter is mode-dependent**:
   - Supervised → `overlap_filter_enabled=True` (default).
   - Unsupervised → `overlap_filter_enabled=False` so every cell gets one niche.
4. **Both node and edge features must be attributable.** GINEConv + GATv2Conv
   with `edge_dim` satisfy this; new architectures must preserve this.
5. **Adversarial debiasing** (`models/adversarial.py`) is the project's
   answer to small-cohort sample bias. Active by default in both pipelines,
   debiases against `patient_id`.
6. **Adaptive train/test split** — patient-level GroupKFold if `n_patients ≥
   min_patients_for_kfold`; subgraph-level 70/30 fallback otherwise.
7. **Matplotlib only.** All plots use the central style in
   `reporting/style.py` (PDF backend, Helvetica, `fonttype=42`, 6 pt default).
   Plotly is intentionally **not** used — figures must be Illustrator-editable.
8. **AnnData export** — the unsupervised pipeline persists
   `artifacts/ecof_annotated.h5ad` with all results in `obs` / `obsm` / `uns`
   so downstream tools can consume the run directly.

## Environment quirks (Intel-Mac compatibility)

Pinned in `pyproject.toml`:

- `torch>=2.2,<2.3` — last version with x86_64 macOS wheels.
- `torch-geometric>=2.5,<2.6`.
- `numba<0.61`, `llvmlite<0.45` — same reason.
- `gseapy>=1.0,<1.1` — gseapy 1.1+ needs a Rust toolchain.

`ecofoundation/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so PyG ops
without MPS kernels silently dispatch to CPU. The supervised explainer
**forces device=cpu** because Captum accumulates in float64 (MPS rejects).

## Repository layout

```
src/ecofoundation/
├── config/          Pydantic schemas + YAML loader
├── io/              AnnData read/validate + RunFolder + ecof_annotated export
├── preprocessing/   QC stats (descriptive)
├── niches/          NicheStrategy ABC + 4 strategies + overlap + characterisation + cluster biology
├── graph/           Per-niche PyG graphs, edge features, LIANA LR resource
├── models/
│   ├── adversarial.py    GradReverse + AdversarialBatchHead
│   ├── heads.py          MultiTaskHead + loss registry
│   ├── pooling.py        mean / max / attention pooling
│   ├── sup/              GINE (default) + GAT-edge
│   └── unsup/            GAE + VGAE + DGI (all custom, GINE-encoder-based)
├── training/        sup trainer, unsup trainer, CV/splits, metrics
├── interpretation/  GNNExplainer + IG + cell-type + LR + pathway + embedding-UMAP
├── reporting/       ReportBuilder + Jinja2 template + ~25 plot helpers
└── pipelines/       clustering, niches, graphs, train, explain, unsup_cluster
configs/             one YAML per demo (hello_world / niches_demo / niches_unsupervised
                     / graphs_demo / train_sonicated / unsup_clustering)
examples/            one script per demo
examples/notebooks/  one notebook per pipeline step
tests/               110 tests (unit + integration)
```

## How to add things

- **New plot** — add to `src/ecofoundation/reporting/plots/<topic>.py`, import
  `new_figure`, `style_axes` from `reporting.style`, return a
  `matplotlib.figure.Figure`. Export from `reporting/plots/__init__.py`.
- **New analysis** — module under the appropriate domain dir
  (`niches/`, `graph/`, `interpretation/`, ...); add Pydantic config if it
  has parameters; write at least one unit test + smoke integration.
- **New pipeline** — `pipelines/<name>.py` calling existing helpers; pass
  `pdf_dir=folder.root / "pdf"` to `ReportBuilder` for Illustrator outputs.
- **Run a demo** — `.venv/bin/python examples/0<N>_<name>.py`. Background
  long runs: use `run_in_background=true` in Bash, log to `/tmp`.

## Testing

```bash
.venv/bin/python -m pytest tests/                  # full suite (~13s)
.venv/bin/python -m pytest tests/unit/             # quick (~2s)
.venv/bin/python -m pytest -m "slow"               # slow integration
```

## Memory location

Project memory lives in
`/Users/henrikheiland/.claude/projects/-Users-henrikheiland-Desktop-Coding-EcoFoundation/memory/`
(MEMORY.md indexes the files). Update it when architectural decisions change.

## Pre-existing data

Demo dataset: `scVI_adata_annotated.h5ad` at the repo root (6.6 GB Xenium 5K
brain, 259k cells, 5 patients × 10 samples). Column conventions:

- `obs['samples']`, `obs['patient']`, `obs['sonicated']` (binary target)
- `obs['celltype_level_1']` (13 broad CTs), `obs['celltype_level_2']` (31 fine)
- `obsm['spatial']`, `obsm['X_scVI']`, `obsm['X_pca']`, `obsm['X_umap']`
- `layers['counts']` (raw), `layers['X_exp']` (log1p-normalised scVI output)

## Style / conventions

- All public functions get docstrings + type hints.
- Use `loguru` via `utils.logging.get_logger(__name__)`.
- Configs are Pydantic with `extra="forbid"`.
- Long-form parquet for tabular outputs, `.npz` for arrays, `.pt` for models.
- Pipeline metadata in `manifest.json` (run_id, config hash, input hash, lib versions).
