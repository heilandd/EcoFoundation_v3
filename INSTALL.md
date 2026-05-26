# EcoFoundation — Installation Guide

This document walks through installing EcoFoundation from the GitHub repo and
running the first demo. For background on the package and its pipelines, see
[`README.md`](README.md).

---

## 1. Prerequisites

EcoFoundation supports Linux, macOS (Apple Silicon and Intel), and Windows
(via WSL2 recommended). You need:

| Tool | Why | Install |
|---|---|---|
| **Python 3.11** | Project runtime | already on macOS / install via `pyenv` on Linux or python.org |
| **`uv` package manager** | Reproducible dependency installs from `uv.lock` | `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Git** | Clone + future updates | system package manager |

Optional:
- **CUDA-capable GPU** for fast supervised training on Linux/Windows
- **Apple-Silicon Mac** for MPS acceleration (Intel macOS works too, just on CPU)

---

## 2. Clone the repository

```bash
git clone https://github.com/heilandd/EcoFoundation_v3.git
cd EcoFoundation_v3
```

---

## 3. Install dependencies

EcoFoundation uses [uv](https://docs.astral.sh/uv/) to manage a project-local
virtual environment from the committed `uv.lock`. Run:

```bash
uv sync --extra dev
```

This creates `.venv/` in the project root (≈ 4 GB after PyTorch + PyG) and
installs the exact dependency versions captured in `uv.lock`:

- `torch>=2.2,<2.3`, `torch-geometric>=2.5,<2.6` (last versions with x86_64 macOS wheels)
- `anndata`, `scanpy`, `squidpy`, `liana`, `gseapy<1.1`
- `pydantic`, `loguru`, `jinja2`, `matplotlib`, `umap-learn`, `nbformat`, `jupyter`
- dev: `pytest`, `hypothesis`, `ruff`, `mypy`

### Platform-specific notes

- **Intel-Mac (x86_64):** the pinned versions are deliberately conservative
  because newer torch/numba/gseapy no longer publish x86_64 macOS wheels.
- **Apple-Silicon Mac (arm64):** the pins still install fine but you can
  relax them in `pyproject.toml` (`torch>=2.7`, `numba>=0.61`) on your fork.
- **Linux + CUDA:** install a matching `torch` wheel for your CUDA version
  *before* `uv sync`:
  ```bash
  uv pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
  uv sync --extra dev
  ```

---

## 4. Verify the installation

```bash
.venv/bin/python -m pytest tests/                       # full suite, ~13 s
.venv/bin/python -c "import ecofoundation; print(ecofoundation.__version__)"
```

All **110 tests** should pass with at most a couple of zarr/dask deprecation
warnings.

---

## 5. Get the demo dataset

The shipped demos expect a 6.6 GB Xenium AnnData at the repo root. It is
**not** in the git repo (too large). Two options:

### a) Use the original demo AnnData

```bash
# Place your scVI-annotated Xenium .h5ad here
cp /path/to/your/scVI_adata_annotated.h5ad ./scVI_adata_annotated.h5ad
```

### b) Use your own AnnData

Edit `configs/*.yaml` to point at your file and adjust the column names:

```yaml
data:
  path: /absolute/path/to/your.h5ad
  sample_id_col: <your sample column>
  patient_id_col: <your patient column>
  condition_col: <your condition column>
  celltype_col: <your cell-type column>
  spatial_key: spatial          # name of obsm key holding (x, y) coords
  counts_layer: counts          # layer with raw counts
  normalized_layer: X_exp       # layer with log1p-normalised counts
  embedding_key: X_scVI         # optional pre-computed embedding
```

The minimum required structure is `obs[sample_id_col]` + `obsm[spatial_key]`.

---

## 6. Run the first demo

```bash
.venv/bin/python examples/01_hello_world.py
```

This loads the AnnData, computes QC + Leiden clusters + marker genes, and
writes an interactive HTML report to `runs/<timestamp>__hello_world__<hash>/`.
Open `report.html` in any browser.

Other demos (each takes 1–60 minutes depending on the step):

```bash
.venv/bin/python examples/02_niches_demo.py            # niches + characterisation
.venv/bin/python examples/03_graphs_demo.py            # PyG graphs + LR scoring
.venv/bin/python examples/04_train_demo.py             # supervised GINE + adversarial
.venv/bin/python examples/05_explain_demo.py           # explainability deep-dive
.venv/bin/python examples/06_unsup_clustering_demo.py  # unsupervised + cluster biology + AnnData export
```

---

## 7. Use the Jupyter notebooks

Install the project's kernel and launch JupyterLab:

```bash
.venv/bin/python -m ipykernel install --user --name ecofoundation \
    --display-name 'Python 3 (ecofoundation)'
.venv/bin/jupyter lab examples/notebooks/
```

Recommended order:

1. `00_overview.ipynb`
2. `01_niches_and_characterization.ipynb`
3. `02_graphs_and_lr.ipynb`
4. `03_supervised_training_and_explain.ipynb`
5. `04_unsupervised_clustering.ipynb`

---

## 8. Use EcoFoundation in your own Python code

Activate the venv (optional — running with `.venv/bin/python` works too):

```bash
source .venv/bin/activate
```

Minimal example:

```python
from ecofoundation.config.loader import load_config
from ecofoundation.pipelines.unsup_cluster import run_unsup_clustering_pipeline

cfg = load_config("configs/unsup_clustering.yaml")
folder = run_unsup_clustering_pipeline(cfg)
print(f"Report: {folder.report_path}")
print(f"AnnData: {folder.artifacts_dir / 'ecof_annotated.h5ad'}")
```

Or piece by piece:

```python
from ecofoundation.io.readers import load_anndata
from ecofoundation.config.schemas import DataConfig, NicheConfig
from ecofoundation.niches.assembly import assign_niches

cfg_data = DataConfig(path="my.h5ad", sample_id_col="samples", patient_id_col="patient")
adata = load_anndata(cfg_data)
niches, _ = assign_niches(adata, cfg_data, NicheConfig(strategy="knn", knn_k=50))
print(f"Built {niches.n_niches} niches")
```

---

## 9. Updating

```bash
git pull
uv sync --extra dev          # refreshes installed deps if uv.lock changed
.venv/bin/python -m pytest tests/
```

---

## 10. Troubleshooting

**`torch` install fails on Intel macOS.**
Already handled — pins are `torch<2.3`. If `uv sync` still complains, you
might be on a system Python that's not 3.11. Force 3.11 via `uv python pin 3.11`.

**`MPS backend doesn't implement aten::scatter_reduce.two_out`.**
EcoFoundation auto-sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so this op falls
back to CPU. No action needed.

**Enrichr times out for pathway enrichment.**
The Step-5.5 / Step-7 pathway enrichment calls the Enrichr web API. If
you are offline or behind a strict proxy, set `pathway_enrichment_enabled
= False` in the config or skip those sections; everything else still runs.

**`gseapy` install fails on x86_64 macOS with a Rust error.**
`gseapy 1.1+` needs a Rust toolchain. We pin to `<1.1` in `pyproject.toml`.
Make sure `uv sync` picked that pin up; if you bumped it, install Rust via
`rustup` first.

**Out-of-memory during the full unsupervised demo.**
259k niches × ~50 cells × 500 features ≈ 25 GB in memory. On a smaller
machine, lower `unsup.subsample_fraction_for_training` (default `0.3`) and
the `graph.n_hvg` (default `500`) in `configs/unsup_clustering.yaml`. The
final inference + AnnData export always run on all niches regardless.

---

## 11. Where things live

```
EcoFoundation_v3/
├── README.md                 high-level overview + pipeline diagram
├── INSTALL.md                this file
├── CLAUDE.md                 context for AI coding agents
├── pyproject.toml            project + dependency definitions
├── uv.lock                   exact dependency versions (committed)
├── configs/                  one YAML per demo / pipeline
├── examples/                 Python demo scripts (01..06)
│   └── notebooks/            five step-by-step Jupyter notebooks
├── src/ecofoundation/        the package itself
│   ├── config, io, preprocessing, niches, graph,
│   ├── models (sup + unsup + adversarial),
│   ├── training, interpretation, reporting, pipelines, utils
├── tests/                    110 unit + integration tests
└── tools/build_notebooks.py  regenerates the notebooks programmatically
```
