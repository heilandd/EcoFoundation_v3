"""Pydantic configuration schemas.

All EcoFoundation pipelines consume one ``RunConfig`` validated by these schemas.
Configs are loaded from YAML via :func:`ecofoundation.config.loader.load_config`.

Default values are tuned for the user's Xenium dataset (cf. ``configs/hello_world.yaml``):
sample_id column ``samples``, patient_id column ``patient``, counts in
layer ``counts`` and normalized expression in layer ``X_exp``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    """Base for all configs: forbid unknown fields, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


# ----------------------------- Data -----------------------------------------


class DataConfig(_StrictModel):
    """Where the input AnnData lives and how its columns are named.

    All column-name fields are configurable because Xenium / Visium / MERFISH
    exports use different conventions and the user's batch column may live
    anywhere in ``adata.obs``.
    """

    path: Path = Field(..., description="Path to the input .h5ad file.")
    sample_id_col: str = Field("samples", description="obs column with per-sample IDs.")
    patient_id_col: str | None = Field(
        "patient", description="obs column with patient IDs (None disables patient grouping)."
    )
    condition_col: str | None = Field(
        None, description="obs column for condition/treatment (used for stratification, optional)."
    )
    celltype_col: str | None = Field(
        "celltype_level_1", description="obs column with cell-type annotation (optional)."
    )
    spatial_key: str = Field("spatial", description="obsm key holding spatial coords.")
    counts_layer: str | None = Field(
        "counts", description="layer with raw counts (None = use .X)."
    )
    normalized_layer: str | None = Field(
        "X_exp", description="layer with normalized expression (None = compute from counts)."
    )
    embedding_key: str | None = Field(
        "X_scVI", description="obsm key with pre-computed embedding (used as feature default)."
    )


# ----------------------------- Preprocessing --------------------------------


class QCConfig(_StrictModel):
    """Quality control thresholds. Set ``enabled=False`` to skip when data is preprocessed."""

    enabled: bool = True
    min_counts_per_cell: int = 50
    max_counts_per_cell: int | None = None
    min_genes_per_cell: int = 10
    max_pct_mt: float | None = None  # only used if mitochondrial genes present


class NormalizationConfig(_StrictModel):
    enabled: bool = True
    target_sum: float = 1e4
    log1p: bool = True


class HVGConfig(_StrictModel):
    enabled: bool = True
    n_top_genes: int = 2000
    flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = "seurat_v3"
    batch_key: str | None = None  # if set, HVGs are picked per batch and pooled


# ----------------------------- Clustering -----------------------------------


class LeidenConfig(_StrictModel):
    enabled: bool = True
    resolution: float = 0.8
    n_neighbors: int = 15
    use_rep: str | None = Field(
        "X_scVI", description="obsm key to use; None falls back to PCA."
    )
    random_state: int = 0


class MarkerGenesConfig(_StrictModel):
    enabled: bool = True
    n_top: int = 25
    method: Literal["wilcoxon", "t-test", "logreg"] = "wilcoxon"


# ----------------------------- Niches ---------------------------------------


NicheStrategyName = Literal["delaunay", "knn", "radius", "tiling"]


class NicheConfig(_StrictModel):
    """Niche construction parameters.

    Default: **kNN with k=50** (niche size = 51 cells incl. center). This gives
    standardised niche sizes across samples — important for downstream GNN
    training because Delaunay-based niches vary heavily with local cell density.
    Use ``strategy="delaunay"`` with ``k_hop=3`` for density-aware niches when
    that variability is the point.
    """

    strategy: NicheStrategyName = "knn"
    k_hop: int = Field(3, ge=1, le=8, description="Hops around the ego cell (Delaunay).")
    knn_k: int = Field(
        50,
        ge=2,
        description="Used when strategy='knn'. Niche size = knn_k + 1 (incl. ego).",
    )
    radius: float | None = Field(
        None, description="Used when strategy='radius'. Same unit as spatial coords (µm)."
    )
    max_edge_length: float | None = Field(
        None, description="Prune edges longer than this in µm. None = no pruning."
    )
    edge_length_quantile_cutoff: float | None = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Alternative to max_edge_length: prune above this quantile per niche.",
    )
    overlap_filter_enabled: bool = Field(
        True,
        description=(
            "If True, apply the overlap filter (use for supervised classification — "
            "prevents information leakage between train/test niches). "
            "If False, keep ALL niches (use for unsupervised niche clustering / "
            "cellular neighborhood analysis — every cell needs its own niche)."
        ),
    )
    max_overlap_fraction: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Max pairwise Jaccard overlap between niches. 0 = strictly disjoint. "
            "Only applied when overlap_filter_enabled=True."
        ),
    )
    min_cells_per_niche: int = Field(5, ge=2)
    max_cells_per_niche: int | None = Field(500, description="Cap to keep PyG graphs tractable.")
    # patient_id_col is taken from DataConfig — strategies must respect it.


# ----------------------------- Graph + Edge Features ------------------------


class LRScoringConfig(_StrictModel):
    """Ligand–receptor scoring on edges (cell-cell pairs in a niche)."""

    enabled: bool = True
    resource: Literal["omnipath_consensus", "cellphonedb", "custom"] = "omnipath_consensus"
    custom_resource_path: Path | None = None
    method: Literal["mean", "logfc", "natmi", "connectome"] = "mean"
    aggregation: Literal["tier1_score", "tier2_topn"] = "tier1_score"
    top_n_pairs: int = Field(20, description="Used when aggregation='tier2_topn'.")


EdgeTopology = Literal["delaunay_intra_niche", "knn_intra_niche"]


# ----------------------------- Adversarial debiasing ------------------------


class AdversarialConfig(_StrictModel):
    """Domain-Adversarial debiasing (DANN — Gradient Reversal + Batch Head).

    Pushes the encoder toward features that do NOT predict the nuisance
    label (``batch_col`` — typically ``patient_id``). Empirically helps
    when training data has strong sample-level technical bias.
    """

    enabled: bool = False
    batch_col: str | None = Field(
        None,
        description=(
            "Niche-level nuisance label column. Defaults to "
            "``data.patient_id_col`` when ``None``. Per-niche label is taken "
            "from the ego cell."
        ),
    )
    lambda_max: float = Field(
        1.0, description="Asymptotic strength of the adversarial gradient (after warmup)."
    )
    warmup_epochs: int = Field(
        5, description="Linear ramp of lambda from 0 to lambda_max over this many epochs."
    )
    hidden_dim: int = 64
    dropout: float = 0.1


# ----------------------------- Training / Model -----------------------------


class TargetSpec(_StrictModel):
    """One prediction target (classification or regression).

    The niche label is derived from per-cell labels in ``obs[obs_column]`` via
    ``label_aggregation``:

    - ``ego``:      label of the center cell (default; trivial when label is
                    sample/patient-level and constant within a sample).
    - ``majority``: most frequent label among niche members.
    - ``first``:    first cell's label (used only as a deterministic fallback).
    """

    name: str = Field(..., description="Short identifier used in metrics / outputs.")
    obs_column: str = Field(..., description="Column in adata.obs with per-cell labels.")
    type: Literal["categorical", "numeric"] = "categorical"
    classes: list[str] | None = Field(
        None,
        description="Optional explicit class order. If None, inferred from the data.",
    )
    loss: Literal["cross_entropy", "bce", "mse", "huber", "mae"] = "cross_entropy"
    weight: float = 1.0
    label_aggregation: Literal["ego", "majority", "first"] = "ego"


class ModelConfig(_StrictModel):
    """GNN architecture parameters (supervised classifier)."""

    architecture: Literal["gine", "gat_edge"] = "gine"
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    pooling: Literal["mean", "max", "attention"] = "attention"
    # GAT-specific
    n_heads: int = 4
    # Optional batch-norm between layers
    batch_norm: bool = True


class UnsupModelConfig(_StrictModel):
    """Architecture for the unsupervised niche-clustering GNN."""

    architecture: Literal["gae", "vgae", "dgi"] = "gae"
    hidden_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.1
    pooling: Literal["mean", "max", "attention"] = "mean"
    batch_norm: bool = True


class UnsupClusteringConfig(_StrictModel):
    """Unsupervised niche-clustering pipeline parameters."""

    model: UnsupModelConfig = Field(default_factory=UnsupModelConfig)
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float | None = 1.0
    # Edge reconstruction loss specifics (GAE / VGAE)
    n_neg_samples_per_edge: int = 1
    # DGI specifics
    dgi_corruption: Literal["row_shuffle", "feature_shuffle"] = "row_shuffle"
    # Clustering on the niche-level embedding space
    leiden_resolution: float = 0.6
    leiden_n_neighbors: int = 30
    seed: int = 0
    # Scalability — when n_niches is huge (~259k), training on all is heavy.
    # ``subsample_fraction_for_training`` (default 0.3 = 30%) is preferred;
    # ``max_niches_for_training`` is an absolute cap (taken if set). Inference /
    # embedding extraction always runs on every niche, regardless of subsampling.
    subsample_fraction_for_training: float | None = Field(
        0.3, ge=0.0, le=1.0,
        description="Fraction of niches used for training. Inference is unaffected.",
    )
    max_niches_for_training: int | None = None
    # Adversarial debiasing — applied during training.
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig)
    # AnnData export
    export_anndata: bool = True
    write_compressed_h5ad: bool = True


class TrainingConfig(_StrictModel):
    """Training loop, optimisation, and CV strategy."""

    targets: list[TargetSpec] = Field(
        ..., description="One or more prediction targets. Must contain at least one."
    )
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw"] = "adamw"
    early_stopping_patience: int = 12
    grad_clip: float | None = 1.0
    cv_strategy: Literal["auto", "patient_kfold", "subgraph_split"] = "auto"
    min_patients_for_kfold: int = 5
    n_folds: int = 5
    train_test_ratio: float = 0.7
    stratify_categorical: bool = True
    num_workers: int = 0  # PyG DataLoader workers; 0 = single-process (safer on MPS)
    # Adversarial debiasing — applied within each fold's training loop.
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig)


class GraphConfig(_StrictModel):
    """How a NicheAssignment is converted into per-niche PyG graphs.

    User-requested default: node features come from normalized expression
    (``layers['X_exp']``). Use ``gene_subset='hvg'`` or set ``custom_genes`` to
    reduce dimensionality when training the GNN.
    """

    node_feature_source: Literal["expression", "embedding", "concat"] = "expression"
    node_expression_layer: str = "X_exp"
    node_embedding_key: str = "X_scVI"
    gene_subset: Literal["all", "hvg", "custom"] = "all"
    n_hvg: int = Field(2000, description="Used when gene_subset='hvg'.")
    custom_genes: list[str] | None = None

    edge_topology: EdgeTopology = "delaunay_intra_niche"
    edge_knn_k: int = 8
    edge_feature_distance: bool = True
    edge_feature_normalize_distance: bool = True
    lr_scoring: LRScoringConfig = Field(default_factory=LRScoringConfig)


# ----------------------------- Reporting ------------------------------------


class ReportConfig(_StrictModel):
    enabled: bool = True
    title: str = "EcoFoundation Report"
    plotly_inline: bool = Field(
        True, description="If True, embed plotly.js inline (offline-readable, ~3 MB)."
    )
    spatial_max_points: int = Field(
        50_000, description="Above this, spatial plots use a sampled overlay."
    )


# ----------------------------- Top-level ------------------------------------


PipelineName = Literal["clustering", "niche_classification"]


class PipelineConfig(_StrictModel):
    """Toggle which sub-pipelines a run executes."""

    name: PipelineName = "clustering"


class RunConfig(_StrictModel):
    """The full run configuration. Top-level object loaded from YAML."""

    run_name: str = "run"
    run_dir: Path = Field(Path("runs"), description="Parent dir; an auto-named subfolder is created.")
    seed: int = 0
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    data: DataConfig
    qc: QCConfig = Field(default_factory=QCConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    hvg: HVGConfig = Field(default_factory=HVGConfig)
    leiden: LeidenConfig = Field(default_factory=LeidenConfig)
    markers: MarkerGenesConfig = Field(default_factory=MarkerGenesConfig)
    niches: NicheConfig = Field(default_factory=NicheConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig | None = Field(
        None,
        description="Required for the niche_classification pipeline; ignored otherwise.",
    )
    unsup: UnsupClusteringConfig | None = Field(
        None,
        description="Required for the unsupervised niche-clustering pipeline; ignored otherwise.",
    )
    report: ReportConfig = Field(default_factory=ReportConfig)

    @field_validator("run_dir", mode="before")
    @classmethod
    def _expand_run_dir(cls, v):  # noqa: D401
        return Path(v).expanduser()

    @model_validator(mode="after")
    def _check_disabled_combos(self):
        if self.hvg.enabled and not self.normalization.enabled:
            # Picking HVGs only makes sense on normalized data, but we don't hard-fail —
            # data may already be normalized in a layer.
            pass
        return self
