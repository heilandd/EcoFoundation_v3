"""Unsupervised training loop (GAE / VGAE / DGI).

One :func:`train_unsup` call iterates the chosen self-supervised loss
over batched PyG graphs. The trained model is returned along with an
epoch-level loss history.

When an :class:`AdversarialBatchHead` is provided, every batch also
incurs a CE loss on a nuisance label (``batch.y_adv``) routed through a
Gradient Reversal Layer — pushing the encoder toward batch-invariant
features (Step 6.5).

After training, :func:`encode_niche_embeddings` runs the encoder + pooling
across all niches and returns a ``(n_niches, hidden_dim)`` matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool

from ecofoundation.config.schemas import UnsupClusteringConfig
from ecofoundation.models.adversarial import AdversarialBatchHead, lambda_schedule
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


ADV_BATCH_ATTR = "y_adv"


@dataclass
class UnsupTrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    last_loss: float = float("nan")
    n_epochs_run: int = 0


def train_unsup(
    model: nn.Module,
    graphs: list[Data],
    cfg: UnsupClusteringConfig,
    *,
    device: str = "cpu",
    adv_head: AdversarialBatchHead | None = None,
) -> UnsupTrainResult:
    """Train the unsupervised model on ``graphs``. Returns loss history.

    When ``adv_head`` is provided, an adversarial CE loss on ``batch.y_adv``
    is added to the main self-supervised loss. The GRL inside ``adv_head``
    negates the gradient that reaches the encoder, so the encoder learns
    features that the adv head cannot use to identify the batch.
    """
    model = model.to(device)
    model.train()
    params = list(model.parameters())
    if adv_head is not None:
        adv_head = adv_head.to(device)
        adv_head.train()
        params = params + list(adv_head.parameters())
    loader = DataLoader(graphs, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history: list[dict[str, float]] = []
    last = float("nan")
    for epoch in range(1, cfg.epochs + 1):
        lam = (
            lambda_schedule(
                epoch - 1,
                lambda_max=cfg.adversarial.lambda_max,
                warmup_epochs=cfg.adversarial.warmup_epochs,
            )
            if adv_head is not None
            else 0.0
        )
        epoch_loss = 0.0
        epoch_adv_loss = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            main, adv = _compute_loss(model, batch, cfg, adv_head=adv_head, adv_lambda=lam)
            loss = main + adv
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_adv_loss += float(adv.detach().cpu()) if torch.is_tensor(adv) else 0.0
            n += 1
        last = epoch_loss / max(n, 1)
        history.append(
            {"epoch": epoch, "loss": last, "adv_loss": epoch_adv_loss / max(n, 1), "lambda": lam}
        )
        _log.info(
            f"[unsup] epoch {epoch:3d} | loss={last:.4f} "
            f"adv={epoch_adv_loss / max(n,1):.4f} lambda={lam:.3f}"
        )
    return UnsupTrainResult(history=history, last_loss=last, n_epochs_run=cfg.epochs)


def _compute_loss(
    model: nn.Module,
    batch,
    cfg: UnsupClusteringConfig,
    *,
    adv_head: AdversarialBatchHead | None,
    adv_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor | float]:
    """Return ``(main_loss, adv_loss_or_0)``. Dispatch by model class."""
    cls = type(model).__name__
    if cls == "GAE":
        z = model.encode(batch.x, batch.edge_index, batch.edge_attr)
        main = model.reconstruction_loss(
            z, batch.edge_index, cfg.n_neg_samples_per_edge, batch.batch
        )
    elif cls == "VGAE":
        z = model.encode(batch.x, batch.edge_index, batch.edge_attr)
        recon = model.reconstruction_loss(
            z, batch.edge_index, cfg.n_neg_samples_per_edge, batch.batch
        )
        main = recon + model.kl_term()
    elif cls == "DGI":
        # DGI's loss computes its own internal encode. We do an extra encode
        # here only when adv is enabled (small cost on niche-sized graphs).
        main = model.loss(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if adv_head is not None and adv_lambda > 0 and hasattr(batch, ADV_BATCH_ATTR):
            z = model.encode(batch.x, batch.edge_index, batch.edge_attr)
        else:
            z = None
    else:
        raise TypeError(f"Unsupported unsupervised model class: {cls}")

    adv = torch.tensor(0.0, device=main.device)
    if adv_head is not None and adv_lambda > 0 and hasattr(batch, ADV_BATCH_ATTR):
        if cls in ("GAE", "VGAE"):
            z_graph = global_mean_pool(z, batch.batch)
        else:
            z_graph = global_mean_pool(z, batch.batch)
        adv_logits = adv_head(z_graph, adv_lambda)
        adv = F.cross_entropy(adv_logits, getattr(batch, ADV_BATCH_ATTR))

    return main, adv


def attach_adv_labels(graphs: list[Data], adv_batch_labels: np.ndarray) -> list[Data]:
    """Attach the per-niche adversarial label to each Data object (in-place clone)."""
    out: list[Data] = []
    for i, g in enumerate(graphs):
        gn = g.clone()
        gn[ADV_BATCH_ATTR] = torch.tensor([int(adv_batch_labels[i])], dtype=torch.long)
        out.append(gn)
    return out


def encode_niche_embeddings(
    model: nn.Module,
    graphs: list[Data],
    *,
    batch_size: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the encoder + pooling over all graphs.

    Returns
    -------
    embeddings
        ``(n_niches, hidden_dim)``.
    niche_ids
        ``(n_niches,)``.
    """
    model = model.to(device).eval()
    pool_fn = global_mean_pool if pooling == "mean" else global_max_pool

    embs: list[np.ndarray] = []
    nids: list[int] = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(batch.x, batch.edge_index, batch.edge_attr)
            z = pool_fn(h, batch.batch)
            embs.append(z.cpu().numpy())
            if hasattr(batch, "niche_id"):
                vals = batch.niche_id
                if torch.is_tensor(vals):
                    nids.extend(vals.cpu().numpy().tolist())
                else:
                    nids.extend(list(vals))
            else:
                nids.extend(range(len(embs[-1])))
    return np.concatenate(embs, axis=0), np.asarray(nids, dtype=np.int64)
