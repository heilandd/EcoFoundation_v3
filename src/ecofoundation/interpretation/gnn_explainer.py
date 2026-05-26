"""PyG GNNExplainer wrapper for per-niche explanations.

Produces, per niche:
  - ``node_mask`` — (n_nodes, n_node_features) attribution score on each gene
    of each cell. Used to identify which CELLS and which GENES drove the call.
  - ``edge_mask`` — (n_edges,) attribution score on each edge. Used to identify
    which cell-cell connections drove the call.

Edge-feature attribution (which channel — distance vs LR — matters) is handled
via Integrated Gradients in ``integrated_gradients.py`` because GNNExplainer's
edge_attr_mask support is limited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig

from ecofoundation.interpretation.wrapper import SingleTargetWrapper
from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class NicheExplanation:
    """Per-niche explanation outputs."""

    niche_id: int
    target_class: int  # the class being explained (for categorical targets)
    node_mask: np.ndarray  # (n_nodes, n_features) or (n_nodes,) — depending on mask_type
    edge_mask: np.ndarray  # (n_edges,)
    target_prob: float | None  # model's predicted probability for ``target_class``


def explain_niche(
    model_wrapper: SingleTargetWrapper,
    data: Data,
    *,
    target_class: int,
    epochs: int = 200,
    lr: float = 0.01,
    n_classes: int,
    device: str = "cpu",
    edge_mask: bool = False,
) -> NicheExplanation:
    """Run GNNExplainer on a single niche.

    Parameters
    ----------
    edge_mask
        If True, also learn a soft edge mask. PyG 2.5 has shape-mismatch issues
        when combining edge_mask with edge_attr in graph-level classification
        (multi-channel ``edge_attr`` causes a broadcast error in the explainer's
        internal forward), so we leave this off by default and use Integrated
        Gradients for edge attribution. Edge importance for plotting falls back
        to ``|edge_attr|`` summed over channels.
    """
    model_wrapper.to(device).eval()
    data = data.to(device)

    explainer = Explainer(
        model=model_wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object" if edge_mask else None,
        model_config=ModelConfig(
            mode="multiclass_classification" if n_classes >= 2 else "regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        target=torch.tensor([target_class], device=device),
        edge_attr=data.edge_attr if data.edge_attr is not None else None,
        batch=torch.zeros(data.num_nodes, dtype=torch.long, device=device),
    )

    nm_attr = getattr(explanation, "node_mask", None)
    em_attr = getattr(explanation, "edge_mask", None)
    node_mask_arr = nm_attr.detach().cpu().numpy() if nm_attr is not None else None
    edge_mask_arr = em_attr.detach().cpu().numpy() if em_attr is not None else None

    # Fallback edge importance when learned mask is disabled.
    if edge_mask_arr is None:
        if data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_mask_arr = np.linalg.norm(
                data.edge_attr.detach().cpu().numpy(), axis=1
            )
        else:
            edge_mask_arr = np.zeros(data.edge_index.shape[1], dtype=np.float32)

    # Also probe the model for the target probability
    with torch.no_grad():
        logits = model_wrapper(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=torch.zeros(data.num_nodes, dtype=torch.long, device=device),
        )
        if logits.ndim == 2 and logits.shape[1] >= 2:
            prob = float(torch.softmax(logits, dim=-1)[0, target_class].item())
        else:
            prob = float(logits.flatten()[0].item())

    return NicheExplanation(
        niche_id=int(getattr(data, "niche_id", -1)),
        target_class=target_class,
        node_mask=node_mask_arr if node_mask_arr is not None else np.zeros((data.num_nodes,)),
        edge_mask=edge_mask_arr,
        target_prob=prob,
    )
