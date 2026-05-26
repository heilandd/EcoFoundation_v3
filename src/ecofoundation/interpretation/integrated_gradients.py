"""Integrated Gradients for node and edge features via Captum.

Captum's `IntegratedGradients` attributes the model output back to an input.
We use it to attribute to:

  - ``edge_attr`` — yields per-channel importance per edge (e.g. ``distance``
    vs ``lr_score_tier1``). Aggregated → "which edge feature matters more".
  - ``x`` — already partially covered by GNNExplainer with ``node_mask_type=
    'attributes'``, but IG gives a complementary, gradient-based view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch_geometric.data import Data

from ecofoundation.interpretation.wrapper import SingleTargetWrapper


@dataclass
class IGResult:
    niche_id: int
    target_class: int
    node_attrs: np.ndarray  # (n_nodes, n_node_features)
    edge_attr_attrs: np.ndarray | None  # (n_edges, n_edge_channels)


def ig_attribute_niche(
    model_wrapper: SingleTargetWrapper,
    data: Data,
    *,
    target_class: int,
    device: str = "cpu",
    steps: int = 32,
) -> IGResult:
    """Run IG on one niche, attributing both node features and edge features."""
    model_wrapper.to(device).eval()
    data = data.to(device)

    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    # --- IG over node features (x) -----------------------------------------
    def _node_forward(x):
        return model_wrapper(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=batch,
        )

    x = data.x.clone().detach().requires_grad_(True)
    ig_nodes = IntegratedGradients(_node_forward)
    node_baseline = torch.zeros_like(x)
    node_attrs = ig_nodes.attribute(
        x,
        baselines=node_baseline,
        target=int(target_class),
        n_steps=steps,
        # internal_batch_size=1 → one IG step per forward pass. Without this,
        # Captum stacks ``n_steps`` copies of ``x`` along dim 0 and the niche's
        # batch tensor (length n_nodes) no longer matches.
        internal_batch_size=1,
    )

    # --- IG over edge_attr -------------------------------------------------
    edge_attrs: torch.Tensor | None = None
    if data.edge_attr is not None and data.edge_attr.numel() > 0:
        ea = data.edge_attr.clone().detach().requires_grad_(True)

        def _edge_forward(edge_attr):
            return model_wrapper(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )

        ig_edges = IntegratedGradients(_edge_forward)
        edge_baseline = torch.zeros_like(ea)
        edge_attrs = ig_edges.attribute(
            ea,
            baselines=edge_baseline,
            target=int(target_class),
            n_steps=steps,
            internal_batch_size=1,
        )

    return IGResult(
        niche_id=int(getattr(data, "niche_id", -1)),
        target_class=int(target_class),
        node_attrs=node_attrs.detach().cpu().numpy(),
        edge_attr_attrs=edge_attrs.detach().cpu().numpy() if edge_attrs is not None else None,
    )
