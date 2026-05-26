"""Niche-embedding extraction + UMAP.

The trained classifier's pooled graph embedding (``model._encode(...)`` → ``z``
of shape ``(B, hidden_dim)``) is the natural representation of a niche.
Computing it for every niche gives an embedding matrix we can:

  - cluster (Step 6 will reuse this idea with an unsupervised GAE),
  - or just visualise — UMAP-projected, coloured by predicted class,
    explained niches highlighted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ecofoundation.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class NicheEmbeddings:
    embeddings: np.ndarray  # (n_niches, hidden_dim)
    umap_2d: np.ndarray | None  # (n_niches, 2) or None if not computed
    niche_ids: np.ndarray  # (n_niches,)


def extract_niche_embeddings(
    model,
    graphs: list[Data],
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> NicheEmbeddings:
    """Return the pooled graph embedding for every niche."""
    model = model.to(device).eval()
    embs: list[np.ndarray] = []
    nids: list[int] = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model._encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embs.append(z.cpu().numpy())
            # PyG batches preserve `.niche_id` per-graph in lists/tensors.
            if hasattr(batch, "niche_id"):
                vals = batch.niche_id
                if torch.is_tensor(vals):
                    nids.extend(vals.cpu().numpy().tolist())
                else:
                    nids.extend(list(vals))
            else:
                nids.extend(range(len(embs[-1])))

    emb_mat = np.concatenate(embs, axis=0)
    _log.info(f"Extracted niche embeddings: {emb_mat.shape}")
    return NicheEmbeddings(embeddings=emb_mat, umap_2d=None, niche_ids=np.asarray(nids))


def compute_niche_umap(
    embeddings: NicheEmbeddings,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 0,
) -> NicheEmbeddings:
    """Add a 2-D UMAP projection to a :class:`NicheEmbeddings` (returns a new instance)."""
    from umap import UMAP

    _log.info(
        f"Running UMAP on {embeddings.embeddings.shape[0]} niches "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})"
    )
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=2,
    )
    coords = reducer.fit_transform(embeddings.embeddings)
    return NicheEmbeddings(
        embeddings=embeddings.embeddings,
        umap_2d=np.asarray(coords),
        niche_ids=embeddings.niche_ids,
    )
