"""LR scoring math."""

from __future__ import annotations

import numpy as np

from ecofoundation.graph.lr_scoring import LRResource, score_edges_lr


def test_score_zero_for_empty_resource():
    expr = np.ones((5, 10), dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    empty_lr = LRResource(
        ligand_gene_idx=np.empty(0, dtype=np.int64),
        receptor_gene_idx=np.empty(0, dtype=np.int64),
        ligand_names=[],
        receptor_names=[],
        n_pairs_total=10,
        n_pairs_kept=0,
    )
    scores = score_edges_lr(expr, edges, empty_lr)
    assert scores.shape == (2,)
    assert (scores == 0).all()


def test_score_symmetric_and_positive():
    # 4 cells, 6 genes. Cell 0 expresses gene-0 (ligand 1), cell 1 expresses gene-1 (receptor 1).
    expr = np.zeros((4, 6), dtype=np.float32)
    expr[0, 0] = 2.0  # ligand
    expr[1, 1] = 3.0  # receptor
    lr = LRResource(
        ligand_gene_idx=np.array([0]),
        receptor_gene_idx=np.array([1]),
        ligand_names=["L"],
        receptor_names=["R"],
        n_pairs_total=1,
        n_pairs_kept=1,
    )
    edges = np.array([[0, 1]], dtype=np.int64)
    scores = score_edges_lr(expr, edges, lr)
    # fwd: L[0]*R[1] = 2*3 = 6; bwd: L[1]*R[0] = 0 → mean=6/1+0/1=6
    assert np.isclose(scores[0], 6.0)


def test_score_no_edges():
    expr = np.ones((3, 4), dtype=np.float32)
    edges = np.empty((0, 2), dtype=np.int64)
    lr = LRResource(
        ligand_gene_idx=np.array([0]),
        receptor_gene_idx=np.array([1]),
        ligand_names=["L"],
        receptor_names=["R"],
        n_pairs_total=1,
        n_pairs_kept=1,
    )
    scores = score_edges_lr(expr, edges, lr)
    assert scores.shape == (0,)
