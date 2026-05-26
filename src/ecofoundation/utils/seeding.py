"""Centralized random-seed management.

All randomness in EcoFoundation must derive from a single ``set_global_seed`` call
so runs are reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA + MPS).

    Parameters
    ----------
    seed
        Integer seed.
    deterministic_torch
        If True, enable PyTorch's deterministic-algorithms mode where supported.
        Some ops are not supported deterministically on MPS — they emit a warning.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError:
        pass
