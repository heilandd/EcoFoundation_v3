"""EcoFoundation — modular spatial transcriptomics analysis.

Two main components:
  * Unsupervised spatial clustering (classical + custom spatial-GNN)
  * Supervised GNN-based niche classification with explainability

Each analysis run produces a self-contained interactive HTML report.
"""

# IMPORTANT: this MUST run before torch is imported anywhere. Several PyG
# operators (scatter_reduce.two_out used by attention pooling, edge_softmax in
# GATv2, ...) are not implemented on MPS yet; the official fallback path is to
# silently dispatch those ops to CPU.
import os as _os

_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ecofoundation")
except PackageNotFoundError:  # not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
