"""Collect package + dependency versions for run manifests."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

_TRACKED_PACKAGES = (
    "ecofoundation",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "anndata",
    "scanpy",
    "squidpy",
    "torch",
    "torch-geometric",
    "captum",
    "liana",
    "plotly",
    "pydantic",
    "loguru",
)


def collect_versions() -> dict[str, str]:
    """Return a dict of package -> version for the tracked dependency set."""
    info: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in _TRACKED_PACKAGES:
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = "not-installed"
    return info
