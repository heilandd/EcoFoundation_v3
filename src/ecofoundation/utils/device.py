"""Device detection across CUDA, Apple Silicon MPS, and CPU."""

from __future__ import annotations

from typing import Literal

DeviceKind = Literal["cuda", "mps", "cpu"]


def resolve_device(prefer: str | None = None) -> str:
    """Return a torch device string.

    Resolution order when ``prefer`` is ``"auto"`` or ``None``:
      1. CUDA if available
      2. MPS (Apple Silicon) if available and usable
      3. CPU

    Parameters
    ----------
    prefer
        ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` (== ``"auto"``).
        An explicit choice that is unavailable falls back to CPU with a warning.
    """
    try:
        import torch
    except ImportError as exc:  # surface clearly
        raise RuntimeError("PyTorch is required for device resolution.") from exc

    pref = (prefer or "auto").lower()

    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return _fallback("cuda not available", "cpu")

    if pref == "mps":
        if _mps_usable(torch):
            return "mps"
        return _fallback("mps not available", "cpu")

    if pref == "cpu":
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    if _mps_usable(torch):
        return "mps"
    return "cpu"


def _mps_usable(torch_mod) -> bool:
    """MPS is only usable on Apple Silicon with a recent torch build."""
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is None:
        return False
    return bool(getattr(mps, "is_available", lambda: False)())


def _fallback(reason: str, fallback: str) -> str:
    from ecofoundation.utils.logging import get_logger

    get_logger(__name__).warning(f"Device fallback: {reason}. Using {fallback!r} instead.")
    return fallback
