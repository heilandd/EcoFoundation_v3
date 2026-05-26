"""Centralised matplotlib styling for EcoFoundation.

User-mandated defaults:

  - PDF backend (``matplotlib.use("pdf")``)
  - Helvetica font family
  - ``pdf.fonttype = 42`` (TrueType — glyphs editable as text in Illustrator)
  - White background, visible black axes frame
  - Default font size 6 pt (adjustable)

All plot helpers in ``ecofoundation.reporting.plots`` go through this module so
the style is consistent across the report and any standalone PDF artefacts.

Note: import order matters. ``apply_style`` must be called BEFORE pyplot is
used the first time by any module in the process. We call it eagerly at import
time of this module so that simply importing any plot helper applies the style.
"""

from __future__ import annotations

import base64
import io
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

# Set backend at import time so first plt.figure() doesn't pick a GUI backend
# in headless contexts. "pdf" is the user-requested default; the Agg-derived
# PDF backend produces font-embedded vector output.
matplotlib.use("pdf", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (after backend set)
from matplotlib import rcParams  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


DEFAULT_FONT_SIZE: float = 6.0
DEFAULT_FONT_FAMILY: str = "Helvetica"
DEFAULT_FRAME_LINEWIDTH: float = 0.5
DEFAULT_FRAME_COLOR: str = "black"


def apply_style(font_size: float = DEFAULT_FONT_SIZE) -> None:
    """Apply EcoFoundation's matplotlib rcParams.

    Idempotent — safe to call repeatedly. The font-size argument overrides the
    default 6 pt but does not change the family / fonttype / backend.
    """
    rcParams["font.family"] = DEFAULT_FONT_FAMILY
    rcParams["font.size"] = font_size
    # Embed fonts as text glyphs (Illustrator-editable), not curves.
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["svg.fonttype"] = "none"  # SVG: don't convert text to paths either

    # White background, visible black frame.
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    rcParams["axes.edgecolor"] = DEFAULT_FRAME_COLOR
    rcParams["axes.linewidth"] = DEFAULT_FRAME_LINEWIDTH
    rcParams["axes.spines.top"] = True
    rcParams["axes.spines.right"] = True
    rcParams["axes.spines.left"] = True
    rcParams["axes.spines.bottom"] = True

    # Consistent small ticks
    rcParams["xtick.major.size"] = 2.0
    rcParams["xtick.major.width"] = 0.5
    rcParams["ytick.major.size"] = 2.0
    rcParams["ytick.major.width"] = 0.5
    rcParams["xtick.minor.size"] = 1.0
    rcParams["xtick.minor.width"] = 0.4
    rcParams["ytick.minor.size"] = 1.0
    rcParams["ytick.minor.width"] = 0.4

    # Legend, grid, labels
    rcParams["legend.frameon"] = False
    rcParams["legend.fontsize"] = font_size
    rcParams["axes.titlesize"] = font_size
    rcParams["axes.labelsize"] = font_size
    rcParams["xtick.labelsize"] = font_size
    rcParams["ytick.labelsize"] = font_size
    rcParams["figure.dpi"] = 150
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = 0.05
    rcParams["lines.linewidth"] = 0.6


# Apply at import time so any subsequent figure construction inherits the style.
apply_style()


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def new_figure(
    width: float = 4.5,
    height: float = 3.0,
    *,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[Figure, Any]:
    """Create a Figure + Axes (or grid of Axes) with the project default style.

    Returns ``(fig, ax)`` for a single subplot, ``(fig, 1D ndarray of Axes)``
    for a row or column, ``(fig, 2D ndarray of Axes)`` for a grid.
    Matches matplotlib's ``squeeze=True`` convention.
    """
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),
        sharex=sharex,
        sharey=sharey,
        squeeze=True,
    )
    fig.patch.set_facecolor("white")
    return fig, axes


def style_axes(ax: Any, *, frame: bool = True) -> None:
    """Apply the project frame/face style to an axis after plotting."""
    ax.set_facecolor("white")
    if frame:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(DEFAULT_FRAME_COLOR)
            ax.spines[side].set_linewidth(DEFAULT_FRAME_LINEWIDTH)
    else:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def fig_to_svg(fig: Figure) -> str:
    """Render a Figure to an SVG string suitable for inline HTML embedding.

    Returns the SVG payload with the XML/DOCTYPE preamble stripped so it
    composes cleanly inside an HTML body.
    """
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    svg = buf.getvalue()
    # Strip XML declaration and doctype (browsers cope, but cleaner inline).
    if "<?xml" in svg:
        svg = svg[svg.find("<svg") :]
    return svg


def fig_to_png_b64(fig: Figure, dpi: int = 150) -> str:
    """Render a Figure to a base64-encoded PNG (for fallback HTML embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_pdf(fig: Figure, path: Path | str) -> Path:
    """Save a Figure as a PDF with fonts embedded as TrueType (Illustrator-editable)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf")
    return p


# ---------------------------------------------------------------------------
# Context manager for ad-hoc style overrides
# ---------------------------------------------------------------------------


@contextmanager
def temporary_style(**overrides: Any):
    """Temporarily set rcParams keys then restore them on exit."""
    snapshot = {k: rcParams[k] for k in overrides}
    try:
        for k, v in overrides.items():
            rcParams[k] = v
        yield
    finally:
        for k, v in snapshot.items():
            rcParams[k] = v
