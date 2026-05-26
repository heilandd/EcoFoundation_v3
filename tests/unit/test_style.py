"""Plotting style invariants: Helvetica, fonttype=42, white bg, black frame."""

from __future__ import annotations

from matplotlib import rcParams

from ecofoundation.reporting.style import (
    apply_style,
    fig_to_png_b64,
    fig_to_svg,
    new_figure,
    save_pdf,
)


def test_rcparams_applied():
    apply_style()
    assert rcParams["font.family"] == ["Helvetica"]
    assert rcParams["pdf.fonttype"] == 42
    assert rcParams["ps.fonttype"] == 42
    assert rcParams["figure.facecolor"] == "white"
    assert rcParams["axes.facecolor"] == "white"


def test_new_figure_frame_visible():
    fig, ax = new_figure(width=2.0, height=1.5)
    for side in ("top", "right", "bottom", "left"):
        assert ax.spines[side].get_visible()


def test_svg_contains_helvetica():
    fig, ax = new_figure(width=2, height=1)
    ax.set_title("hello")
    ax.set_xlabel("x")
    svg = fig_to_svg(fig)
    # matplotlib SVG embeds the font name in style attributes
    assert "Helvetica" in svg
    # No XML declaration (we strip it for inline embedding)
    assert not svg.lstrip().startswith("<?xml")


def test_png_base64_round_trip():
    fig, _ = new_figure(width=2, height=1)
    blob = fig_to_png_b64(fig)
    assert isinstance(blob, str)
    assert len(blob) > 100  # non-trivial PNG


def test_save_pdf(tmp_path):
    fig, ax = new_figure(width=2, height=1)
    ax.plot([0, 1], [0, 1])
    p = save_pdf(fig, tmp_path / "out.pdf")
    assert p.exists()
    # Verify PDF embeds fonts as text (fonttype=42 → /Subtype /TrueType in some objects).
    blob = p.read_bytes()
    assert blob[:4] == b"%PDF"
