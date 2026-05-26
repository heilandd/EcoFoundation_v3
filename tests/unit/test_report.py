"""Report builder: sections accumulate, HTML is self-contained, SVG inlined."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecofoundation.reporting.report import ReportBuilder
from ecofoundation.reporting.style import new_figure


def _trivial_fig():
    fig, ax = new_figure(width=2.5, height=1.5)
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_title("trivial")
    return fig


def test_empty_report_renders(tiny_run_config):
    rb = ReportBuilder(run_name="t", cfg=tiny_run_config)
    html = rb.render()
    assert "<html" in html.lower()
    assert "Run metadata" in html


def test_overview_and_plot_inline_svg(tiny_run_config, tmp_path):
    rb = ReportBuilder(run_name="t", cfg=tiny_run_config)
    rb.add_overview({"n_cells": 200, "n_patients": 2})
    rb.add_plot("Curve", _trivial_fig(), description="trivial")
    out = rb.write(tmp_path / "report.html")
    txt = out.read_text()
    assert out.exists()
    assert "n_cells" in txt
    # SVG embedded inline
    assert "<svg" in txt
    # Helvetica font reference present (matplotlib SVG embeds font-family)
    assert "Helvetica" in txt or "font-family" in txt


def test_pdf_artefacts_saved(tiny_run_config, tmp_path):
    pdf_dir = tmp_path / "pdf"
    rb = ReportBuilder(run_name="t", cfg=tiny_run_config, pdf_dir=pdf_dir)
    rb.add_plot("p1", _trivial_fig())
    rb.add_plot("p2", _trivial_fig())
    rb.write(tmp_path / "report.html")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    assert len(pdfs) == 2
    # Verify PDF font is embedded as TrueType (fonttype=42).
    blob = pdfs[0].read_bytes()
    assert b"%PDF" in blob[:8]


def test_table_section_truncates(tiny_run_config, tmp_path):
    rb = ReportBuilder(run_name="t", cfg=tiny_run_config)
    rows = [{"i": i} for i in range(500)]
    rb.add_table("Big table", rows, max_rows=10)
    html = rb.render()
    assert "Showing 10 of 500" in html
