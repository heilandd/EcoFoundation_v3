"""Report sections.

A ``ReportSection`` is anything that knows how to render itself into an
HTML fragment that the Jinja2 template can embed.

Sections are dataclasses to keep them serializable and easy to compose.

Plots are now matplotlib figures (PDF-backend, Illustrator-editable). They
are embedded as inline SVG in the HTML report; a PDF artefact is saved
on disk alongside the run folder when the ReportBuilder is configured with
a ``pdf_dir``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportSection(ABC):
    """Abstract base. Subclasses implement :meth:`render`."""

    title: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def render(self) -> str:  # pragma: no cover - abstract
        """Return an HTML fragment for this section."""
        raise NotImplementedError

    @property
    def kind(self) -> str:
        return self.__class__.__name__


@dataclass
class TextSection(ReportSection):
    """Plain text block (newlines preserved as <br>)."""

    body: str = ""

    def render(self) -> str:
        return f"<div class='text-block'>{_html_escape_preserve_breaks(self.body)}</div>"


@dataclass
class TableSection(ReportSection):
    """A table rendered from a list of dicts or a pandas DataFrame."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] | None = None
    max_rows: int = 200

    def render(self) -> str:
        if not self.rows:
            return "<p><em>(empty table)</em></p>"
        truncated = len(self.rows) > self.max_rows
        rows = self.rows[: self.max_rows]
        cols = self.columns or list(rows[0].keys())
        head = "".join(f"<th>{_e(c)}</th>" for c in cols)
        body = "".join(
            "<tr>" + "".join(f"<td>{_e(r.get(c, ''))}</td>" for c in cols) + "</tr>"
            for r in rows
        )
        suffix = (
            f"<p class='note'>Showing {self.max_rows} of {len(self.rows)} rows.</p>"
            if truncated
            else ""
        )
        return (
            f"<table class='report-table'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>{suffix}"
        )


@dataclass
class PlotSection(ReportSection):
    """A matplotlib Figure embedded inline as SVG.

    Optionally a PDF artefact is saved to disk and the path recorded.
    """

    figure: Any = None  # matplotlib.figure.Figure
    pdf_path: Path | None = None

    def render(self) -> str:
        if self.figure is None:
            return "<p><em>(no figure)</em></p>"
        from ecofoundation.reporting.style import fig_to_svg

        svg = fig_to_svg(self.figure)
        link = (
            f"<p class='artifact-link'>PDF: <code>{_e(self.pdf_path.name)}</code></p>"
            if self.pdf_path is not None
            else ""
        )
        return f"<div class='plot-wrap'>{svg}{link}</div>"


@dataclass
class HTMLSection(ReportSection):
    """Raw HTML payload — escape hatch for custom content."""

    html: str = ""

    def render(self) -> str:
        return self.html


@dataclass
class OverviewSection(ReportSection):
    """Run-level overview rendered as a labeled key-value grid."""

    items: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        rows = "".join(
            f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in self.items.items()
        )
        return f"<dl class='overview-grid'>{rows}</dl>"


# --- helpers ----------------------------------------------------------------


def _e(x: Any) -> str:
    import html as _html

    return _html.escape(str(x))


def _html_escape_preserve_breaks(s: str) -> str:
    import html as _html

    return _html.escape(s).replace("\n", "<br>")
