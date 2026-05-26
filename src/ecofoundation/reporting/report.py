"""Report builder — accumulates sections and renders one self-contained HTML file.

Plots are matplotlib figures (PDF backend, Helvetica, fonttype=42), embedded
inline as SVG. When ``pdf_dir`` is set on the builder, every PlotSection's
figure is additionally saved as a PDF artefact in that directory so the user
can open it in Illustrator and edit individual glyphs / strokes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2

from ecofoundation.config.schemas import RunConfig
from ecofoundation.reporting.sections import (
    HTMLSection,
    OverviewSection,
    PlotSection,
    ReportSection,
    TableSection,
    TextSection,
)
from ecofoundation.reporting.style import save_pdf
from ecofoundation.utils.logging import get_logger
from ecofoundation.utils.versioning import collect_versions

_log = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _slug(s: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in s.lower().strip()]
    return "".join(keep)[:80] or "section"


@dataclass
class ReportBuilder:
    """Accumulator pattern: pipeline pushes sections; final ``write`` emits HTML."""

    run_name: str
    cfg: RunConfig
    pdf_dir: Path | None = None
    sections: list[ReportSection] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ----- adding sections ---------------------------------------------------

    def add(self, section: ReportSection) -> None:
        self.sections.append(section)

    def add_overview(self, items: dict[str, Any], *, title: str = "Overview") -> None:
        self.add(OverviewSection(title=title, items=items))

    def add_text(
        self,
        title: str,
        body: str,
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.add(
            TextSection(
                title=title,
                description=description,
                parameters=parameters or {},
                body=body,
            )
        )

    def add_table(
        self,
        title: str,
        rows: list[dict[str, Any]],
        *,
        columns: list[str] | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        max_rows: int = 200,
    ) -> None:
        self.add(
            TableSection(
                title=title,
                description=description,
                parameters=parameters or {},
                rows=rows,
                columns=columns,
                max_rows=max_rows,
            )
        )

    def add_plot(
        self,
        title: str,
        figure: Any,  # matplotlib.figure.Figure
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        pdf_path: Path | None = None
        if self.pdf_dir is not None and figure is not None:
            n = len([s for s in self.sections if isinstance(s, PlotSection)])
            pdf_path = save_pdf(figure, self.pdf_dir / f"{n:02d}_{_slug(title)}.pdf")
        self.add(
            PlotSection(
                title=title,
                description=description,
                parameters=parameters or {},
                figure=figure,
                pdf_path=pdf_path,
            )
        )

    def add_html(self, title: str, html: str, *, description: str = "") -> None:
        self.add(HTMLSection(title=title, description=description, html=html))

    # ----- rendering ---------------------------------------------------------

    def render(self) -> str:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("report.html.j2")

        rendered_sections: list[dict[str, Any]] = []
        for sec in self.sections:
            rendered_sections.append(
                {
                    "title": sec.title,
                    "description": sec.description,
                    "kind": sec.kind,
                    "parameters": sec.parameters,
                    "html": sec.render(),
                }
            )

        ctx = {
            "title": self.cfg.report.title,
            "run_name": self.run_name,
            "created_at": self.created_at,
            "config_dump": self.cfg.model_dump(mode="json"),
            "versions": collect_versions(),
            "sections": rendered_sections,
            "pdf_dir_relative": (
                str(self.pdf_dir.name) if self.pdf_dir is not None else None
            ),
        }
        return template.render(**ctx)

    def write(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html = self.render()
        out.write_text(html, encoding="utf-8")
        _log.info(
            f"Wrote report: {out} ({len(html)/1024:.1f} KB, {len(self.sections)} sections)"
        )
        return out
