"""Hello-World run script.

Usage
-----
From the repo root, with the venv activated::

    .venv/bin/python examples/01_hello_world.py

Runs the clustering pipeline on the user's scVI-annotated Xenium dataset
defined in ``configs/hello_world.yaml`` and produces a self-contained HTML
report under ``runs/<timestamp>__hello_world__<hash>/report.html``.
"""

from __future__ import annotations

from pathlib import Path

from ecofoundation.pipelines import run_from_yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hello_world.yaml"


def main() -> None:
    folder = run_from_yaml(CONFIG_PATH)
    print()
    print("=" * 60)
    print(f"Run ID   : {folder.run_id}")
    print(f"Folder   : {folder.root}")
    print(f"Report   : {folder.report_path}")
    print(f"Manifest : {folder.manifest_path}")
    print(f"Log      : {folder.log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
