"""Build Colab .ipynb files from the percent-format sources in colab/notebooks/src/ (stdlib only).

Run from the repo root: python3 colab/tools/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def _cells(text: str) -> list[dict]:
    cells, kind, lines = [], None, []

    def flush():
        body = "\n".join(lines).strip("\n")
        if kind and body:
            source = body.splitlines(keepends=True)
            if kind == "markdown":
                cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
            else:
                cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                              "source": source})

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        else:
            lines.append(line)
    flush()
    return cells


def build(src: Path, dst: Path) -> None:
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": _cells(Path(src).read_text(encoding="utf-8")),
    }
    Path(dst).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    for src in sorted((NOTEBOOKS / "src").glob("*.py")):
        dst = NOTEBOOKS / f"{src.stem}.ipynb"
        build(src, dst)
        print(f"built {dst.relative_to(NOTEBOOKS.parents[1])}")


if __name__ == "__main__":
    main()
