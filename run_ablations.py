"""Run the automated portion of ablation_analysis.ipynb without Jupyter.

The notebook remains the source of truth. Cells below the "Selected ablation
inspection" heading are intentionally interactive and are not executed here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "ablation_analysis.ipynb"
STOP_HEADING = "## Selected ablation inspection"


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def automated_code_cells(notebook: dict) -> list[tuple[int, str]]:
    code_cells: list[tuple[int, str]] = []
    found_stop = False

    for index, cell in enumerate(notebook.get("cells", [])):
        source = source_text(cell)
        if cell.get("cell_type") == "markdown" and STOP_HEADING in source:
            found_stop = True
            break
        if cell.get("cell_type") == "code" and source.strip():
            code_cells.append((index, source))

    if not found_stop:
        raise ValueError(
            f"Could not find notebook boundary {STOP_HEADING!r}; refusing to run inspection cells."
        )
    return code_cells


def main() -> int:
    if not NOTEBOOK_PATH.is_file():
        print(f"[error] Missing notebook: {NOTEBOOK_PATH}", file=sys.stderr)
        return 2

    # Plot files should be generated without opening blocking GUI windows.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(BASE_DIR)

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    namespace = {
        "__name__": "__ablation_pipeline__",
        "__file__": str(NOTEBOOK_PATH),
    }
    current_cell: int | None = None
    try:
        for index, source in automated_code_cells(notebook):
            current_cell = index
            print(f"[notebook] running code cell {index}", flush=True)
            exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"), namespace)
    except Exception as exc:
        location = f"code cell {current_cell}" if current_cell is not None else "notebook setup"
        print(f"[error] Ablation notebook failed during {location}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
