"""Shared helpers for the result notebooks: project paths, loading results, and the figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.report_figures.make_results_figures as figures  # noqa: E402

figures.INLINE = True     # show figures in the notebook instead of writing them to report/


def load(relative_path: str) -> dict:
    """Read a saved result file, given its path from the project root."""
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
