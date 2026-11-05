"""Make the harness importable the way it imports itself.

``benchmarks/harness/report_format.py``, ``report_sections.py`` and ``reporter.py`` import
their sibling as the bare package ``harness`` (``from harness.protocol import ...``), because
``benchmarks/run_benchmark.py`` puts its own directory on ``sys.path`` before using them.
A test that renders a document imports those presentation modules directly, without going
through the CLI, so it has to mirror that one sys.path entry; ``test/conftest.py`` only puts
the checkout root on the path (for ``benchmarks.harness.protocol``), never ``benchmarks/``.
Appended, never prepended, so it cannot shadow ``src/``, ``test/`` or the checkout root.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The harness's own directory: ``<repo>/benchmarks`` (this file is ``<repo>/test/unit/benchmarks``).
BENCHMARKS_DIR = Path(__file__).resolve().parents[3] / "benchmarks"

_entry = str(BENCHMARKS_DIR)
if _entry not in sys.path:
    sys.path.append(_entry)
