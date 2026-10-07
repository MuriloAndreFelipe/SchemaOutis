"""
`schemaoutis` imports neither `corpora` nor `evaluation`. Checked in a
subprocess, since pytest has already imported both.
"""

from __future__ import annotations

import subprocess
import sys

CHILD = """
import sys
import schemaoutis
from schemaoutis import bench, cache, inventory, run
leaked = sorted(m for m in ("corpora", "evaluation") if m in sys.modules)
print(",".join(leaked))
"""


def _imported_by_schemaoutis() -> list[str]:
    """Modules present after importing the library, in a clean interpreter."""
    out = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    return [m for m in out.stdout.strip().split(",") if m]


def test_the_library_imports_neither_the_corpora_nor_the_metrics():
    """
    A subprocess, not an assertion about `sys.modules` here: pytest has already
    imported both packages by the time any test runs, so an in-process check
    would pass no matter what the library does.
    """
    assert _imported_by_schemaoutis() == []


def test_the_metrics_may_import_the_library():
    """The arrow points this way, and `scoring` and `ablation` both use it."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; from evaluation import scoring, ablation; "
         "print('schemaoutis' in sys.modules)"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True"


def test_the_pipeline_package_is_gone():
    """Its modules moved into the two packages above; the name must not return."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util as u; print(u.find_spec('pipeline') is None)"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True"
