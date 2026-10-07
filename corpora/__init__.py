"""
Loading the benchmark corpora (GDC and Valentine).

A pair is a plain dict, so a hand-written dict in a test works as a corpus:

    {"name": str,
     "source_df": DataFrame,
     "target_df": DataFrame,
     "ground_truth": [(source_column, target_column), ...]}

`schemaoutis` does not import this package.
"""

from .registry import CORPORA, load, load_one, resolve_root

__all__ = ["CORPORA", "load", "load_one", "resolve_root"]
