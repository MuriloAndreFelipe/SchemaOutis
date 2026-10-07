"""
Where the corpora live, and how to read them.

Column names are never canonicalised: ground truth and DataFrame columns must
match by exact string. Asking for one Valentine pair reads only that pair.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

# Overridable so a notebook can point at a copy elsewhere.
DATA_ROOT_ENV = "SCHEMAOUTIS_DATA"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def resolve_root(data_root: str | Path | None = None) -> Path:
    """The corpora directory: the argument, then the env var, then the default."""
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get(DATA_ROOT_ENV)
    return Path(env) if env else DEFAULT_DATA_ROOT


def _gdc(root: Path) -> Iterator[dict]:
    """
    Every GDC study, against the one shared target schema.

        <root>/gdc/target-tables/gdc_unique_columns_concat_values.csv
        <root>/gdc/source-tables/<study>.csv
        <root>/gdc/ground-truth/<study>.csv
    """
    base = root / "gdc"
    target_path = base / "target-tables" / "gdc_unique_columns_concat_values.csv"
    if not target_path.exists():
        raise FileNotFoundError(
            f"GDC target table not found: {target_path}\n"
            f"Set {DATA_ROOT_ENV} or pass data_root= if the corpora live elsewhere."
        )
    # Read once: all ten studies share the same 736 target columns.
    target_df = pd.read_csv(target_path, low_memory=False)

    for gt_file in sorted((base / "ground-truth").glob("*.csv")):
        source_path = base / "source-tables" / gt_file.name
        if not source_path.exists():
            continue
        gt = pd.read_csv(gt_file).dropna()
        yield {
            "name": gt_file.stem,
            "source_df": pd.read_csv(source_path),
            "target_df": target_df,
            "ground_truth": list(gt.itertuples(index=False, name=None)),
        }


VALENTINE_SUBSETS = ("ChEMBL", "OpenData", "TPC-DI", "Wikidata", "Magellan")


def _valentine_mapping(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(m["source_column"], m["target_column"]) for m in data["matches"]]


def _valentine(root: Path, subset: str | None = None) -> Iterator[dict]:
    """
    Valentine pairs, optionally one subset. The pair name is the path relative
    to the corpus root (e.g. ChEMBL/Joinable/<pair>), unique across subsets.
    """
    base = root / "Valentine-datasets"
    search = base / subset if subset else base
    if not search.exists():
        raise FileNotFoundError(
            f"Valentine corpus not found: {search}\n"
            f"Set {DATA_ROOT_ENV} or pass data_root= if the corpora live elsewhere."
        )

    for mapping_path in sorted(search.rglob("*_mapping.json")):
        folder = mapping_path.parent
        src = folder / f"{folder.name}_source.csv"
        tgt = folder / f"{folder.name}_target.csv"
        if not (src.exists() and tgt.exists()):
            continue
        gt = _valentine_mapping(mapping_path)
        if not gt:
            continue
        yield {
            "name": str(folder.relative_to(base)).replace("\\", "/"),
            # Chunked type inference can read one column two ways.
            "source_df": pd.read_csv(src, low_memory=False),
            "target_df": pd.read_csv(tgt, low_memory=False),
            "ground_truth": gt,
        }


CORPORA = {
    "gdc": _gdc,
    "valentine": lambda root: _valentine(root, None),
    **{f"valentine/{s.lower()}": (lambda root, s=s: _valentine(root, s))
       for s in VALENTINE_SUBSETS},
}


def load(name: str, data_root: str | Path | None = None,
         names: list[str] | None = None) -> list[dict]:
    """Every pair in a corpus, or only those in `names`."""
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}. Available: {sorted(CORPORA)}")
    root = resolve_root(data_root)
    wanted = set(names) if names else None

    out = []
    for pair in CORPORA[name](root):
        if wanted is None or pair["name"] in wanted:
            out.append(pair)
            if wanted is not None and len(out) == len(wanted):
                break

    if wanted:
        found = {p["name"] for p in out}
        gap = sorted(wanted - found)
        if gap:
            raise KeyError(f"{name!r} has no pair(s) named {gap}")
    _warn_unreachable_gold(name, out)
    return out


def _warn_unreachable_gold(corpus: str, pairs: Sequence[dict]) -> None:
    """
    Warn about gold naming a source column the frame does not have.

    MRR divides by every gold source column and the precision counts only the
    ones that exist, so such a pair makes the two disagree and breaks the
    decomposition of the abstention metric. Two exist in valentine/wikidata.
    """
    missing = {}
    for pair in pairs:
        cols = {str(c) for c in pair["source_df"].columns}
        gap = sorted({str(s) for s, _ in pair["ground_truth"]} - cols)
        if gap:
            missing[pair["name"]] = gap
    if not missing:
        return
    detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(missing.items()))
    warnings.warn(
        f"{corpus}: ground truth names {sum(len(v) for v in missing.values())} "
        f"source column(s) that are not in the source frame ({detail}). No "
        "matcher can reach them, and they are counted by MRR but not by the "
        "precision, so the two use different denominators.",
        stacklevel=2)


def load_one(name: str, pair: str, data_root: str | Path | None = None) -> dict:
    """One pair, by name."""
    return load(name, data_root, names=[pair])[0]
