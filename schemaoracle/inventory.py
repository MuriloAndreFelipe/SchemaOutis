"""The runs on disk, as one table.

`check` drops runs that are not safe to score, most importantly runs whose
columns ended in an API error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .resume import errored_columns, fallback_columns, retryable_columns

from . import cache

COLUMNS = ["path", "kind", "pair", "retriever", "arm", "provider", "window",
           "threshold", "top_k", "prune", "rep", "corpus"]


def _field(row, name, default=None):
    """A field from an itertuples row, a dict or a Series."""
    if hasattr(row, name):
        return getattr(row, name)
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        return default


def _roots(results) -> list[Path]:
    if isinstance(results, (str, Path)):
        return [Path(results)]
    return [Path(r) for r in results]


def scan(results, corpus: str = "gdc",
         retrievers: Iterable[str] | None = None,
         kinds: Iterable[str] = ("retrieval", "rerank", "gated", "paged"),
         pairs: Iterable[str] | None = None,
         ) -> pd.DataFrame:
    """
    Every run of the requested kinds under `results`, one directory or several.
    """
    kinds = tuple(kinds)
    rows: list[dict[str, Any]] = []

    # Filenames are sanitised, so "valentine/opendata" is on disk as
    # "valentine-opendata".
    wanted_corpus = cache.fs_safe(corpus)

    for root in _roots(results):
        for f in sorted(root.glob("*.json")):
            g = cache.parse_stem(f.stem)
            if (g is None or g.get("corpus") != wanted_corpus
                    or g["kind"] not in kinds):
                continue
            provider, arm = cache.split_arm(g.pop("arm", ""))
            rows.append({"path": f, "window": 20, "threshold": 0.0,
                         "top_k": 20, "rep": 1, "prune": "",
                         **g, "arm": arm, "provider": provider})

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    inv = pd.DataFrame(rows).reindex(columns=COLUMNS)
    if retrievers:
        inv = inv[inv["retriever"].isin(set(retrievers))]
    if pairs is not None:
        inv = inv[inv["pair"].isin({cache.fs_safe(p) for p in pairs})]
    return inv.reset_index(drop=True)


# Share of a run's asked columns that may be answered by the retriever. On GDC
# the rate is bimodal (almost every run is at zero and none sits between 0.1%
# and 2%), so this separates the two groups rather than tuning anything.
MAX_FALLBACK = 0.02


def check(inv: pd.DataFrame, verbose: bool = True,
          max_fallback: float = MAX_FALLBACK,
          drop_errored: bool = True) -> pd.DataFrame:
    """
    Drop runs that are not safe to score, reading every payload once.

    A run is dropped if a column ended in an API error (unless `drop_errored`
    is False) or if more than `max_fallback` of its columns were answered by
    the retriever. Callers deciding what still needs sending pass
    `drop_errored=False, max_fallback=1.0`, so a run with one failed column is
    resumed rather than quoted in full.

    Adds `n_columns`, `n_fallback`, `n_retryable` (API errors) and `n_stuck`
    (unreadable replies, which a retry would not change).
    """
    ok, bad, drowned, sizes, retry = [], [], [], {}, {}
    for r in inv.itertuples():
        payload = load(r)
        columns = payload.get("trace", {}).get("columns", [])
        sizes[r.Index] = len(columns)
        broken = errored_columns(payload)
        n_retry = len(retryable_columns(payload))
        retry[r.Index] = (n_retry, len(broken) - n_retry)
        if broken and drop_errored:
            bad.append((r.Index, len(broken)))
            continue
        # Pruned columns were never asked, so they are left out of the share.
        asked = [c for c in columns if c.get("stop_reason") != "pruned"]
        n_fb = len(fallback_columns(payload))
        share = n_fb / len(asked) if asked else 0.0
        if share > max_fallback:
            drowned.append((r.Index, n_fb, len(asked), share))
        else:
            ok.append((r.Index, n_fb))

    def report(rows, headline, line):
        if not rows or not verbose:
            return
        print(headline)
        for row in rows[:10]:
            r = inv.loc[row[0]]
            print(f"  {r['kind']}/{r['retriever']}/{r['pair']} "
                  f"th={r['threshold']:.2f} {r['arm']}: {line(row)}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more")

    report(bad, f"{len(bad)} run(s) have columns that ended in an API error "
                "or an unreadable reply and are excluded:",
           lambda row: f"{row[1]} column(s)")
    report(drowned,
           f"{len(drowned)} run(s) answer more than {max_fallback:.0%} of their "
           "columns from the retriever and are excluded:",
           lambda row: f"{row[1]}/{row[2]} column(s), {row[3]:.1%}")

    out = inv.loc[[i for i, _ in ok]].copy()
    out["n_columns"] = [sizes[i] for i, _ in ok]
    out["n_fallback"] = [n for _, n in ok]
    out["n_retryable"] = [retry[i][0] for i, _ in ok]
    out["n_stuck"] = [retry[i][1] for i, _ in ok]
    return out.reset_index(drop=True)


# What makes two runs the same configuration, per arm. `rep` is not in it.
KEYS = {
    "retrieval": ["retriever"],
    "rerank": ["retriever", "provider", "arm", "top_k", "prune"],
    "gated": ["retriever", "provider", "arm", "top_k", "prune"],
    "paged": ["retriever", "provider", "arm", "window", "threshold", "prune"],
}


def coverage(inv: pd.DataFrame, pairs: Iterable[str], *, kind: str,
             reps: int = 1, max_names: int = 8) -> pd.DataFrame:
    """
    One row per configuration: how many (pair, repeat) runs are done and how
    many are left. `missing` lists at most `max_names` pairs.
    """
    pairs = list(pairs)
    want = {(p, r) for p in pairs for r in range(1, int(reps) + 1)}
    keys = KEYS[kind]

    rows = []
    for values, runs in inv[inv["kind"] == kind].groupby(keys, dropna=False):
        have = {(p, int(r)) for p, r in zip(runs["pair"], runs["rep"])}
        short = sorted({p for p, _ in want - have})
        shown = short[:max_names]
        if len(short) > max_names:
            shown = shown + [f"... and {len(short) - max_names} more"]
        rows.append({
            **dict(zip(keys, values if isinstance(values, tuple) else (values,))),
            "done": len(want & have), "todo": len(want - have),
            "extra": len(have - want),      # reps or studies beyond what is asked
            "n_missing": len(short), "missing": shown,
        })
    if not rows:
        return pd.DataFrame(columns=[*keys, "done", "todo", "extra",
                                     "n_missing", "missing"])
    return (pd.DataFrame(rows)
            .sort_values(["done", *keys], ascending=False)
            .reset_index(drop=True))


def todo(inv: pd.DataFrame, pairs: Iterable[str], *, kind: str, reps: int = 1,
         **config) -> list[tuple[str, int]]:
    """
    The (pair, rep) runs of one configuration not yet on disk. Keys left out of
    `config` are unconstrained.
    """
    pairs = [cache.fs_safe(p) for p in pairs]
    want = [(p, r) for p in pairs for r in range(1, int(reps) + 1)]
    rows = inv[inv["kind"] == kind]
    for key, value in config.items():
        if key in rows.columns:
            rows = rows[rows[key] == value]
    have = {(p, int(r)) for p, r in zip(rows["pair"], rows["rep"])}
    return [pr for pr in want if pr not in have]


def short(inv: pd.DataFrame, columns: Mapping[str, int], *, kind: str,
          reps: int = 1, **config) -> dict[tuple[str, int], int]:
    """
    Runs on disk that hold fewer columns than their study, and how many fewer.
    `todo` counts these as done, so a cost estimate has to add them.
    """
    rows = inv[inv["kind"] == kind]
    for key, value in config.items():
        if key in rows.columns:
            rows = rows[rows[key] == value]
    if "n_columns" not in rows.columns:
        return {}
    columns = {cache.fs_safe(p): n for p, n in columns.items()}
    out = {}
    for pair, rep, n in zip(rows["pair"], rows["rep"], rows["n_columns"]):
        want = columns.get(pair)
        if want is not None and int(rep) <= int(reps) and int(n) < want:
            out[(pair, int(rep))] = want - int(n)
    return out


def load(row) -> dict[str, Any]:
    """One run's payload, as it was written."""
    return json.loads(Path(_field(row, "path")).read_text(encoding="utf-8"))
