"""
Decision rules replayed on recorded paged walks, without new calls.

A rule can only decline an acceptance, never walk on past it: the blocks after
an acceptance were never asked, so declining is the one change the recorded
walk can answer exactly.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from schemaoutis import inventory

from . import scoring as report


def replay(columns: Sequence[Mapping[str, Any]], *, cap: int | None = None,
           target_free: bool = False) -> list[dict]:
    """
    One pair's paged columns with each rule-breaking acceptance declined.

    `cap` declines an acceptance at block `cap` or later. `target_free`
    declines a later acceptance of a target another column committed at
    block 0, which is only sound on 1:1 gold.
    """
    early = Counter(c["ranking"][0][0] for c in columns
                    if c.get("committed") and c.get("accepted_block") == 0
                    and c.get("ranking"))
    out = []
    for c in columns:
        c = copy.deepcopy(dict(c))
        k = c.get("accepted_block")
        late = bool(c.get("committed")) and k is not None and k >= 1
        drop = (bool(c.get("committed")) and k is not None
                and cap is not None and k >= cap)
        drop |= (late and target_free and bool(c.get("ranking"))
                 and early[c["ranking"][0][0]] > 0)
        if drop:
            c.update(committed=False, accepted=False, abstained=True,
                     ranking=[], stop_reason="replay_declined")
        out.append(c)
    return out


def score_replay(runs: Any, pairs: Sequence[dict], *, results: Path,
                 corpus: str, cap: int | None = None,
                 target_free: bool = False) -> dict[str, Any]:
    """`report.metrics` for one paged cell under the rules, plus calls per column."""
    by_name = report.by_pair_name(pairs)
    rows = runs.to_dict("records") if hasattr(runs, "to_dict") else list(runs)
    rankings: dict[str, list] = {}
    committed: dict[str, bool] = {}
    ground_truth: list[tuple[str, str]] = []
    calls = 0
    for row in rows:
        name = inventory._field(row, "pair")
        pair = by_name.get(name)
        if pair is None:
            continue
        columns = inventory.load(row)["trace"]["columns"]
        for c in replay(columns, cap=cap, target_free=target_free):
            key = report.namespaced(name, c["source_column"])
            rankings[key] = [(t, float(s)) for t, s in (c.get("ranking") or [])]
            committed[key] = bool(c.get("committed"))
            n = int(c.get("api_calls", 0) or 0)
            calls += min(n, cap) if cap is not None else n
        ground_truth += [(report.namespaced(name, s), t)
                         for s, t in pair["ground_truth"]]
    retriever = {inventory._field(r, "retriever") for r in rows}
    fallback: dict[str, list] = {}
    if len(retriever) == 1:
        fallback, _ = report.retrieval_rankings(
            pairs, results=results, corpus=corpus, retriever=retriever.pop(),
            covered=[inventory._field(r, "pair") for r in rows])
    row = report.metrics(rankings, ground_truth, committed, fallback=fallback)
    row["n_columns"] = len(rankings)
    row["calls_per_column"] = calls / max(len(rankings), 1)
    return row
