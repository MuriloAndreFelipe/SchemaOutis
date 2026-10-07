"""Finished runs into the numbers the paper reports.

Reads `results/runs/` and writes nothing.

Studies are pooled, not averaged: rankings are namespaced by pair and scored
together, so Clark's 16 columns and Dou's 179 weigh what they contain.

Precision is reported twice. As committed, a declined column leaves the
denominator; as evaluated, every column contributes its rank-1 target. The gap
between them is what declining buys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .blocking import blocking_report
from .matching import (mrr, mrr_abstain, precision_recall,
                       recall_at_ground_truth, recall_at_ground_truth_bounds,
                       recall_at_k, top1_accuracy)

from schemaoutis import cache, inventory

# 1 is the headline, 20 the block size and rerank's ceiling, 10 in between.
DEFAULT_KS = (1, 10, 20)


def namespaced(pair: str, col: str) -> str:
    """A column name unique across studies, so pairs can be pooled."""
    return f"{pair}␟{col}"


def by_pair_name(pairs: Sequence[dict]) -> dict[str, dict]:
    """
    The corpus keyed as filenames spell pair names. A Valentine pair loaded as
    `OpenData/Joinable/...` is on disk as `OpenData-Joinable-...`.
    """
    return {cache.fs_safe(p["name"]): p for p in pairs}


def pool(runs: Iterable[Any], pairs: Sequence[dict]
         ) -> tuple[dict, list, dict, dict]:
    """
    Every run's columns as one (rankings, ground_truth, committed, abstained).
    Only the pairs a run covers are pooled. `abstained` is narrower than "not
    committed": a column whose reply could not be read is neither.
    """
    by_name = by_pair_name(pairs)
    rankings: dict[str, list] = {}
    committed: dict[str, bool] = {}
    abstained: dict[str, bool] = {}
    ground_truth: list[tuple[str, str]] = []
    seen: set[str] = set()

    for row in runs:
        pair_name = inventory._field(row, "pair")
        pair = by_name.get(pair_name)
        if pair is None or pair_name in seen:
            continue
        seen.add(pair_name)

        payload = inventory.load(row)
        for c in payload["trace"]["columns"]:
            key = namespaced(pair_name, c["source_column"])
            rankings[key] = [(t, float(s)) for t, s in (c.get("ranking") or [])]
            committed[key] = bool(c.get("committed"))
            abstained[key] = bool(c.get("abstained"))
        ground_truth += [(namespaced(pair_name, s), t)
                         for s, t in pair["ground_truth"]]

    return rankings, ground_truth, committed, abstained


def per_column(runs: Iterable[Any], pairs: Sequence[dict]) -> pd.DataFrame:
    """One row per source column: its cost and its outcome."""
    by_name = by_pair_name(pairs)
    rows = []
    seen: set[str] = set()
    for row in runs:
        pair_name = inventory._field(row, "pair")
        pair = by_name.get(pair_name)
        # Same guard as `pool`: a concatenated inventory can hold a pair twice,
        # and counting its calls twice while `n_columns` comes from `pool`
        # inflates every per-column cost.
        if pair is None or pair_name in seen:
            continue
        seen.add(pair_name)
        gold = {}
        for s, t in pair["ground_truth"]:
            gold.setdefault(s, set()).add(t)

        for c in inventory.load(row)["trace"]["columns"]:
            col = c["source_column"]
            ranked = [t for t, _ in (c.get("ranking") or [])]
            want = gold.get(col, set())
            rows.append({
                "pair": pair_name, "source_column": col,
                "api_calls": int(c.get("api_calls", 0) or 0),
                # A rerank call carries the whole shortlist and a paged call one
                # window, so tokens are the fairer cost.
                "tokens_in": int(c.get("tokens_in", 0) or 0),
                "tokens_out": int(c.get("tokens_out", 0) or 0),
                "wall_ms": float(c.get("wall_ms", 0) or 0),
                "accepted_block": c.get("accepted_block"),
                "n_blocks_available": c.get("n_blocks_available"),
                "stop_reason": c.get("stop_reason"),
                "committed": bool(c.get("committed")),
                "abstained": bool(c.get("abstained")),
                "has_gold": bool(want),
                "correct": bool(ranked[:1] and ranked[0] in want),
            })
    return pd.DataFrame(rows)


def cost_shape(cols: pd.DataFrame) -> dict[str, float]:
    """
    How unevenly calls fall across columns. `calls_share_top` is the share of
    all calls made by the most expensive tenth of columns.
    """
    if not len(cols) or not cols["api_calls"].sum():
        return {}
    calls = cols["api_calls"].sort_values(ascending=False)
    top = max(1, len(calls) // 10)
    with_gold = cols[cols["has_gold"]]
    without = cols[~cols["has_gold"]]
    return {
        "calls_median": float(calls.median()),
        "calls_max": int(calls.max()),
        "calls_share_top": float(calls.head(top).sum() / calls.sum()),
        "calls_with_gold": float(with_gold["api_calls"].mean())
                            if len(with_gold) else float("nan"),
        "calls_without_gold": float(without["api_calls"].mean())
                            if len(without) else float("nan"),
    }


def metrics(rankings: Mapping[str, list], ground_truth: Sequence[tuple],
            committed: Mapping[str, bool],
            ks: Sequence[int] = DEFAULT_KS,
            fallback: Mapping[str, list] | None = None,
            abstained: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """
    The metric columns of one row. Shared by `score` and `score_competitors`,
    so arms and baselines are computed the same way.

    Ranking metrics read only committed rankings, so a declined or pruned
    column scores zero. `fallback` fills an uncommitted column for the
    evaluated precision only. `abstained` marks the columns that declined, as
    against those whose reply could not be read: without it the two are one,
    and an unreadable reply is credited as a correct decline.
    """
    held = {k: (v if committed.get(k, False) else [])
            for k, v in rankings.items()}
    filled = {k: (v if committed.get(k, False)
                  else (fallback or {}).get(k) or v)
              for k, v in rankings.items()}
    row: dict[str, Any] = {f"recall@{k}": recall_at_k(held, ground_truth, k)
                           for k in ks}
    row["mrr"] = mrr(held, ground_truth)
    row["mrr_abstain"] = mrr_abstain(held, ground_truth, committed,
                                     abstained=abstained)
    row["top1"] = top1_accuracy(held, ground_truth)
    # Valentine's headline metric, with its tie-break interval.
    row["recall@gt"] = recall_at_ground_truth(held, ground_truth)
    row["recall@gt_lo"], row["recall@gt_hi"] = (
        recall_at_ground_truth_bounds(held, ground_truth))
    row.update(precision_recall(filled, ground_truth, committed,
                                abstained=abstained))
    row.update(abstention(row))
    return row


def score_competitors(pairs: Sequence[dict], matchers: Sequence[str], *,
                      results: Path, corpus: str, retriever: str = "qwen",
                      ks: Sequence[int] = DEFAULT_KS) -> pd.DataFrame:
    """
    One row per published matcher, pooled over the studies it covers.

    A column the matcher proposed nothing for counts as declined; these
    matchers are sparse, and this is how `gated` is treated too. `retriever`
    names the order that fills those columns for the evaluated view, which is
    the same order our arms are filled from.
    """
    from . import competitors

    rows = []
    for matcher in matchers:
        rankings: dict[str, list] = {}
        committed: dict[str, bool] = {}
        ground_truth: list[tuple[str, str]] = []
        covered, wall, proposed = [], 0.0, 0
        calls = tokens_in = 0

        for pair in pairs:
            payload = competitors.load(results, corpus, pair["name"], matcher)
            if payload is None:
                continue
            # The filename spelling, so these keys meet the retriever's.
            name = cache.fs_safe(pair["name"])
            covered.append(name)
            wall += float(payload.get("wall_s") or 0.0)
            proposed += int(payload.get("n_pairs_proposed") or 0)
            # `llm_calls` is {column: [call, ...]}, which is where a matcher
            # that calls a model records them. Reading a flat `n_calls` key
            # that none of these payloads carries reported every matcher at
            # zero calls, including the one that makes a call per column.
            spent = payload.get("llm_calls") or {}
            calls += int(payload.get("n_calls") or 0) + sum(
                len(cs) for cs in spent.values())
            tokens_in += int(payload.get("tokens_in") or 0) + sum(
                int(c.get("tokens_in") or 0) for cs in spent.values()
                for c in cs)
            for col, ranked in payload["rankings"].items():
                key = namespaced(name, col)
                rankings[key] = [(t, float(s)) for t, s in ranked]
                committed[key] = bool(ranked)
            ground_truth += [(namespaced(name, s), t)
                             for s, t in pair["ground_truth"]]

        if not ground_truth:
            continue
        # The same fill our arms get. Without it a matcher that proposes
        # nothing for a column can never be right there, while a declined
        # column of ours can, and the evaluated view compares the two.
        fallback, _ = retrieval_rankings(pairs, results=results, corpus=corpus,
                                         retriever=retriever, covered=covered)
        rows.append({
            "matcher": matcher,
            "n_studies": len(covered),
            "n_columns": len(rankings),
            "n_gold_pairs": len(ground_truth),
            "calls": calls,
            "wall_s": round(wall, 1),
            "n_pairs_proposed": proposed,
            **({"tokens_in_per_column": round(tokens_in / len(rankings), 1)}
               if tokens_in and rankings else {}),
            **({"calls_per_column": round(calls / len(rankings), 2)}
               if calls and rankings else {}),
            **metrics(rankings, ground_truth, committed, ks, fallback)})
    return pd.DataFrame(rows)


def abstention(counts: Mapping[str, Any]) -> dict[str, float]:
    """
    Declining scored as its own binary task ("this column has no target"),
    which no ranking metric reflects.
    """
    tn = int(counts.get("tn_declined", 0))
    fn = int(counts.get("fn_abstained", 0))
    fp = int(counts.get("fp_invented", 0))
    # A column with no readable answer declined nothing, so it stays out of
    # the precision, but it is still a gold-free column the matcher failed to
    # decline, so it stays in the recall's denominator.
    unreadable_free = int(counts.get("no_answer_free", 0))
    declined, no_gold = tn + fn, tn + fp + unreadable_free
    return {
        "n_declined": declined,
        "n_without_gold": no_gold,
        "abstain_precision": tn / declined if declined else float("nan"),
        "abstain_recall": tn / no_gold if no_gold else float("nan"),
    }


def blocking(runs: Iterable[Any], pairs: Sequence[dict], *, results: Path,
             corpus: str, retriever: str, window: int) -> dict[str, Any]:
    """
    A paged or gated run scored as a blocking scheme (see `blocking.py`).

    Blocks are located on the cached retrieval order. A run that pruned
    targets cut its blocks from a filtered order, so it raises rather than
    reporting depths against the wrong schema.
    """
    by_name = by_pair_name(pairs)
    columns: list[dict] = []
    orders: dict[str, list[str]] = {}
    ground_truth: list[tuple[str, str]] = []
    n_targets = n_brute = 0

    for row in runs:
        name = inventory._field(row, "pair")
        pair = by_name.get(name)
        if pair is None:
            continue
        meta = inventory.load(row)["trace"].get("metadata") or {}
        pruned_targets = (meta.get("prune") or {}).get("n_targets_dropped", 0)
        if pruned_targets:
            raise ValueError(
                f"{name}: the run dropped {pruned_targets} target column(s), so "
                "its blocks were cut from a filtered order and the cached "
                "retrieval file cannot say where they fell. Score the blocking "
                "measures against the filtered order, or omit them.")
        path = cache.retrieval_path(results, corpus, name, retriever)
        if not path.exists():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))["rankings"]

        for col, ranked in stored.items():
            orders[namespaced(name, col)] = [t for t, _ in ranked]
        for c in inventory.load(row)["trace"]["columns"]:
            columns.append({**c,
                            "source_column": namespaced(name, c["source_column"])})
        ground_truth += [(namespaced(name, s), t) for s, t in pair["ground_truth"]]
        n_targets = max(n_targets, pair["target_df"].shape[1])
        # Per schema pair, because only GDC's studies share a target schema.
        # Against the widest one, reduction ratio counts comparisons that
        # never existed and reads high by 25% to 44% on every Valentine corpus.
        n_brute += (len(inventory.load(row)["trace"]["columns"])
                    * pair["target_df"].shape[1])

    if not columns:
        return {}
    rep = blocking_report(columns, orders, ground_truth, window=window,
                          n_source_columns=len(columns),
                          n_target_columns=n_targets,
                          n_pairs_brute=n_brute)
    return rep.to_dict()


def retrieval_rankings(pairs: Sequence[dict], *, results: Path, corpus: str,
                       retriever: str, covered: Iterable[str] | None = None
                       ) -> tuple[dict[str, list], list[tuple[str, str]]]:
    """
    The cached retrieval order and gold of the pairs on disk, pooled. Keys and
    `covered` use the filename spelling of a pair name, as `pool` does.
    """
    wanted = set(covered) if covered is not None else None
    rankings: dict[str, list] = {}
    ground_truth: list[tuple[str, str]] = []

    for pair in pairs:
        name = cache.fs_safe(pair["name"])
        if wanted is not None and name not in wanted:
            continue
        path = cache.retrieval_path(results, corpus, pair["name"], retriever)
        if not path.exists():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))["rankings"]
        for col, ranked in stored.items():
            rankings[namespaced(name, col)] = [(t, float(s)) for t, s in ranked]
        ground_truth += [(namespaced(name, s), t) for s, t in pair["ground_truth"]]
    return rankings, ground_truth


def retrieval_ceiling(pairs: Sequence[dict], *, results: Path, corpus: str,
                      retriever: str, ks: Sequence[int] = DEFAULT_KS,
                      covered: Iterable[str] | None = None) -> dict[str, float]:
    """What the retriever alone reaches, on the same pooled denominator."""
    rankings, ground_truth = retrieval_rankings(
        pairs, results=results, corpus=corpus, retriever=retriever,
        covered=covered)
    if not ground_truth:
        return {}
    out = {f"ret_recall@{k}": recall_at_k(rankings, ground_truth, k) for k in ks}
    out["ret_mrr"] = mrr(rankings, ground_truth)
    out["ret_recall@gt"] = recall_at_ground_truth(rankings, ground_truth)
    return out


def score(inv: pd.DataFrame, pairs: Sequence[dict], *, results: Path,
          corpus: str, kind: str = "rerank", ks: Sequence[int] = DEFAULT_KS,
          ceiling: bool = True, group_by: str | None = None) -> pd.DataFrame:
    """
    One row per (configuration, repeat) with the metrics the paper reports.

    `inv` is an inventory of one arm, named by `kind`. Repeats stay separate.
    `group_by` adds a grouping key, such as the Valentine subset; None pools
    everything.
    """
    if not len(inv):
        return pd.DataFrame()

    keys = [*inventory.KEYS[kind], "rep"]
    if group_by:
        if group_by not in inv.columns:
            raise KeyError(f"{group_by!r} is not a column of the inventory; "
                           f"have {sorted(inv.columns)}")
        keys.append(group_by)
    ceilings: dict[tuple, dict] = {}
    rows = []

    for values, runs in inv.groupby(keys, dropna=False):
        cfg = dict(zip(keys, values))
        rankings, ground_truth, committed, abstained = pool(runs.itertuples(),
                                                            pairs)
        if not ground_truth:
            continue

        covered = sorted(set(runs["pair"]))
        # Counts come from the columns, not `trace.totals`, which a splice can
        # leave stale.
        cols = per_column(runs.itertuples(), pairs)
        row: dict[str, Any] = {
            **cfg,
            "model": cache.arm_tag(cfg["provider"], cfg["arm"]),
            "n_studies": len(covered),
            "n_columns": len(rankings),
            "n_gold_pairs": len(ground_truth),
            "calls": int(cols["api_calls"].sum()) if len(cols) else 0,
        }
        fallback, _ = retrieval_rankings(pairs, results=results, corpus=corpus,
                                         retriever=cfg["retriever"],
                                         covered=covered)
        row.update(metrics(rankings, ground_truth, committed, ks, fallback,
                           abstained))

        row.update(cost_shape(cols))
        # Fallbacks count as committed, so the precision columns cannot show
        # them; they are counted here instead.
        row["n_fallback"] = int(
            cols["stop_reason"].isin(("unparseable_fallback", "rerank_fallback")
                                     ).sum()) if len(cols) else 0
        row["n_unparseable"] = int(
            (cols["stop_reason"] == "unparseable").sum()) if len(cols) else 0
        # Abstentions split by who made them: the filter, or the model. Read
        # from `abstained`, since the model declines under two stop reasons.
        if len(cols):
            pruned = cols["stop_reason"] == "pruned"
            row["n_pruned"] = int(pruned.sum())
            row["n_model_declined"] = int((cols["abstained"] & ~pruned).sum())
        else:
            row["n_pruned"] = row["n_model_declined"] = 0
        if kind in ("paged", "gated"):
            # For gated the one block is the top_k shortlist.
            window = (int(cfg.get("top_k") or 20) if kind == "gated"
                      else int(cfg.get("window") or 20))
            try:
                row.update(blocking(runs.itertuples(), pairs, results=results,
                                    corpus=corpus, retriever=cfg["retriever"],
                                    window=window))
            except ValueError as exc:
                # Target-pruned: the row stands without the blocking columns.
                row["blocking_skipped"] = str(exc)

        if ceiling:
            ck = (cfg["retriever"], tuple(covered))
            if ck not in ceilings:
                ceilings[ck] = retrieval_ceiling(
                    pairs, results=results, corpus=corpus,
                    retriever=cfg["retriever"], ks=ks, covered=covered)
            row.update(ceilings[ck])

        row["calls_per_column"] = round(row["calls"] / max(row["n_columns"], 1), 2)
        row["tokens_in_per_column"] = (
            float(cols["tokens_in"].sum()) / max(row["n_columns"], 1)
            if len(cols) else 0.0)
        rows.append(row)

    return (pd.DataFrame(rows)
            .sort_values(["retriever", "model", "rep"])
            .reset_index(drop=True))


def across_repeats(scored: pd.DataFrame,
                   metrics: Sequence[str] = ("mrr", "recall@1", "top1",
                                             "precision_committed",
                                             "precision_evaluated",
                                             "mrr_abstain",
                                             "calls")) -> pd.DataFrame:
    """
    Mean, standard deviation and count per configuration, over repeats.

    The arm and the filter are part of the configuration. Without them a
    rerank run at one call per column and a paged run at nineteen land in one
    group, and `inventory.scan` defaults `top_k` to 20 even for paged, so
    nothing else separates them.
    """
    if not len(scored):
        return pd.DataFrame()
    have = [m for m in metrics if m in scored.columns]
    keys = [k for k in ("retriever", "provider", "arm", "model", "kind",
                        "prune", "top_k", "window", "threshold")
            if k in scored.columns]
    agg = scored.groupby(keys, dropna=False)[have].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index().rename(columns={f"{have[0]}_count": "reps"})


def short_model(arm: str) -> str:
    """
    The model as a table names it: the provider dropped, the variant kept,
    since two variants are two prompts.
    """
    parts = str(arm).split("__")
    if parts and parts[0] in cache.PROVIDERS:
        parts = parts[1:]
    return " ".join(parts) if parts else str(arm)
