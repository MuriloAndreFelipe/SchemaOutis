"""
Scoring a ranking against ground truth.

Every function takes a `rankings` dict (source column -> ordered targets) and a
`ground_truth` list of (source, target) pairs. Metrics are computed over the
complete ranking; where a metric must tell "answered" from "fell back", that is
passed in explicitly rather than inferred from the ranking's length.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

Ranking = Sequence[tuple[str, float]] | Sequence[str]
Rankings = Mapping[str, Ranking]
GroundTruth = Iterable[tuple[str, str]]


def _targets(ranking: Ranking) -> list[str]:
    """Target names in order, from (name, score) pairs or plain names."""
    out = []
    for item in ranking:
        out.append(str(item[0]) if isinstance(item, (tuple, list)) else str(item))
    return out


def gold_by_source(ground_truth: GroundTruth) -> dict[str, set[str]]:
    """source column -> its gold targets (there may be several)."""
    gold: dict[str, set[str]] = defaultdict(set)
    for src, tgt in ground_truth:
        gold[str(src)].add(str(tgt))
    return dict(gold)


def gold_rank(ranking: Ranking, gold: set[str]) -> int | None:
    """1-based rank of the first correct target, or None."""
    for i, name in enumerate(_targets(ranking), start=1):
        if name in gold:
            return i
    return None


def recall_at_k(rankings: Rankings, ground_truth: GroundTruth, k: int = 1) -> float:
    """
    Share of gold pairs whose target is in the top k of its column. Divided by
    pairs, so on 1:N gold it cannot reach 1.0 at k=1; see `top1_accuracy`.
    """
    gold = gold_by_source(ground_truth)
    n_pairs = sum(len(v) for v in gold.values())
    if not n_pairs:
        return 0.0
    hits = 0
    for src, tgts in gold.items():
        top = set(_targets(rankings.get(src) or [])[:k])
        hits += len(top & tgts)
    return hits / n_pairs


def recall_at_ground_truth(rankings: Rankings,
                           ground_truth: GroundTruth) -> float:
    """
    Valentine's Recall@GT: keep the `len(ground_truth)` best-scoring pairs
    across the whole schema pair. The budget is global, not per column. Ties
    are broken arbitrarily, as in the reference implementation.
    """
    gold = {(str(s), str(t)) for s, t in ground_truth}
    n = len(gold)
    if not n:
        return 0.0
    scored = [(float(score), str(src), str(tgt))
              for src, ranked in rankings.items() for tgt, score in ranked]
    if not scored:
        return 0.0
    scored.sort(key=lambda row: -row[0])
    return sum(1 for _, s, t in scored[:n] if (s, t) in gold) / n


def recall_at_ground_truth_bounds(
        rankings: Rankings, ground_truth: GroundTruth) -> tuple[float, float]:
    """
    The worst and best Recall@GT over tie-breaks. Coarse LLM scores can put
    many pairs on the cut, so the interval can be wide.
    """
    gold = {(str(s), str(t)) for s, t in ground_truth}
    n = len(gold)
    if not n:
        return 0.0, 0.0
    scored = [(float(score), str(src), str(tgt))
              for src, ranked in rankings.items() for tgt, score in ranked]
    if not scored:
        return 0.0, 0.0

    def hits(rows):
        return sum(1 for _, s, t in rows[:n] if (s, t) in gold) / n

    worst = sorted(scored, key=lambda r: (-r[0], (r[1], r[2]) in gold))
    best = sorted(scored, key=lambda r: (-r[0], (r[1], r[2]) not in gold))
    return hits(worst), hits(best)


def top1_accuracy(rankings: Rankings, ground_truth: GroundTruth) -> float:
    """Share of gold-bearing columns whose rank-1 target is correct."""
    gold = gold_by_source(ground_truth)
    if not gold:
        return 0.0
    hits = sum(1 for src, tgts in gold.items()
               if (t := _targets(rankings.get(src) or [])[:1]) and t[0] in tgts)
    return hits / len(gold)


def mrr(rankings: Rankings, ground_truth: GroundTruth) -> float:
    """
    Mean reciprocal rank over every gold-bearing column. An unanswered column
    scores zero; dividing by answered columns would reward declining.
    """
    gold = gold_by_source(ground_truth)
    if not gold:
        return 0.0
    total = 0.0
    for src, tgts in gold.items():
        r = gold_rank(rankings.get(src) or [], tgts)
        if r:
            total += 1.0 / r
    return total / len(gold)


def mrr_abstain(rankings: Rankings, ground_truth: GroundTruth,
                committed: Mapping[str, bool],
                source_columns: Iterable[str] | None = None,
                abstained: Mapping[str, bool] | None = None) -> float:
    """
    Mean over every source column, with and without gold. A gold-bearing
    column scores its reciprocal rank if committed, else 0; a gold-free column
    scores 1 if the matcher said so, by declining or by being pruned.

    `abstained` separates saying no from failing to answer. A column whose
    reply could not be read asserted nothing, but it did not decline either,
    and crediting it would pay a model for being unreadable. Without the
    argument, anything not committed counts as a decline, which is right only
    where no column can fail.
    """
    gold = gold_by_source(ground_truth)
    cols = list(source_columns) if source_columns is not None else list(rankings)
    if not cols:
        return 0.0
    total = 0.0
    for col in cols:
        did = bool(committed.get(col, False))
        said_no = (not did if abstained is None
                   else bool(abstained.get(col, False)))
        if col not in gold:
            total += 1.0 if said_no else 0.0
        elif did:
            r = gold_rank(rankings.get(col) or [], gold[col])
            total += 1.0 / r if r else 0.0
    return total / len(cols)


# The outcomes a source column can have, disjoint and exhaustive. The last two
# are columns that neither asserted nor declined, because the reply could not
# be read or the call failed. They are not abstentions and must not be scored
# as any kind of decision.
OUTCOMES = ("TP", "FP_invented", "FP_FN_wrong", "FN_abstained", "TN_declined",
            "no_answer_gold", "no_answer_free")


def classify_columns(
    rankings: Rankings,
    ground_truth: GroundTruth,
    committed: Mapping[str, bool],
    source_columns: Iterable[str] | None = None,
    abstained: Mapping[str, bool] | None = None,
) -> dict[str, str]:
    """
    Put every source column in one of `OUTCOMES`. `committed` must be passed
    in, since a declined column can still carry a ranking, and `abstained`
    separates a decline from a reply that could not be read.

      TP              has gold, committed, rank 1 correct
      FP_invented     no gold, committed
      FP_FN_wrong     has gold, committed, rank 1 wrong
      FN_abstained    has gold, declined
      TN_declined     no gold, declined
      no_answer_gold  has gold, neither committed nor declined
      no_answer_free  no gold, neither committed nor declined
    """
    gold = gold_by_source(ground_truth)
    cols = list(source_columns) if source_columns is not None else list(rankings)

    out = {}
    for col in cols:
        has_gold = bool(gold.get(col))
        did = bool(committed.get(col, False))
        said_no = (not did if abstained is None
                   else bool(abstained.get(col, False)))
        top = _targets(rankings.get(col) or [])[:1]
        correct = bool(has_gold and top and top[0] in gold[col])

        if did:
            out[col] = ("TP" if correct else "FP_FN_wrong") if has_gold \
                else "FP_invented"
        elif said_no:
            out[col] = "FN_abstained" if has_gold else "TN_declined"
        else:
            out[col] = "no_answer_gold" if has_gold else "no_answer_free"
    return out


def precision_recall(
    rankings: Rankings,
    ground_truth: GroundTruth,
    committed: Mapping[str, bool],
    source_columns: Iterable[str] | None = None,
    abstained: Mapping[str, bool] | None = None,
) -> dict[str, float]:
    """
    Precision and recall over source columns, two ways. As committed, declined
    columns leave the denominator. As evaluated, every column contributes its
    rank-1 target. They coincide for an arm that cannot decline.
    """
    cls = classify_columns(rankings, ground_truth, committed, source_columns,
                           abstained)
    n = {k: sum(1 for v in cls.values() if v == k) for k in OUTCOMES}

    gold = gold_by_source(ground_truth)
    cols = list(cls)
    n_gold = sum(1 for c in cols if gold.get(c))
    claimed = n["TP"] + n["FP_invented"] + n["FP_FN_wrong"]

    tp_eval = sum(1 for c in cols if gold.get(c)
                  and (t := _targets(rankings.get(c) or [])[:1])
                  and t[0] in gold[c])

    p_com = n["TP"] / claimed if claimed else float("nan")
    r_com = n["TP"] / n_gold if n_gold else float("nan")
    p_ev = tp_eval / len(cols) if cols else float("nan")
    r_ev = tp_eval / n_gold if n_gold else float("nan")

    def _f1(p, r):
        return 2 * p * r / (p + r) if p and r else 0.0

    return {
        **{k.lower(): n[k] for k in OUTCOMES},
        "n_columns": len(cols), "n_gold_columns": n_gold,
        "precision_committed": p_com, "recall_committed": r_com,
        "f1_committed": _f1(p_com, r_com),
        "precision_evaluated": p_ev, "recall_evaluated": r_ev,
        "f1_evaluated": _f1(p_ev, r_ev),
    }


def mrr_forfeited(rankings: Rankings, ground_truth: GroundTruth,
                  removed: Iterable[str]) -> dict[str, float]:
    """
    The MRR a source prune gives up, read off the unpruned ranking: removing a
    column subtracts its reciprocal rank and leaves the denominator alone.
    `removed` is namespaced as `rankings` is.
    """
    gold = gold_by_source(ground_truth)
    gone = set(map(str, removed))
    n = len(gold)
    lost = 0.0
    n_cols = 0
    for src in gold:
        if src not in gone:
            continue
        n_cols += 1
        r = gold_rank(rankings.get(src) or [], gold[src])
        if r:
            lost += 1.0 / r
    return {
        "n_gold_columns": n,
        "n_gold_columns_removed": n_cols,
        "mrr_forfeited": lost / n if n else 0.0,
    }


def gold_beyond_k(rankings: Rankings, ground_truth: GroundTruth,
                  k: int = 20) -> list[tuple[str, str, int | None]]:
    """
    Gold pairs ranked deeper than `k`, or absent (rank None): what a
    `k`-candidate shortlist can never reach. Per pair, not per column.
    """
    ranked = {src: _targets(r) for src, r in rankings.items()}
    at = {src: {t: i for i, t in enumerate(ts, start=1)}
          for src, ts in ranked.items()}
    out = []
    for src, tgt in ground_truth:
        src, tgt = str(src), str(tgt)
        r = at.get(src, {}).get(tgt)
        if r is None or r > k:
            out.append((src, tgt, r))
    return out
