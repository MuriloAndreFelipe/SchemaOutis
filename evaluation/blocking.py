"""
Scoring a paged run as a blocking scheme.

The measures are those of Papadakis, Skoutas, Thanos and Palpanas, Blocking
and Filtering Techniques for Entity Resolution: A Survey (ACM CSUR 53(2),
2020), mapped onto schema matching:

| the survey                     | here                                        |
|--------------------------------|---------------------------------------------|
| entity collections `E1`, `E2`  | the source columns, the target schema       |
| brute force `‖E‖`              | every (source, target) pair                 |
| duplicates `D(E)`              | the gold matches                            |
| Block Building                 | the retriever cutting its order into windows|
| Block Cleaning                 | the model accepting or declining a window   |
| Block Scheduling / Pruning     | walking in cosine order and stopping early  |

Pair Completeness is reported twice: `pc_traversed` is gold the walk was
shown, `pc_accepted` gold that survived into the answer. Reduction Ratio is
compared with `rr_stop_at_gold`, an oracle that stops at the gold block but
walks a column with no match to the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .matching import gold_by_source


def gold_positions(retrieval_order: Sequence[str], gold: Iterable[str]) -> list[int]:
    """0-based ranks of the gold targets in the retriever's order, ascending."""
    want = set(gold)
    return sorted(i for i, name in enumerate(retrieval_order) if name in want)


@dataclass
class ColumnBlocking:
    """One column's walk, scored against where its gold was."""

    column: str
    n_blocks_shown: int
    n_blocks_available: int
    n_comparisons: int
    accepted_block: int | None
    has_gold: bool
    # Block holding the earliest gold target, or None.
    oracle_block: int | None
    gold_ranks: list[int] = field(default_factory=list)
    n_gold_traversed: int = 0
    n_gold_accepted: int = 0
    # Block decisions: accepted or declined, against held gold or not.
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def blocks_needed(self) -> int | None:
        """Blocks an oracle would have opened to reach the gold."""
        return None if self.oracle_block is None else self.oracle_block + 1


def score_column(
    record: Mapping[str, Any],
    retrieval_order: Sequence[str],
    gold: Iterable[str],
    window: int,
) -> ColumnBlocking:
    """
    Score one column record. `retrieval_order` must be the order the blocks
    were cut from (the retriever's), not the run's own ranking.
    """
    ranks = gold_positions(retrieval_order, gold)
    accepted = record.get("accepted_block")
    accepted = None if accepted is None or accepted < 0 else int(accepted)
    blocks = record.get("blocks") or []
    n_shown = len(blocks)

    rec = ColumnBlocking(
        column=str(record.get("source_column", "")),
        n_blocks_shown=n_shown,
        n_blocks_available=int(record.get("n_blocks_available") or 0),
        n_comparisons=0,
        accepted_block=accepted,
        has_gold=bool(ranks),
        oracle_block=(ranks[0] // window if ranks and window > 0 else None),
        gold_ranks=ranks,
    )

    for bi in range(n_shown):
        start, end = bi * window, min((bi + 1) * window, len(retrieval_order))
        rec.n_comparisons += max(0, end - start)
        inside = sum(1 for r in ranks if start <= r < end)
        rec.n_gold_traversed += inside
        if accepted is not None and bi == accepted:
            rec.n_gold_accepted += inside
            rec.tp += 1 if inside else 0
            rec.fp += 0 if inside else 1
        else:
            rec.fn += 1 if inside else 0
            rec.tn += 0 if inside else 1
    return rec


@dataclass
class BlockingReport:
    """Survey measures for one (study, configuration) run."""

    n_source_columns: int
    n_target_columns: int
    n_gold_pairs: int
    n_comparisons: int
    n_gold_traversed: int
    n_gold_accepted: int
    n_blocks_shown: int
    tp_gold: int
    fp_gold: int
    fn_gold: int
    n_comparisons_oracle_scan: int
    # The true number of pairs, summed per schema pair. Not the product of the
    # two counts above: on a corpus where every pair has its own target schema
    # that product counts comparisons no matcher could ever make.
    n_pairs_brute: int = 0
    columns: list[ColumnBlocking] = field(default_factory=list)

    @property
    def n_comparisons_brute(self) -> int:
        return self.n_pairs_brute or self.n_source_columns * self.n_target_columns

    @property
    def pc_traversed(self) -> float:
        """Pair Completeness over what the model was shown."""
        return (self.n_gold_traversed / self.n_gold_pairs) if self.n_gold_pairs else 0.0

    @property
    def pc_accepted(self) -> float:
        """Pair Completeness over what survived into the answer."""
        return (self.n_gold_accepted / self.n_gold_pairs) if self.n_gold_pairs else 0.0

    @property
    def rr(self) -> float:
        """Reduction Ratio against brute force."""
        b = self.n_comparisons_brute
        return (1 - self.n_comparisons / b) if b else 0.0

    @property
    def rr_stop_at_gold(self) -> float:
        """RR of an oracle that stops at the gold but must still prove a negative."""
        b = self.n_comparisons_brute
        return (1 - self.n_comparisons_oracle_scan / b) if b else 0.0

    # The model as a block classifier, over columns that have gold. Pooled over
    # all columns these are dominated by columns where accepting is always wrong.

    @property
    def block_precision_gold(self) -> float:
        """Whether it picks the right block when there is one."""
        d = self.tp_gold + self.fp_gold
        return float("nan") if d == 0 else self.tp_gold / d

    @property
    def block_recall_gold(self) -> float:
        d = self.tp_gold + self.fn_gold
        return float("nan") if d == 0 else self.tp_gold / d

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_target_columns": self.n_target_columns,
            "n_gold_pairs": self.n_gold_pairs,
            "n_comparisons": self.n_comparisons,
            "n_comparisons_brute": self.n_comparisons_brute,
            "n_comparisons_oracle_scan": self.n_comparisons_oracle_scan,
            "n_gold_traversed": self.n_gold_traversed,
            "n_gold_accepted": self.n_gold_accepted,
            "n_blocks_shown": self.n_blocks_shown,
            "tp_gold": self.tp_gold, "fp_gold": self.fp_gold,
            "fn_gold": self.fn_gold,
            "pc_traversed": self.pc_traversed, "pc_accepted": self.pc_accepted,
            "rr": self.rr, "rr_stop_at_gold": self.rr_stop_at_gold,
            "block_precision_gold": self.block_precision_gold,
            "block_recall_gold": self.block_recall_gold,
        }


def blocking_report(
    columns: Sequence[Mapping[str, Any]],
    retrieval_orders: Mapping[str, Sequence[str]],
    ground_truth: Iterable[tuple[str, str]],
    *,
    window: int,
    n_source_columns: int | None = None,
    n_target_columns: int | None = None,
    n_pairs_brute: int = 0,
) -> BlockingReport:
    """
    Survey measures for a whole run. `retrieval_orders` maps each source
    column to the complete order its blocks were cut from.
    """
    gold = gold_by_source(ground_truth)
    scored = [
        score_column(c, retrieval_orders.get(c["source_column"]) or [],
                     gold.get(c["source_column"], set()), window)
        for c in columns
    ]

    n_targets = n_target_columns or max(
        (len(v) for v in retrieval_orders.values()), default=0)
    n_sources = n_source_columns or len(columns)

    # The oracle stops at the gold, and walks a column with none to the end.
    n_oracle_scan = (
        sum(min((c.blocks_needed or 0) * window,
                len(retrieval_orders.get(c.column) or []))
            for c in scored if c.has_gold)
        + sum(len(retrieval_orders.get(c.column) or [])
              for c in scored if not c.has_gold))

    return BlockingReport(
        n_source_columns=n_sources,
        n_target_columns=n_targets,
        n_pairs_brute=n_pairs_brute,
        n_gold_pairs=sum(len(v) for v in gold.values()),
        n_comparisons=sum(c.n_comparisons for c in scored),
        n_gold_traversed=sum(c.n_gold_traversed for c in scored),
        n_gold_accepted=sum(c.n_gold_accepted for c in scored),
        n_blocks_shown=sum(c.n_blocks_shown for c in scored),
        tp_gold=sum(c.tp for c in scored if c.has_gold),
        fp_gold=sum(c.fp for c in scored if c.has_gold),
        fn_gold=sum(c.fn for c in scored if c.has_gold),
        n_comparisons_oracle_scan=n_oracle_scan,
        columns=scored,
    )
