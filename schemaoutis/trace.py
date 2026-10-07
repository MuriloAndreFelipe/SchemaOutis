"""What a run records: the run, each column, and each block shown.

Anything that changes what the model was asked or answered is stored; the rest,
such as the call count, is derived so the two cannot disagree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Why a column's traversal ended.
STOP_ACCEPTED = "accepted"           # a block was accepted
STOP_EXHAUSTED = "blocks_exhausted"  # every block was shown and declined
STOP_CAPPED = "block_cap"            # max_blocks reached with candidates left
STOP_ERROR = "error"                 # the request raised
STOP_UNPARSEABLE = "unparseable"     # the reply could not be read
STOP_RETRIEVAL = "retrieval"         # the retrieval arm: cosine order, no model
STOP_FALLBACK = "rerank_fallback"    # rerank reply unusable, cosine order kept
STOP_UNPARSEABLE_FALLBACK = "unparseable_fallback"  # paged, retriever's window kept
STOP_DECLINED = "declined"           # gated refused its only block
STOP_PRUNED = "pruned"               # skipped by a source prune, never asked

# Declining counts as answered: on a corpus where most columns have no match it
# is usually the right answer.
ANSWERED_STOPS = (STOP_ACCEPTED, STOP_EXHAUSTED, STOP_RETRIEVAL,
                  STOP_FALLBACK, STOP_UNPARSEABLE_FALLBACK, STOP_DECLINED,
                  STOP_PRUNED)

# A fallback counts as committed, not declined: the model did not refuse, its
# reply was unreadable.
COMMITTED_STOPS = (STOP_ACCEPTED, STOP_RETRIEVAL, STOP_FALLBACK,
                   STOP_UNPARSEABLE_FALLBACK)

ABSTAINED_STOPS = (STOP_EXHAUSTED, STOP_DECLINED, STOP_PRUNED)


@dataclass
class BlockRecord:
    """One window shown to the model, and what came back."""

    index: int                       # 0-based block number
    start: int                       # first rank of the window, inclusive
    end: int                         # last rank, exclusive
    verdict: str                     # accept | reject | unparseable | error
    reason: str = ""                 # explicit NONE, or the threshold
    top_score: float | None = None
    n_scored: int = 0                # candidates given a score
    n_hallucinated: int = 0          # names returned that were not in the window
    parse_note: str = ""
    # Requests for this block; more than one means a parse retry. Still one
    # call in `api_calls`.
    n_attempts: int = 1
    tokens_in: int = 0
    tokens_out: int = 0
    cache_write: int = 0
    cache_read: int = 0
    wall_ms: float = 0.0
    error: str = ""
    n_chars: int = 0                 # reply length given to the parser
    n_thinking_chars: int = 0        # reasoning the backend reported separately
    raw_response: str = ""           # only with capture_raw
    # Decision backend only. `probabilities` covers every candidate shown, so
    # the below-`p_none` cut can be recomputed after the run.
    probabilities: dict[str, float] | None = None
    p_none: float | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnRecord:
    """One source column's walk down its retrieval order."""

    source_column: str
    n_blocks_available: int = 0
    accepted_block: int | None = None
    stop_reason: str = STOP_EXHAUSTED
    ranking: list[tuple[str, float]] = field(default_factory=list)
    blocks: list[BlockRecord] = field(default_factory=list)
    error: str = ""
    wall_ms: float = 0.0

    @property
    def api_calls(self) -> int:
        return len(self.blocks)

    @property
    def accepted(self) -> bool:
        return self.accepted_block is not None

    @property
    def abstained(self) -> bool:
        return self.stop_reason in ABSTAINED_STOPS

    @property
    def answered(self) -> bool:
        return self.stop_reason in ANSWERED_STOPS

    @property
    def committed(self) -> bool:
        """Whether the arm claimed a match."""
        return self.stop_reason in COMMITTED_STOPS

    @property
    def tokens_in(self) -> int:
        return sum(b.tokens_in for b in self.blocks)

    @property
    def tokens_out(self) -> int:
        return sum(b.tokens_out for b in self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_column": self.source_column,
            "n_blocks_available": self.n_blocks_available,
            "accepted_block": self.accepted_block,
            "stop_reason": self.stop_reason,
            "api_calls": self.api_calls,
            "accepted": self.accepted,
            "abstained": self.abstained,
            "answered": self.answered,
            "committed": self.committed,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "wall_ms": round(self.wall_ms, 2),
            "error": self.error,
            "ranking": [[t, s] for t, s in self.ranking],
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass
class RunTrace:
    """Every column of one (study, configuration) run, and how it was made."""

    metadata: dict[str, Any] = field(default_factory=dict)
    columns: list[ColumnRecord] = field(default_factory=list)

    def column(self, name: str) -> ColumnRecord:
        """The record for `name`, created on first use."""
        for c in self.columns:
            if c.source_column == name:
                return c
        rec = ColumnRecord(source_column=name)
        self.columns.append(rec)
        return rec

    def completeness(self) -> dict[str, Any]:
        """
        Counts by outcome. `complete` means every column was answered, which
        includes columns declined with no ranking.
        """
        by_reason: dict[str, int] = {}
        for c in self.columns:
            by_reason[c.stop_reason] = by_reason.get(c.stop_reason, 0) + 1
        n = len(self.columns)
        n_answered = sum(1 for c in self.columns if c.answered)
        n_accepted = sum(1 for c in self.columns if c.accepted)
        return {
            "n_columns": n,
            "n_accepted": n_accepted,
            "n_committed": sum(1 for c in self.columns if c.committed),
            "n_abstained": sum(1 for c in self.columns if c.abstained),
            "n_answered": n_answered,
            "n_unresolved": n - n_answered,
            "n_errored": by_reason.get(STOP_ERROR, 0),
            "complete": n > 0 and n_answered == n,
            "by_stop_reason": by_reason,
        }

    def totals(self) -> dict[str, int]:
        return {
            "api_calls": sum(c.api_calls for c in self.columns),
            "tokens_in": sum(c.tokens_in for c in self.columns),
            "tokens_out": sum(c.tokens_out for c in self.columns),
            "cache_write": sum(b.cache_write for c in self.columns
                               for b in c.blocks),
            "cache_read": sum(b.cache_read for c in self.columns
                              for b in c.blocks),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "completeness": self.completeness(),
            "totals": self.totals(),
            "columns": [c.to_dict() for c in self.columns],
        }
