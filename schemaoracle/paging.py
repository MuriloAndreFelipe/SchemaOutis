"""The traversal: one window of candidates at a time.

A one-shot reranker sees the top k, so gold at rank 34 with k=20 is out of
reach. Paging walks the order in blocks (ranks 0-19, 20-39, ...) until the
model accepts one or the schema runs out.

In the terms of Papadakis et al. (ACM CSUR 53(2), 2020), the retriever is
block building and declining a block is block cleaning.

A window is declined in one of two ways, recorded separately in the trace: an
explicit NONE (or []), or a best score below `accept_threshold`. Cost is one
call per block, at most ceil(n_targets / window), which is 37 on GDC.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from .prompts import (
    MAGNETO_SYSTEM,
    PARSE_FAILED,
    PARSE_OK,
    PARSE_ONLY_THINKING,
    PARSE_RETRY_SYSTEM,
    PROMPT_LAYOUTS,
    RERANK_STOP_SEQUENCES,
    UNPARSEABLE_POLICIES,
    VERDICT_ACCEPT,
    VERDICT_ERROR,
    VERDICT_REJECT,
    VERDICT_UNPARSEABLE,
    _norm,
    _text_of,
    build_prompt,
    parse_response,
    says_none,
    serialize_for_prompt,
)
from .trace import (
    STOP_ACCEPTED,
    STOP_CAPPED,
    STOP_DECLINED,
    STOP_ERROR,
    STOP_EXHAUSTED,
    STOP_PRUNED,
    STOP_UNPARSEABLE,
    STOP_UNPARSEABLE_FALLBACK,
    BlockRecord,
    ColumnRecord,
    RunTrace,
)

# Magneto's reranker uses these whatever the retriever sampled with.
RERANK_SAMPLING_MODE = "priority_sampling"
RERANK_SAMPLING_SIZE = 10

DEFAULT_WINDOW = 20
DEFAULT_ACCEPT_THRESHOLD = 0.0

# Client options that carry the request but cannot change the answer, so two
# runs differing only in these are the same measurement.
TRANSPORT_ONLY = ("timeout_s", "base_url", "api_key", "max_retries")

# Ceiling on re-asking a block whose reply could not be parsed.
MAX_PARSE_RETRIES = 3


class PartialRun(RuntimeError):
    """
    A run stopped early. Carries the trace so the caller can save what was
    answered: those columns were paid for and are not reproducible for free.
    """

    def __init__(self, message: str, trace: RunTrace):
        super().__init__(message)
        self.trace = trace


class BlockBudgetExceeded(PartialRun):
    """The per-pair token ceiling was reached mid-traversal."""


class RunAborted(PartialRun):
    """Consecutive columns failed and the run stopped. Carries the partial trace."""


# Consecutive failed columns before a run gives up, e.g. on an expired key.
DEFAULT_MAX_CONSECUTIVE_ERRORS = 5


def block_spans(n_total: int, window: int, max_blocks: int | None = None):
    """
    The (start, end) ranks of each block in walk order, and how many blocks the
    full ranking has. Blocks tile the ranking: each target is in exactly one.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    n_available = (n_total + window - 1) // window
    n = n_available if max_blocks is None else min(n_available, max(1, int(max_blocks)))
    return [(i * window, min(i * window + window, n_total)) for i in range(n)], n_available


class Pager:
    """
    Walks one source column's ranking window by window. Keeps no state between
    columns except the run-wide token count, so columns are independent.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        window: int = DEFAULT_WINDOW,
        accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
        max_blocks: int | None = None,
        # This block is the only one the arm will see, so refusing it is a
        # decline rather than running out of budget (gated).
        terminal: bool = False,
        on_unparseable: str = "stop",
        prompt_layout: str = "standard",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_total_tokens: int = 50_000_000,
        sampling_mode: str = RERANK_SAMPLING_MODE,
        sampling_size: int = RERANK_SAMPLING_SIZE,
        seed: int = 0,
        tolerant: bool = True,
        capture_raw: bool = False,
        # What the caller passed to `build_client`. Recorded, never sent from
        # here: the client already holds it.
        client_options: Mapping[str, Any] | None = None,
        max_parse_retries: int = 0,
    ):
        if not 0.0 <= accept_threshold <= 1.0:
            raise ValueError(
                f"accept_threshold must be in [0, 1], got {accept_threshold}"
            )
        if prompt_layout not in PROMPT_LAYOUTS:
            raise ValueError(f"prompt_layout must be one of {PROMPT_LAYOUTS}")
        if on_unparseable not in UNPARSEABLE_POLICIES:
            raise ValueError(f"on_unparseable must be one of {UNPARSEABLE_POLICIES}")
        if not 0 <= max_parse_retries <= MAX_PARSE_RETRIES:
            raise ValueError(
                f"max_parse_retries must be in [0, {MAX_PARSE_RETRIES}], got "
                f"{max_parse_retries}")

        self.client = client
        self.model = model
        self.window = window
        self.accept_threshold = accept_threshold
        self.max_blocks = max_blocks
        self.terminal = terminal
        self.on_unparseable = on_unparseable
        self.prompt_layout = prompt_layout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_total_tokens = max_total_tokens
        self.sampling_mode = sampling_mode
        self.sampling_size = sampling_size
        self.seed = seed
        self.tolerant = tolerant
        self.capture_raw = capture_raw
        self.client_options = dict(client_options or {})
        self.max_parse_retries = max_parse_retries
        self.total_tokens = 0

    def config(self) -> dict[str, Any]:
        """
        What changes the request, as recorded in the trace. For a decision
        backend the prompt-only settings are None, since none of them is sent.
        """
        return {
            "model": self.model,
            "window": self.window,
            "accept_threshold": self.accept_threshold,
            "max_blocks": self.max_blocks,
            "on_unparseable": self.on_unparseable,
            "prompt_layout": self.prompt_layout,
            "temperature": self.temperature,
            "max_parse_retries": self.max_parse_retries,
            "sampling_mode": self.sampling_mode,
            "sampling_size": self.sampling_size,
            "seed": self.seed,
            "client": self.recorded_client_options() or None,
        }

    def recorded_client_options(self) -> dict[str, Any]:
        """
        The client options that change the answer. `think` and `num_ctx` both
        do and neither is in the filename, so without this a finished run
        cannot say which settings produced it. Transport-only keys are left
        out; anything unrecognised is kept, so a new knob is recorded by
        default rather than silently dropped.
        """
        return {k: v for k, v in sorted(self.client_options.items())
                if k not in TRANSPORT_ONLY}

    def run_column(
        self,
        source_column: str,
        source_series: pd.Series,
        full_ranking: Sequence[tuple[str, float]],
        target_text: dict[str, str],
    ) -> ColumnRecord:
        """
        Walk `full_ranking`, the complete order over the target schema, until a
        window is accepted or the order runs out.
        """
        started = time.perf_counter()
        cand_text = serialize_for_prompt(
            source_column, source_series, self.sampling_size,
            self.sampling_mode, self.seed,
        )

        spans, n_available = block_spans(len(full_ranking), self.window,
                                         self.max_blocks)
        rec = ColumnRecord(source_column=source_column,
                           n_blocks_available=n_available)

        pooled: dict[str, float] = {}
        for bi, (start, end) in enumerate(spans):
            names = [t for t, _ in full_ranking[start:end]]
            if not names:
                break

            block = self._run_block(cand_text, names, target_text, bi,
                                    len(spans), start, end)
            rec.blocks.append(block)

            for name, score in getattr(block, "_ranking", []):
                pooled[name] = max(pooled.get(name, 0.0), score)

            if block.verdict == VERDICT_ACCEPT:
                rec.accepted_block = bi
                rec.ranking = list(getattr(block, "_ranking", []))
                rec.stop_reason = STOP_ACCEPTED
                break

            if block.verdict == VERDICT_ERROR:
                rec.stop_reason = STOP_ERROR
                rec.error = block.error
                break

            # An unparseable reply is not a rejection: moving on would skip a
            # window the model never answered.
            if block.verdict == VERDICT_UNPARSEABLE:
                if self.on_unparseable == "fallback":
                    # The retriever's window stands in, as for rerank.
                    rec.ranking = [(t, s) for t, s in full_ranking[start:end]]
                    rec.stop_reason = STOP_UNPARSEABLE_FALLBACK
                    rec.error = ""
                    print(f"    !! {source_column!r}: block {bi} unparseable "
                          f"({block.parse_note}), falling back to the "
                          f"retriever's top {end - start}")
                else:
                    rec.stop_reason = STOP_UNPARSEABLE
                    rec.error = block.parse_note
                break

            if self.total_tokens >= self.max_total_tokens:
                # No trace here; `run_paging` attaches the partial one so the
                # columns already answered are still written.
                raise BlockBudgetExceeded(
                    f"per-pair ceiling of {self.max_total_tokens:,} tokens "
                    f"reached while walking {source_column!r} at block "
                    f"{bi + 1}/{len(spans)}", None)

        if rec.accepted_block is None and rec.stop_reason not in (
                STOP_ERROR, STOP_UNPARSEABLE, STOP_UNPARSEABLE_FALLBACK):
            if pooled:
                # Nothing was accepted, but the model's own scores are kept.
                rec.ranking = sorted(pooled.items(), key=lambda kv: (-kv[1], kv[0]))
            if self.terminal:
                rec.stop_reason = STOP_DECLINED
            else:
                rec.stop_reason = (STOP_EXHAUSTED if len(spans) == n_available
                                   else STOP_CAPPED)

        rec.wall_ms = (time.perf_counter() - started) * 1000
        return rec

    def _settle(self, block: BlockRecord, pairs, declined: bool) -> BlockRecord:
        """Turn a block's scores into a verdict."""
        block.n_scored = len(pairs)

        if pairs:
            top = max(score for _, score in pairs)
            accept = top >= self.accept_threshold
            block.top_score = top
            block.verdict = VERDICT_ACCEPT if accept else VERDICT_REJECT
            block.reason = (f"top_score>={self.accept_threshold:.2f}" if accept
                            else f"top_score<{self.accept_threshold:.2f}")
            # Not serialised; only the accepted block's ranking is stored,
            # on the column.
            block._ranking = sorted(pairs, key=lambda kv: (-kv[1], kv[0]))
            return block

        if declined:
            block.verdict = VERDICT_REJECT
            block.reason = "explicit_none"
            block._ranking = []
            return block

        block.verdict = VERDICT_UNPARSEABLE
        block._ranking = []
        return block

    def _run_block(self, cand_text, names, target_text, index, n_blocks,
                   start, end) -> BlockRecord:
        started = time.perf_counter()
        targets = [target_text[n] for n in names]
        valid = {_norm(n): n for n in names}
        prompt = build_prompt(cand_text, targets, index, n_blocks,
                              layout=self.prompt_layout)

        block = BlockRecord(index=index, start=start, end=end,
                            verdict=VERDICT_UNPARSEABLE)
        # An unreadable reply may be asked again with the format spelled out.
        # Every attempt's tokens count, though the block is still one call.
        for attempt in range(self.max_parse_retries + 1):
            block.n_attempts = attempt + 1
            try:
                response = self._create(prompt, correct_format=attempt > 0)
            except Exception as exc:                   # noqa: BLE001
                block.verdict = VERDICT_ERROR
                block.error = f"{type(exc).__name__}: {exc}"
                block.wall_ms = (time.perf_counter() - started) * 1000
                return block

            usage = getattr(response, "usage", None)
            block.tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
            block.tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
            block.cache_write += int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0)
            block.cache_read += int(
                getattr(usage, "cache_read_input_tokens", 0) or 0)
            self.total_tokens += (int(getattr(usage, "input_tokens", 0) or 0)
                                  + int(getattr(usage, "output_tokens", 0) or 0))

            raw = _text_of(response)
            if self.capture_raw:
                block.raw_response = raw
            thinking = str(getattr(response, "thinking", "") or "")
            block.n_chars = len(raw)
            block.n_thinking_chars = len(thinking)

            pairs, hallucinated, note = parse_response(
                raw, valid, tolerant=self.tolerant,
                texts={n: target_text[n] for n in names})
            # No content but a full thinking channel: a reasoning model spent
            # its budget deliberating. Not a formatting failure, so not retried.
            if not raw and thinking and note == PARSE_FAILED:
                note = PARSE_ONLY_THINKING

            if note != PARSE_FAILED or says_none(raw):
                break

        block.parse_note = note
        block.n_hallucinated = hallucinated
        block.wall_ms = (time.perf_counter() - started) * 1000
        return self._settle(block, pairs, says_none(raw))

    def _create(self, prompt: str, *, correct_format: bool = False) -> Any:
        system = (MAGNETO_SYSTEM + PARSE_RETRY_SYSTEM if correct_format
                  else MAGNETO_SYSTEM)
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            stop_sequences=list(RERANK_STOP_SEQUENCES),
            messages=[{"role": "user", "content": prompt}],
        )


def run_paging(
    source: pd.DataFrame,
    rankings: dict[str, list[tuple[str, float]]],
    target_text: dict[str, str],
    pager: Pager,
    *,
    only_columns: Iterable[str] | None = None,
    skip: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
    on_column: Callable[[ColumnRecord], None] | None = None,
    verbose: bool = False,
    max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
) -> RunTrace:
    """
    Walk every source column and return the trace.

    `rankings` maps each source column to its complete order, and
    `target_text` each target column to its serialised form. `only_columns`
    restricts the walk; `skip` columns are recorded as pruned with no call.
    After `max_consecutive_errors` failed columns in a row the run raises
    RunAborted with the partial trace (0 disables this).
    """
    wanted = list(rankings) if only_columns is None else list(only_columns)
    missing = [c for c in wanted if c not in rankings]
    if missing:
        raise KeyError(
            f"no retrieval ranking for {len(missing)} column(s): {missing[:5]}"
        )

    trace = RunTrace(metadata=dict(metadata or {}))
    trace.metadata.setdefault("config", pager.config())
    trace.metadata["only_columns"] = (None if only_columns is None
                                      else sorted(set(wanted)))

    pruned = set(skip or ())
    trace.metadata["n_pruned"] = len(pruned & set(wanted))

    started = time.perf_counter()
    in_a_row = 0
    for i, col in enumerate(wanted, start=1):
        if col in pruned:
            rec = ColumnRecord(source_column=col, stop_reason=STOP_PRUNED)
            trace.columns.append(rec)
            if on_column is not None:
                on_column(rec)
            continue
        try:
            rec = pager.run_column(col, source[col], rankings[col], target_text)
        except BlockBudgetExceeded as exc:
            trace.metadata["wall_s"] = round(time.perf_counter() - started, 2)
            trace.metadata["aborted"] = True
            trace.metadata["abort_reason"] = str(exc)
            raise BlockBudgetExceeded(str(exc), trace) from exc
        trace.columns.append(rec)
        if on_column is not None:
            on_column(rec)
        if verbose:
            where = (f"accepted block {rec.accepted_block + 1}"
                     if rec.accepted else rec.stop_reason)
            print(f"  [{i}/{len(wanted)}] {col}: {rec.api_calls} call(s), {where}")

        in_a_row = in_a_row + 1 if rec.stop_reason == STOP_ERROR else 0
        if max_consecutive_errors and in_a_row >= max_consecutive_errors:
            trace.metadata["wall_s"] = round(time.perf_counter() - started, 2)
            trace.metadata["aborted"] = True
            trace.metadata["abort_reason"] = rec.error
            raise RunAborted(
                f"{in_a_row} column(s) in a row failed and the run stopped at "
                f"{col!r} ({i}/{len(wanted)}). The last error was:\n"
                f"  {rec.error}\n"
                "Fix it and re-run with resume=True.",
                trace,
            )

    trace.metadata["wall_s"] = round(time.perf_counter() - started, 2)
    # Parameters the backend refused, so the trace does not record them as sent.
    dropped = sorted(getattr(pager.client, "dropped", ()) or ())
    if dropped:
        trace.metadata["dropped_params"] = dropped
    return trace
