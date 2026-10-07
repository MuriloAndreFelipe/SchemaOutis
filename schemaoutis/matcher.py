"""The four arms, and an entry point shaped like Magneto's.

  retrieval   no LLM: the cosine order.
  rerank      one call over the top `top_k`, Magneto's prompt, cannot decline.
  gated       the same block, with the option to reply NONE.
  paged       one call per window, walking down the whole order.

Every arm reads the same retrieval order, so differences between them come
only from what the model may see and whether it may refuse. rerank against
gated isolates declining; gated against paged isolates depth.

Names follow Magneto (get_matches, get_top_k_matches,
convert_to_valentine_format, DEFAULT_PARAMS) so a Valentine harness can drive
either.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import pandas as pd

from . import pruning
from .paging import Pager, PartialRun, run_paging
from .prompts import (
    MAGNETO_RERANK_SAMPLING_MODE,
    MAGNETO_RERANK_SAMPLING_SIZE,
)
from .trace import (
    STOP_CAPPED,
    STOP_EXHAUSTED,
    STOP_FALLBACK,
    STOP_RETRIEVAL,
    STOP_UNPARSEABLE,
    ColumnRecord,
    RunTrace,
)

ARMS = ("retrieval", "rerank", "gated", "paged")

# The stops `fall_back_to_retrieval` may overwrite. Listed positively so that a
# pruned or declined column is never given the retriever's answer.
FALLBACK_ELIGIBLE = (STOP_EXHAUSTED, STOP_CAPPED, STOP_UNPARSEABLE)

ARM_SETTINGS = {
    "rerank": {"max_blocks": 1, "prompt_layout": "oneshot",
               "accept_threshold": 0.0},
    "gated": {"max_blocks": 1, "prompt_layout": "oneshot_none",
              "accept_threshold": 0.0, "terminal": True},
    "paged": {"max_blocks": None, "prompt_layout": "standard"},
}


def convert_to_valentine_format(
    rankings: dict[str, list[tuple[str, float]]],
    source_name: str = "source",
    target_name: str = "target",
) -> dict[tuple[tuple[str, str], tuple[str, str]], float]:
    """((source_table, source_col), (target_table, target_col)) -> score."""
    return {
        ((source_name, source_col), (target_name, target_col)): float(score)
        for source_col, ranked in rankings.items()
        for target_col, score in ranked
    }


def get_top_k_matches(col_matches, top_k: int) -> list[tuple[str, float]]:
    """The top_k highest-scoring targets, ties broken by name."""
    items = col_matches.items() if isinstance(col_matches, dict) else col_matches
    ordered = sorted(items, key=lambda kv: (-float(kv[1]), str(kv[0])))
    return [(str(name), float(score)) for name, score in ordered[:top_k]]


def run_retrieval(
    source_columns,
    rankings: dict[str, list[tuple[str, float]]],
    top_k: int = 20,
    *,
    metadata: dict[str, Any] | None = None,
) -> RunTrace:
    """The no-LLM arm as a trace. Every column commits."""
    started = time.perf_counter()
    trace = RunTrace(metadata=dict(metadata or {}))
    for col in source_columns:
        ranked = list(rankings.get(col) or [])
        trace.columns.append(ColumnRecord(
            source_column=col,
            ranking=ranked[:top_k] if top_k else ranked,
            stop_reason=STOP_RETRIEVAL,
        ))
    trace.metadata["wall_s"] = round(time.perf_counter() - started, 2)
    return trace


def build_pager(arm: str, client, model: str, **kw) -> Pager:
    """A Pager configured for `arm`; explicit kwargs win."""
    if arm not in ARM_SETTINGS:
        raise ValueError(f"{arm!r} needs no pager; LLM arms are "
                         f"{sorted(ARM_SETTINGS)}")
    return Pager(client, model, **{**ARM_SETTINGS[arm], **kw})


def run_arm(
    arm: str,
    source: pd.DataFrame,
    rankings: dict[str, list[tuple[str, float]]],
    target_text: dict[str, str],
    *,
    client: Any = None,
    model: str = "",
    top_k: int = 20,
    window: int = 20,
    metadata: dict[str, Any] | None = None,
    only_columns: Iterable[str] | None = None,
    skip: Iterable[str] | None = None,
    verbose: bool = False,
    **kw,
) -> RunTrace:
    """
    Run one arm over a ranking and return its trace.

    `rankings` must be the complete order over the target schema; a truncated
    one silently caps paged. `only_columns` restricts the run (for resuming),
    and `skip` columns are recorded as pruned without a call. The retrieval arm
    ignores `skip`.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    meta = {"arm": arm, **(metadata or {})}
    if arm == "retrieval":
        columns = list(only_columns) if only_columns is not None else list(source.columns)
        return run_retrieval(columns, rankings, top_k, metadata=meta)

    if client is None:
        raise ValueError(f"the {arm!r} arm makes LLM calls and needs a client")

    # For the one-block arms the window is the shortlist.
    one_block = arm in ("rerank", "gated")
    pager = build_pager(arm, client, model,
                        window=top_k if one_block else window, **kw)
    try:
        trace = run_paging(source, rankings, target_text, pager, metadata=meta,
                           only_columns=only_columns, skip=skip, verbose=verbose)
    except PartialRun as exc:
        # The stopped run is still written, so it needs the same treatment.
        # Without this its stranded columns keep an unreadable or capped stop
        # reason, which the health check reads as a broken run forever, or as
        # an abstention on the arm that cannot abstain.
        if arm == "rerank" and exc.trace is not None:
            fall_back_to_retrieval(exc.trace, rankings, top_k)
        raise
    # Rerank only: this is what makes it unable to abstain.
    if arm == "rerank":
        fall_back_to_retrieval(trace, rankings, top_k)
    return trace


def fall_back_to_retrieval(trace: RunTrace, rankings, top_k: int) -> int:
    """
    Give each rerank column the model failed the retriever's top_k, as
    Magneto's LLMReranker does after its retries. Such columns are marked
    `rerank_fallback`, not `accepted`. Returns how many.
    """
    n = 0
    for rec in trace.columns:
        if rec.stop_reason not in FALLBACK_ELIGIBLE:
            continue
        rec.ranking = list(rankings.get(rec.source_column) or [])[:top_k]
        rec.stop_reason = STOP_FALLBACK
        n += 1
    trace.metadata["n_fallback"] = n
    return n


class SchemaOutis:
    """
    Match two DataFrames' columns with one of the arms. Usage mirrors
    magneto.Magneto:

        oracle = SchemaOutis(arm="paged", window=20)
        matches = oracle.get_matches(source_df, target_df)   # Valentine format

    Afterwards `rankings` holds the answers, `retrieval_order` the complete
    cosine order, and `trace` what the run did.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        # the arm
        "arm": "paged",
        "top_k": 20,             # rerank/gated: the shortlist
        "window": 20,            # paged: candidates per block
        "accept_threshold": 0.0,  # paged: a block scoring below this is declined
        "max_blocks": None,      # paged: cap the walk; None walks to the end
        # None, not False: False would override gated's terminal=True.
        "terminal": None,
        # None, any of pruning.POLICIES, or a combination. See `pruning`.
        "prune": None,
        # the retriever
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "encoding_mode": "header_values_verbose",
        "sampling_mode": "mixed",
        "sampling_size": 10,
        "vector_store": None,
        # the model
        "client": None,
        "llm_model": "",
        "prompt_layout": None,   # None = the arm's own
        "temperature": 0.0,
        "seed": 0,
    }

    # Forwarded to Pager when not None.
    _PAGER_PARAMS = ("accept_threshold", "max_blocks", "terminal",
                     "prompt_layout", "temperature",
                     "sampling_size", "seed")

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - set(self.DEFAULT_PARAMS)
        if unknown:
            raise TypeError(f"unknown parameter(s): {sorted(unknown)}")
        self.params = {**self.DEFAULT_PARAMS, **kwargs}
        if self.params["arm"] not in ARMS:
            raise ValueError(
                f"unknown arm {self.params['arm']!r}; expected one of {ARMS}")
        self.params["prune"] = pruning.normalise(self.params["prune"])

        self.retrieval_order: dict[str, list[tuple[str, float]]] = {}
        self.rankings: dict[str, list[tuple[str, float]]] = {}
        self.trace: RunTrace | None = None
        self.prune_plan = pruning.PrunePlan()

    @property
    def arm(self) -> str:
        return self.params["arm"]

    def get_retrieval_order(
        self, source: pd.DataFrame, target: pd.DataFrame
    ) -> dict[str, list[tuple[str, float]]]:
        """The biencoder's complete ordering for each source column."""
        from .retrieval import BiEncoderIndex

        p = self.params
        index = BiEncoderIndex(
            source=source, target=target,
            model_name=p["embedding_model"], encoding_mode=p["encoding_mode"],
            sampling_mode=p["sampling_mode"], n_samples=p["sampling_size"],
            seed=p["seed"], vector_store=p["vector_store"],
        )
        return {col: index.full_ranking(col) for col in index.source_columns}

    def get_target_text(self, target: pd.DataFrame) -> dict[str, str]:
        """
        The line the prompt shows for each target column, sampled with
        priority_sampling as Magneto's reranker does.
        """
        from .serializers import build_column_profiles

        return build_column_profiles(
            target,
            encoding_mode=self.params["encoding_mode"],
            sampling_mode=MAGNETO_RERANK_SAMPLING_MODE,
            n_samples=MAGNETO_RERANK_SAMPLING_SIZE,
            seed=self.params["seed"],
        )

    def get_matches(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        source_name: str = "source",
        target_name: str = "target",
    ) -> dict[tuple[tuple[str, str], tuple[str, str]], float]:
        """
        Match `source`'s columns against `target`'s, in Valentine format.

        Declined columns contribute nothing; use `trace` to tell them from
        pruned ones, and `retrieval_order` to fill them if needed. A source
        prune skips columns, while a target prune shrinks the order itself, so
        it applies to the retrieval arm too.
        """
        if source.shape[1] == 0 or target.shape[1] == 0:
            return {}

        p = self.params
        llm = p["arm"] != "retrieval"
        self.retrieval_order = self.get_retrieval_order(source, target)
        self.prune_plan = pruning.plan(self.retrieval_order, p["prune"])
        order = self.prune_plan.apply(self.retrieval_order)

        pager_kw = ({k: p[k] for k in self._PAGER_PARAMS if p[k] is not None}
                    if llm else {})
        self.trace = run_arm(
            p["arm"], source, order,
            self.get_target_text(target) if llm else {},
            client=p["client"], model=p["llm_model"],
            top_k=p["top_k"], window=p["window"],
            skip=self.prune_plan.skip(),
            metadata={
                "embedding_model": p["embedding_model"],
                "encoding_mode": p["encoding_mode"],
                "sampling_mode": p["sampling_mode"],
                "n_source_columns": source.shape[1],
                "n_target_columns": target.shape[1],
                "prune": self.prune_plan.to_dict(),
            },
            **pager_kw,
        )

        self.rankings = {c.source_column: list(c.ranking)
                         for c in self.trace.columns if c.committed}
        return convert_to_valentine_format(self.rankings, source_name,
                                           target_name)
