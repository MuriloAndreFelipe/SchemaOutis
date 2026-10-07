"""One pair, one arm, one configuration, cached on disk.

The retrieval order is cached separately, keyed without a model or threshold,
and every arm reads that one order. Looping over models, levels and repeats is
`bench.Bench.run`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import pruning
from .matcher import ARMS, run_arm
from .paging import TRANSPORT_ONLY, PartialRun, RunAborted
from .resume import (_completeness_of, _totals_of, check_cached,
                     errored_columns, merge_resumed, retryable_columns,
                     unparseable_columns)
from .serializers import build_column_profiles

from . import cache


def _requested_config(arm_kw: Mapping[str, Any]) -> dict[str, Any]:
    """
    The un-keyed part of the configuration a caller is asking for, in the
    shape `Pager.config` records it. Built from the kwargs rather than from a
    Pager, so a cached run can be checked without a client.
    """
    client = {k: v for k, v in sorted((arm_kw.get("client_options") or {}).items())
              if k not in TRANSPORT_ONLY}
    return {"on_unparseable": arm_kw.get("on_unparseable", "stop"),
            "client": client or None}


def target_text(target_df: pd.DataFrame, *, sampling_mode: str,
                sampling_size: int, seed: int, encoding_mode: str) -> dict[str, str]:
    """Target column -> the line the prompt shows for it."""
    return build_column_profiles(
        target_df, n_samples=sampling_size, sampling_mode=sampling_mode,
        seed=seed, encoding_mode=encoding_mode,
    )


def retrieval_order(
    pair: dict,
    *,
    results: Path,
    corpus: str,
    retriever: str,
    embedding_model: str,
    encoding_mode: str,
    sampling_mode: str,
    sampling_size: int,
    seed: int = 0,
    vector_store: Any = None,
    rebuild: bool = False,
    batch_size: int = 64,
    token_budget: int | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    The retriever's complete ordering for each source column, cached on disk.

    `retriever` is a label, not a key: a cached order built with a different
    configuration raises rather than being served.
    """
    path = cache.retrieval_path(results, corpus, pair["name"], retriever)
    asked = {"embedding_model": embedding_model, "encoding_mode": encoding_mode,
             "sampling_mode": sampling_mode, "sampling_size": sampling_size,
             "seed": seed}
    if path.exists() and not rebuild:
        stored = json.loads(path.read_text(encoding="utf-8"))
        drift = {k: (stored.get(k), v) for k, v in asked.items()
                 if k in stored and stored.get(k) != v}
        if drift:
            detail = ", ".join(f"{k}: cached {a!r} != asked {b!r}"
                               for k, (a, b) in sorted(drift.items()))
            raise ValueError(
                f"{path.name} was built with a different retriever ({detail}).\n"
                f"  Give this configuration its own label instead of {retriever!r}, "
                "or pass rebuild=True to overwrite."
            )
        return {k: [(t, float(s)) for t, s in v]
                for k, v in stored["rankings"].items()}

    # Imported here so reading cached runs needs neither torch nor
    # sentence-transformers.
    from .retrieval import BiEncoderIndex

    started = time.perf_counter()
    # Batches are padded to their longest member, so one very wide column can
    # run a GPU out of memory; `batch_size` and `token_budget` bound that.
    index = BiEncoderIndex(
        source=pair["source_df"], target=pair["target_df"],
        model_name=embedding_model, encoding_mode=encoding_mode,
        sampling_mode=sampling_mode, n_samples=sampling_size, seed=seed,
        vector_store=vector_store, pair_name=pair["name"],
        batch_size=batch_size,
        **({} if token_budget is None else {"token_budget": token_budget}),
    )
    rankings = {col: index.full_ranking(col) for col in index.source_columns}
    cache.write(path, {
        "corpus": corpus, "pair": pair["name"], "retriever": retriever,
        "embedding_model": embedding_model, "encoding_mode": encoding_mode,
        "sampling_mode": sampling_mode, "sampling_size": sampling_size,
        "seed": seed, "wall_s": round(time.perf_counter() - started, 2),
        "rankings": {k: [[t, float(s)] for t, s in v] for k, v in rankings.items()},
    })
    return rankings


def worst_case_calls(kind: str, n_columns: int, n_targets: int,
                     window: int) -> int:
    """The most LLM calls an arm could make for one pair."""
    if kind == "retrieval":
        return 0
    if kind in ("rerank", "gated"):
        return n_columns
    return n_columns * -(-n_targets // window)


def _to_redo(payload: dict[str, Any], rankings: dict[str, Any],
             max_parse_retries: int = 0) -> list[str]:
    """
    Columns a new attempt could change: API errors, plus columns the payload
    never reached.

    An unreadable reply joins them only when retries are on. At temperature 0
    the same request returns the same reply, so re-sending it changes nothing;
    a retry is a different request, with the answer format spelled out, and can
    change the answer. Retries are a policy, not an identity, so the re-ask
    lands in this same file rather than a separate one.
    """
    seen = {str(c.get("source_column", ""))
            for c in payload.get("trace", {}).get("columns", [])}
    never_tried = [c for c in rankings if c not in seen]
    redo = retryable_columns(payload)
    if max_parse_retries:
        redo += [c for c in unparseable_columns(payload) if c not in redo]
    return redo + never_tried


def run_pair(
    pair: dict,
    *,
    results: Path,
    corpus: str,
    retriever: str,
    rankings: dict[str, list[tuple[str, float]]],
    kind: str = "paged",
    client: Any = None,
    model: str = "",
    arm: str = "",
    targets: dict[str, str] | None = None,
    top_k: int = 20,
    window: int = 20,
    threshold: float = 0.0,
    rep: int = 1,
    prune: Any = None,
    provider: str = "",
    resume: bool = False,
    use_cache: bool = True,
    dry_run: bool = True,
    verbose: bool = False,
    # Named, not left in `arm_kw`, because the resume needs it to decide
    # whether an unreadable column is worth asking again.
    max_parse_retries: int = 0,
    **arm_kw,
) -> tuple[dict[str, Any], bool]:
    """
    Run one pair and return (payload, was_cached). Nothing is sent while
    `dry_run` is True; the retrieval arm ignores it.
    """
    if kind not in ARMS:
        raise ValueError(f"unknown arm {kind!r}; expected one of {ARMS}")

    path = cache.run_path(results, corpus, pair["name"], retriever, kind,
                          arm=arm, window=window, threshold=threshold,
                          top_k=top_k, rep=rep, prune=prune)

    # Applied as `SchemaOutis.get_matches` applies it.
    plan = pruning.plan(rankings, prune)
    rankings = plan.apply(rankings)

    resume_from = None
    if use_cache and path.exists():
        payload = cache.load(path, allow_errors=resume)
        todo = _to_redo(payload, rankings, max_parse_retries)
        if not todo:
            # A complete hit never reaches `check_compatible`, which only runs
            # when columns are spliced, so this is the only place an un-keyed
            # change can be caught before the file is served.
            if kind != "retrieval":
                check_cached(payload, _requested_config(arm_kw),
                             name=path.name)
            return payload, True
        resume_from = payload

    if dry_run and kind != "retrieval":
        todo = (_to_redo(resume_from, rankings, max_parse_retries)
                if resume_from else list(rankings))
        n = worst_case_calls(kind, len(todo), pair["target_df"].shape[1], window)
        raise RuntimeError(
            f"{path.name} is not cached and dry_run is True. This arm makes "
            f"real LLM calls: {len(todo)} column(s), up to {n:,} calls. "
            "Set dry_run=False to send them."
        )

    only = (_to_redo(resume_from, rankings, max_parse_retries)
            if resume_from is not None else None)
    if only and verbose:
        kept = len(resume_from["trace"]["columns"]) - len(only)
        print(f"  resuming {pair['name']}: {len(only)} column(s) to redo, "
              f"keeping {max(kept, 0)}")

    source = pair["source_df"]
    aborted = None
    try:
        trace = run_arm(
            kind, source, rankings, targets or {},
            client=client, model=model, top_k=top_k, window=window,
            only_columns=only, skip=plan.skip(), verbose=verbose,
            max_parse_retries=max_parse_retries,
            **({"accept_threshold": threshold} if kind == "paged" else {}),
            metadata={"corpus": corpus, "pair": pair["name"],
                      "retriever": retriever, "arm": arm, "provider": provider,
                      "wire_model": model, "rep": rep,
                      "n_source_columns": source.shape[1],
                      "n_target_columns": pair["target_df"].shape[1],
                      "prune": plan.to_dict()},
            **arm_kw,
        )
    except PartialRun as exc:
        # Keep what was answered before the failure, unless nothing was. Every
        # one of those columns was paid for.
        trace, aborted = exc.trace, exc
        if trace is None or not any(c.committed for c in trace.columns):
            raise
    payload = {"trace": trace.to_dict()}

    if resume_from is not None:
        # Splice the columns that came back, never replace the old payload.
        got = {c.source_column for c in trace.columns}
        payload = merge_resumed(resume_from, payload,
                                [c for c in only if c in got])

    cache.write(path, payload)
    left = errored_columns(payload)
    if aborted is not None:
        kept = len(payload["trace"]["columns"]) - len(left)
        raise type(aborted)(
            f"{aborted}\n  Kept {kept} answered column(s) in {path.name}; "
            "pass resume=True once it is fixed to finish the rest.",
            trace,
        ) from aborted
    if left and verbose:
        print(f"  {len(left)} column(s) still errored, saved anyway, re-run to retry")
    return payload, False


def summarise(pair_name: str, kind: str, payload: dict, *, cached: bool,
              wall_s: float, **extra) -> dict[str, Any]:
    """
    One run as a row of the sweep frame. Counts are recomputed from the
    columns, not read from the stored summary, which a splice can leave stale.
    """
    comp = _completeness_of(payload.get("trace", {}).get("columns", []))
    tot = _totals_of(payload.get("trace", {}).get("columns", []))
    return {
        "pair": pair_name, "arm": kind, "cached": cached,
        "columns": comp["n_columns"], "committed": comp.get("n_committed", 0),
        "accepted": comp["n_accepted"], "abstained": comp["n_abstained"],
        "errored": comp["n_errored"], "complete": comp["complete"],
        "calls": tot["api_calls"], "tokens_in": tot["tokens_in"],
        "tokens_out": tot["tokens_out"], "wall_s": round(wall_s, 1), **extra,
    }
