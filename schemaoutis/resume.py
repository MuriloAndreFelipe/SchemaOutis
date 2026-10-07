"""Resuming a run that stopped partway.

A hard API failure does not stop a run: each remaining column records
stop_reason="error" and the payload looks otherwise healthy. `errored_columns`
finds those, and `merge_resumed` splices a re-run of them into the old payload.
This is sound because every column's walk is independent of the others.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from .llm import canonical_model
from .prompts import VERDICT_REJECT
from .trace import (ABSTAINED_STOPS, ANSWERED_STOPS, COMMITTED_STOPS,
                    STOP_ERROR, STOP_FALLBACK, STOP_UNPARSEABLE,
                    STOP_UNPARSEABLE_FALLBACK)

# Fields that must agree between a payload and its resume. The model is
# compared separately through `canonical_of`, so the same checkpoint reached
# over two transports still matches.
#
# `max_parse_retries` is deliberately absent. It changes a request only after a
# reply fails to parse, so every column that parsed was sent the same bytes
# under either policy, and the two are the same measurement. Treating it as an
# identity refused to splice a run made without retries into one asking for
# them, which is exactly the resume the policy exists to allow. Each column
# records its own `n_attempts`, so what happened stays visible.
IDENTITY_FIELDS = (
    "window", "accept_threshold", "max_blocks", "on_unparseable",
    "prompt_layout", "temperature", "sampling_mode", "sampling_size",
    "seed",
)

FAILED_STOPS = (STOP_ERROR, STOP_UNPARSEABLE)

# Only API errors are retried. An unreadable reply at temperature 0 comes back
# the same, so retrying it never converges.
RETRYABLE_STOPS = (STOP_ERROR,)


class ResumeMismatch(RuntimeError):
    """The resume ran under a different configuration."""


def _columns(payload: Mapping[str, Any]) -> list[dict]:
    return list(payload.get("trace", {}).get("columns", []))


def _config(payload: Mapping[str, Any]) -> dict:
    return (payload.get("trace", {}).get("metadata", {}) or {}).get("config", {}) or {}


def errored_columns(payload: Mapping[str, Any]) -> list[str]:
    """Columns whose walk ended without an answer, in trace order."""
    return [str(c.get("source_column", "")) for c in _columns(payload)
            if c.get("stop_reason") in FAILED_STOPS]


def retryable_columns(payload: Mapping[str, Any]) -> list[str]:
    """The errored columns another attempt could change: the API errors."""
    return [str(c.get("source_column", "")) for c in _columns(payload)
            if c.get("stop_reason") in RETRYABLE_STOPS]


def unparseable_columns(payload: Mapping[str, Any]) -> list[str]:
    return [str(c.get("source_column", "")) for c in _columns(payload)
            if c.get("stop_reason") == STOP_UNPARSEABLE]


def fallback_columns(payload: Mapping[str, Any]) -> list[str]:
    """
    Columns answered by the retriever because the model's reply was unusable.

    A column the model declined is not one of them. `rerank` cannot abstain,
    so a clean NONE is answered with the retriever's shortlist by design, and
    counting it as an unreadable reply would exclude a run for doing exactly
    what the arm specifies. The verdict says which happened, not the stop
    reason: both end as a fallback, and a clean NONE carries an "unparseable"
    parse note too, because the score parser finds no pairs in it.
    """
    out = []
    for c in _columns(payload):
        if c.get("stop_reason") not in (STOP_FALLBACK, STOP_UNPARSEABLE_FALLBACK):
            continue
        blocks = c.get("blocks") or []
        if blocks and blocks[-1].get("verdict") == VERDICT_REJECT:
            continue
        out.append(str(c.get("source_column", "")))
    return out


def error_messages(payload: Mapping[str, Any]) -> list[str]:
    """Distinct error strings recorded on the failed columns."""
    seen: list[str] = []
    for c in _columns(payload):
        if c.get("stop_reason") not in FAILED_STOPS:
            continue
        msg = str(c.get("error") or "").strip()
        if msg and msg not in seen:
            seen.append(msg)
    return seen


def is_usable(payload: Mapping[str, Any]) -> bool:
    """
    True when no column is waiting on a retry. An unparseable column does not
    count against it: that is the model's answer, not a missing one.
    """
    return not retryable_columns(payload)


def canonical_of(payload: Mapping[str, Any]) -> str | None:
    """
    The checkpoint a payload came from, independent of transport.

    The fallbacks record the wire id, which carries the vendor namespace on a
    gateway: the same checkpoint reads `anthropic/claude-sonnet-5` through
    OpenRouter and `claude-sonnet-5` direct. Stripping the namespace is what
    lets a run started on one transport be resumed on the other, which is the
    whole point of `record_as`.
    """
    meta = payload.get("trace", {}).get("metadata", {}) or {}
    name = (meta.get("canonical_model") or _config(payload).get("model")
            or meta.get("model"))
    return canonical_model(name) if name else name


def identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The configuration two payloads must share to be spliced."""
    cfg = _config(payload)
    out = {k: cfg.get(k) for k in IDENTITY_FIELDS}
    out["canonical_model"] = canonical_of(payload)
    # From the trace metadata, not the pager config: the prune is decided
    # before the pager exists.
    meta = payload.get("trace", {}).get("metadata") or {}
    out["prune"] = tuple((meta.get("prune") or {}).get("rules", ()))
    return out


# Config fields that change the request without changing the filename, so a
# cache hit is the only thing standing between two of them sharing a file.
# Everything else that matters (model, retriever, window, threshold, prune,
# rep) is in the path already and is not re-checked here.
#
# A field compared here must be one the running code still records. A recorded
# value meets `None` when the field is retired, and the check below reads that
# as drift and refuses the file.
UNKEYED_FIELDS = ("client",)


class ConfigMismatch(RuntimeError):
    """A cached run was made under a different configuration."""


def check_cached(payload: Mapping[str, Any], wanted: Mapping[str, Any], *,
                 name: str = "") -> None:
    """
    Raise unless a cached run was made under the configuration now asked for.

    Only the un-keyed fields are compared, and a field the stored run never
    recorded is unknown rather than different: older runs predate the field
    and must keep resolving.

    `on_unparseable` is deliberately not compared: both surviving values send
    the same bytes and differ only in what is recorded afterwards. A value
    that changes the requests sent would need the check back.
    """
    cfg = _config(payload)
    drift = {k: (cfg.get(k), wanted.get(k)) for k in UNKEYED_FIELDS
             if cfg.get(k) is not None and cfg.get(k) != wanted.get(k)}

    if not drift:
        return
    detail = ", ".join(f"{k}: {x!r} on disk, {y!r} asked for"
                       for k, (x, y) in sorted(drift.items()))
    raise ConfigMismatch(
        f"{name or 'the cached run'} was made under a different "
        f"configuration ({detail}), and that is not in its filename, so "
        "serving it would report one measurement as another. Name the "
        "difference in the model's variant, or delete the file to re-run it."
    )


def check_compatible(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    """Raise unless the two payloads ran under the same configuration."""
    a, b = identity(old), identity(new)
    drift = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    if drift:
        detail = ", ".join(f"{k}: {x!r} -> {y!r}" for k, (x, y) in sorted(drift.items()))
        raise ResumeMismatch(
            f"the resume ran under a different configuration ({detail}). "
            "Re-run the pair instead of splicing."
        )


def _totals_of(columns: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """`RunTrace.totals` recomputed from column dicts."""
    cols = list(columns)
    blocks = [b for c in cols for b in (c.get("blocks") or [])]
    return {
        # Prefer the column's own count: runs ported from SchemaSteer record
        # calls per column but carry no block records.
        "api_calls": sum(int(c["api_calls"]) if c.get("api_calls") is not None
                         else len(c.get("blocks") or []) for c in cols),
        "tokens_in": sum(int(b.get("tokens_in") or 0) for b in blocks),
        "tokens_out": sum(int(b.get("tokens_out") or 0) for b in blocks),
        "cache_write": sum(int(b.get("cache_write") or 0) for b in blocks),
        "cache_read": sum(int(b.get("cache_read") or 0) for b in blocks),
    }


def _completeness_of(columns: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """`RunTrace.completeness` recomputed from column dicts."""
    cols = list(columns)
    by_reason: dict[str, int] = {}
    for c in cols:
        stop = str(c.get("stop_reason") or "")
        by_reason[stop] = by_reason.get(stop, 0) + 1
    n = len(cols)
    n_answered = sum(1 for c in cols
                     if c.get("stop_reason") in ANSWERED_STOPS)
    return {
        "n_columns": n,
        "n_accepted": sum(1 for c in cols if c.get("accepted_block") is not None),
        "n_committed": sum(1 for c in cols
                           if c.get("stop_reason") in COMMITTED_STOPS),
        "n_abstained": sum(1 for c in cols if c.get("stop_reason") in ABSTAINED_STOPS),
        "n_answered": n_answered,
        "n_unresolved": n - n_answered,
        "n_errored": by_reason.get(STOP_ERROR, 0),
        "complete": n > 0 and n_answered == n,
        "by_stop_reason": by_reason,
    }


def merge_resumed(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    `old` with `columns` replaced by their records from `new`. Defaults to the
    columns that failed in `old`; nothing else in `new` is taken.
    """
    check_compatible(old, new)

    wanted = list(columns) if columns is not None else errored_columns(old)
    if not wanted:
        return copy.deepcopy(dict(old))

    by_col = {str(c.get("source_column", "")): c for c in _columns(new)}
    missing = [c for c in wanted if c not in by_col]
    if missing:
        raise KeyError(
            f"the resume has no record for {len(missing)} requested column(s): "
            f"{missing[:5]}. Re-run it with only_columns covering them."
        )

    still_broken = [c for c in wanted
                    if by_col[c].get("stop_reason") in FAILED_STOPS]
    merged = copy.deepcopy(dict(old))
    wanted_set = set(wanted)
    had = {str(c.get("source_column", "")) for c in _columns(old)}
    merged["trace"]["columns"] = [
        copy.deepcopy(by_col[str(c.get("source_column", ""))])
        if str(c.get("source_column", "")) in wanted_set else c
        for c in _columns(old)
    ]
    # Columns the old payload never reached, because the run aborted first.
    merged["trace"]["columns"] += [
        copy.deepcopy(by_col[c]) for c in wanted if c not in had
    ]

    new_meta = new.get("trace", {}).get("metadata", {}) or {}
    history = list(merged["trace"]["metadata"].get("resumed", []))
    history.append({
        "n_columns": len(wanted),
        "columns": sorted(wanted),
        "still_errored": sorted(still_broken),
        "provider": new_meta.get("provider"),
        "wire_model": new_meta.get("wire_model"),
        "resume_wall_s": new_meta.get("wall_s"),
    })
    merged["trace"]["metadata"]["resumed"] = history
    merged["trace"]["metadata"]["wall_s"] = round(
        float(merged["trace"]["metadata"].get("wall_s") or 0)
        + float(new_meta.get("wall_s") or 0), 2)
    # Recomputed, since the columns changed.
    merged["trace"]["totals"] = _totals_of(merged["trace"]["columns"])
    merged["trace"]["completeness"] = _completeness_of(
        merged["trace"]["columns"])
    return merged
