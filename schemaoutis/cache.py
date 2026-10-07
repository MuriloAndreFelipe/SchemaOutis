"""Run filenames, and loading a cached run.

A filename holds everything that changes what the model was asked:

    {corpus}__{pair}__{retriever}__retrieval.json
    {corpus}__{pair}__{retriever}__k{top_k}[__p{prune}]__{arm}[__r{rep}]__rerank.json
    {corpus}__{pair}__{retriever}__k{top_k}[__p{prune}]__{arm}[__r{rep}]__gated.json
    {corpus}__{pair}__{retriever}__w{window}__th{threshold}[__p{prune}]__{arm}[__r{rep}]__paged.json

`arm` is provider, model and prompt variant. `prune` and `rep` are omitted at
their defaults, so older runs keep their names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .resume import errored_columns, error_messages, is_usable

# NTFS reads a colon as an alternate data stream: the write succeeds and the
# file on disk is empty.
_UNSAFE = re.compile(r'[<>:"/\\|?*]')

_HEAD = r"^(?P<corpus>[^_]+)__(?P<pair>.+?)__(?P<retriever>[^_]+)"

# The arm before this is non-greedy, so __r2 is read as a repeat rather than as
# part of the model name.
_REP = r"(?:__r(?P<rep>\d+))?"

# Left of the arm, between k/th and a mandatory "__". On the right, next to
# `rep`, it can be absorbed into the model name.
#
# A closed alternation, longest first so `zerosrc` is not read as `zero`. It
# lists every tag `pruning.tag` can spell, and `test_cache_regex_covers_every
# _prune` fails if a rule is added without extending it.
_PRUNE_TAGS = ("mutual-zerosrc-zero", "mutual-zerosrc", "zerosrc-zero",
               "mutual-zero", "zerosrc", "mutual", "zero")
_PRUNE = r"(?:__p(?P<prune>" + "|".join(_PRUNE_TAGS) + r"))?"

STEMS = {
    "retrieval": re.compile(_HEAD + r"__retrieval$"),
    "rerank": re.compile(_HEAD + r"__k(?P<top_k>\d+)" + _PRUNE
                         + r"__(?P<arm>.+?)" + _REP + r"__rerank$"),
    # Same fields as rerank; only the suffix tells them apart.
    "gated": re.compile(_HEAD + r"__k(?P<top_k>\d+)" + _PRUNE
                        + r"__(?P<arm>.+?)" + _REP + r"__gated$"),
    "paged": re.compile(_HEAD + r"__w(?P<window>\d+)__th(?P<threshold>[\d.]+)"
                        + _PRUNE + r"__(?P<arm>.+?)" + _REP + r"__paged$"),
}

_NUMERIC = {"top_k": int, "window": int, "threshold": float, "rep": int}


def fs_safe(name: str) -> str:
    return _UNSAFE.sub("-", str(name))


PROVIDERS = ("anthropic", "openrouter", "ollama", "echo")


def arm_tag(provider: str, model: str, variant: str = "") -> str:
    """The arm's identity as it appears in a filename."""
    parts = [p for p in (provider, model, variant) if p]
    return fs_safe("__".join(parts))


def split_arm(arm: str) -> tuple[str, str]:
    """
    Split the provider off an arm tag:
    "anthropic__claude-sonnet-5__v" -> ("anthropic", "claude-sonnet-5__v").
    Only a known provider is split off.
    """
    head, _, rest = str(arm).partition("__")
    return (head, rest) if head in PROVIDERS and rest else ("", str(arm))


def rep_tag(rep: int) -> str:
    """__r2, __r3, and nothing for the first run."""
    rep = int(rep)
    if rep < 1:
        raise ValueError(f"rep is 1-based; got {rep}")
    return "" if rep == 1 else f"__r{rep}"


def prune_tag(prune: Any = None) -> str:
    """__pmutual, __pzerosrc, __pzero, a dash-joined combination, or nothing.
    Spelled by `pruning.tag`, so the filename cannot disagree with the plan."""
    from .pruning import tag

    name = tag(prune)
    return f"__p{name}" if name else ""


def run_path(results: Path, corpus: str, pair: str, retriever: str,
             kind: str = "paged", *, arm: str = "", window: int = 20,
             threshold: float = 0.0, top_k: int = 20, rep: int = 1,
             prune: Any = None) -> Path:
    """Where a run of `kind` for this pair and configuration belongs."""
    p = prune_tag(prune)
    if kind == "retrieval":
        stem = f"{corpus}__{pair}__{retriever}__retrieval"
    elif kind == "rerank":
        stem = (f"{corpus}__{pair}__{retriever}__k{top_k}{p}"
                f"__{arm}{rep_tag(rep)}__rerank")
    elif kind == "gated":
        stem = (f"{corpus}__{pair}__{retriever}__k{top_k}{p}"
                f"__{arm}{rep_tag(rep)}__gated")
    elif kind == "paged":
        stem = (f"{corpus}__{pair}__{retriever}__w{window}"
                f"__th{threshold:.2f}{p}__{arm}{rep_tag(rep)}__paged")
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {tuple(STEMS)}")
    return Path(results) / (fs_safe(stem) + ".json")


def retrieval_path(results: Path, corpus: str, pair: str, retriever: str) -> Path:
    return run_path(results, corpus, pair, retriever, "retrieval")


def parse_stem(stem: str) -> dict[str, Any] | None:
    """The fields a run filename encodes, or None if it is not a run."""
    for kind, pattern in STEMS.items():
        m = pattern.match(stem)
        if not m:
            continue
        g = {"kind": kind, **m.groupdict()}
        if "rep" in g and g["rep"] is None:
            g["rep"] = 1
        if "prune" in g and g["prune"] is None:
            g["prune"] = ""
        return {k: _NUMERIC[k](v) if k in _NUMERIC else v for k, v in g.items()}
    return None


class PoisonedCache(RuntimeError):
    """The cached run has columns that ended in an API error."""


def write(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def load(path: Path, allow_errors: bool = False) -> dict[str, Any] | None:
    """
    Read a cached run, or None if it is not there. Raises PoisonedCache if any
    column ended in an API error, unless `allow_errors` (used when resuming).
    """
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not allow_errors and not is_usable(payload):
        broken = errored_columns(payload)
        first = (error_messages(payload) or ["(no message recorded)"])[0][:160]
        raise PoisonedCache(
            f"{path.name} has {len(broken)}/"
            f"{len(payload['trace']['columns'])} columns that ended in an API "
            f"error, e.g. {broken[:3]}.\n  {first}\n"
            "Pass resume=True to re-run only those columns, or delete the file."
        )
    return payload
