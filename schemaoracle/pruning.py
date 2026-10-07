"""
Deciding what not to compare, from the retrieval order alone.

Three rules, none with a threshold, all computed on locally scaled (CSLS)
similarities from the score matrix the retriever already produced:

  mutual   drops a source column with no mutual nearest neighbour: its best
           target does not rank it first in return.
  zerosrc  drops a source column whose best target is no closer to it than its
           own neighbourhood (CSLS <= 0).
  zero     drops a target column whose best source is no closer to it than its
           own neighbourhood (CSLS <= 0).

Dropping a source column means never asking about it; dropping a target means
never offering it. No labels are read here: the rules see only the
retrieval order, which is what lets them run on an unseen schema pair.

`zerosrc` and `zero` are one test read on each margin of the same corrected
matrix, so both sides may fire at once. `mutual` has no such pair: mutual
nearest neighbour is a matching, so at most min(n_sources, n_targets) targets
are anyone's, and read on the target side of a schema with more targets than
sources it drops most of them by construction.

References
    Conneau et al. Word Translation Without Parallel Data. ICLR 2018.
    Radovanovic et al. Hubs in Space. JMLR 11 (2010).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

# Canonical order, and the order the rules act in: which columns get asked
# before which candidates get offered.
POLICIES = ("mutual", "zerosrc", "zero")
SIDE = {"mutual": "source", "zerosrc": "source", "zero": "target"}

# Neighbourhood size for the CSLS correction, as in Conneau et al. Fixed rather
# than tuned; paper_figures section A.3 sweeps it.
CSLS_K = 10

# How deep `mutual_depth` searches. The rule itself only uses depth 1.
MUTUAL_DEPTH = 10


def score_matrix(rankings: Mapping[str, Sequence[tuple[str, float]]]
                 ) -> tuple[np.ndarray, list[str], list[str]]:
    """
    A retrieval order as a dense sources-by-targets matrix. Raises on a partial
    row, which would corrupt the target-side correction.
    """
    sources = list(rankings)
    if not sources:
        return np.zeros((0, 0), dtype=np.float64), [], []

    targets = [t for t, _ in rankings[sources[0]]]
    at = {t: j for j, t in enumerate(targets)}
    if len(at) != len(targets):
        raise ValueError(f"duplicate target in the ranking of {sources[0]!r}")

    M = np.full((len(sources), len(targets)), np.nan, dtype=np.float64)
    for i, s in enumerate(sources):
        row = rankings[s]
        if len(row) != len(targets):
            raise ValueError(
                f"{s!r} ranks {len(row)} targets, expected {len(targets)}; "
                "the correction is computed over the complete row")
        for t, v in row:
            j = at.get(t)
            if j is None:
                raise ValueError(f"{s!r} ranks unknown target {t!r}")
            M[i, j] = v
    if np.isnan(M).any():
        raise ValueError("incomplete score matrix after filling")
    return M, sources, targets


def _mean_top_k(M: np.ndarray, k: int) -> np.ndarray:
    """Mean of each row's k largest values."""
    k = max(1, min(k, M.shape[1]))
    part = np.partition(M, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def csls(M: np.ndarray, k: int = CSLS_K,
         k_frac: float | None = None) -> np.ndarray:
    """
    Locally scaled similarities, the same shape as `M`:

        csls(s, t) = 2 cos(s, t) - mean cos(s, its k nearest targets)
                                 - mean cos(t, its k nearest sources)

    `k_frac` sets each side's k as a fraction of the other side's size.
    """
    if M.size == 0:
        return M
    n_sources, n_targets = M.shape
    if k_frac is not None:
        k_s = max(1, round(k_frac * n_targets))
        k_t = max(1, round(k_frac * n_sources))
    else:
        k_s = k_t = k
    r_source = _mean_top_k(M, k_s)
    r_target = _mean_top_k(M.T, k_t)
    return 2.0 * M - r_source[:, None] - r_target[None, :]


def ranks(M: np.ndarray) -> np.ndarray:
    """1-based rank of every entry within its row, best first."""
    order = np.argsort(-M, axis=1, kind="stable")
    out = np.empty_like(order)
    rows = np.arange(M.shape[0])[:, None]
    out[rows, order] = np.arange(1, M.shape[1] + 1)[None, :]
    return out


def mutual_depth(M: np.ndarray, depth: int = MUTUAL_DEPTH) -> np.ndarray:
    """
    For each row, the smallest d at which it and some column are in each
    other's top d. Depth 1 is the mutual nearest neighbour; rows with none
    within `depth` get `depth + 1`.
    """
    if M.size == 0:
        return np.zeros(0, dtype=int)
    fwd = ranks(M)
    rev = ranks(M.T).T
    both = np.maximum(fwd, rev)
    out = np.where(both <= depth, both, depth + 1)
    return out.min(axis=1).astype(int)


def normalise(prune: str | Iterable[str] | None) -> tuple[str, ...]:
    """
    A prune request as a validated tuple in canonical order, so
    ("zero", "mutual") and ("mutual", "zero") are the same run. Accepts the
    `tag` spelling ("mutual-zero") too.
    """
    if prune is None:
        return ()
    if isinstance(prune, str):
        rules = tuple(p for p in prune.split("-") if p) if prune else ()
    else:
        rules = tuple(prune)
    unknown = [r for r in rules if r not in POLICIES]
    if unknown:
        raise ValueError(
            f"unknown prune rule(s) {unknown}; expected from {POLICIES}")
    return tuple(r for r in POLICIES if r in set(rules))


def tag(prune: str | Iterable[str] | None) -> str:
    """The filename spelling of a prune request, empty for none."""
    return "-".join(normalise(prune))


def masks(M: np.ndarray, rules: Iterable[str], *, csls_k: int = CSLS_K,
          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Boolean drop masks over sources and targets. Both come from one correction
    of the unfiltered matrix, so combining the rules does not depend on order
    and `mutual-zero` decomposes into its two halves.
    """
    rules = normalise(rules)
    n_s, n_t = M.shape
    drop_s = np.zeros(n_s, dtype=bool)
    drop_t = np.zeros(n_t, dtype=bool)
    if not rules or not M.size:
        return drop_s, drop_t

    C = csls(M, k=csls_k)
    # Both source rules accumulate, so a column either rule drops stays
    # dropped and the two compose the way the margins of `zero` do.
    if "mutual" in rules:
        drop_s |= mutual_depth(C) > 1
    if "zerosrc" in rules:
        drop_s |= C.max(axis=1) <= 0
    if "zero" in rules:
        drop_t |= C.T.max(axis=1) <= 0
    return drop_s, drop_t


@dataclass(frozen=True)
class PrunePlan:
    """
    What a prune decided, by name. `emptied` lists source columns left with no
    candidates by a target prune; `skip()` includes them so they are recorded
    as pruned rather than as a model that declined everything.
    """

    rules: tuple[str, ...] = ()
    csls_k: int = CSLS_K
    sources: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    emptied: tuple[str, ...] = ()
    n_sources: int = 0
    n_targets: int = 0

    @property
    def active(self) -> bool:
        return bool(self.rules)

    @property
    def tag(self) -> str:
        return tag(self.rules)

    @property
    def cuts_targets(self) -> bool:
        """True when the prompts differ from an unpruned run's."""
        return any(SIDE[r] == "target" for r in self.rules)

    def skip(self) -> tuple[str, ...]:
        """Source columns the arm must not ask about."""
        return tuple(dict.fromkeys(self.sources + self.emptied))

    def apply(self, rankings: Mapping[str, Sequence[tuple[str, float]]]
              ) -> dict[str, list[tuple[str, float]]]:
        """
        The order with dropped targets removed. Source columns stay, so the arm
        records them as pruned.
        """
        if not self.targets:
            return {c: list(r) for c, r in rankings.items()}
        gone = set(self.targets)
        return {c: [(t, s) for t, s in r if t not in gone]
                for c, r in rankings.items()}

    def to_dict(self) -> dict:
        """The record stored in `trace.metadata["prune"]`."""
        return {
            "rules": list(self.rules),
            "csls_k": self.csls_k,
            "n_sources": self.n_sources,
            "n_sources_dropped": len(self.sources),
            "n_targets": self.n_targets,
            "n_targets_dropped": len(self.targets),
            "sources": list(self.sources),
            "targets": list(self.targets),
            "emptied": list(self.emptied),
            # The retrieval arm ignores `skip`, so a source prune does not
            # reach it; a target prune does.
            "retrieval_honours_source_prune": False,
        }


def plan(rankings: Mapping[str, Sequence[tuple[str, float]]],
         prune: str | Iterable[str] | None, *,
         csls_k: int = CSLS_K) -> PrunePlan:
    """
    Decide a prune over one pair's retrieval order. With no rules the plan is
    inert and applying it changes nothing.
    """
    rules = normalise(prune)
    if not rules:
        return PrunePlan(n_sources=len(rankings),
                         n_targets=len(next(iter(rankings.values()), ())))

    M, sources, targets = score_matrix(rankings)
    drop_s, drop_t = masks(M, rules, csls_k=csls_k)

    dropped_s = tuple(s for s, drop in zip(sources, drop_s) if drop)
    dropped_t = tuple(t for t, drop in zip(targets, drop_t) if drop)

    # Every row is complete, so a column is emptied exactly when every target
    # was dropped.
    if drop_t.all() and len(targets):
        kept = set(sources) - set(dropped_s)
        emptied = tuple(s for s in sources if s in kept)
    else:
        emptied = ()

    return PrunePlan(rules=rules, csls_k=csls_k, sources=dropped_s,
                     targets=dropped_t, emptied=emptied,
                     n_sources=len(sources), n_targets=len(targets))
