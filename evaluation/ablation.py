"""Retrieval ablations: where does the gold sit?

Every arm inherits the retriever's order, so the gold's rank in it bounds them
all. This sweeps embedding model, serialiser, sampler and sample size, and
caches each configuration's complete ranking (no LLM, but hours of GPU), so
the cache can later answer new questions or hand an order to an arm via
`rankings_for` without embedding again.

The target schema is stored once per pair and rankings as (index, score) into
it. Configurations whose serialised text is identical (`header_only` ignores
the sampler) share one embedding pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .matching import gold_by_source
from schemaoutis.serializers import (
    ENCODING_MODES,
    SAMPLING_MODES,
    build_column_profiles,
)

from schemaoutis import cache

# What identifies one point of the grid.
CONFIG_FIELDS = ("embedding_model", "encoding_mode", "sampling_mode",
                 "sampling_size", "seed")

# Serialisers that emit no values, so the sampler axes collapse.
HEADER_ONLY_MODES = ("header_only",)

DEFAULT_SIZES = (5, 10, 20)
DEFAULT_KS = (1, 5, 10, 20, 50, 100)

# Six decimals is well below the embedding's own noise.
_SCORE_DP = 6

# Bumped when the stored layout changes; an older payload is rebuilt.
FORMAT_VERSION = 2


def normalise_config(cfg: Any) -> dict[str, Any]:
    """
    A configuration as plain Python types. A config rebuilt from a DataFrame
    row carries numpy ints, which would hash to a different id.
    """
    out = {}
    for field in CONFIG_FIELDS:
        value = cfg[field]
        out[field] = (int(value) if field in ("sampling_size", "seed")
                      else str(value))
    return out


def config_id(cfg: Any) -> str:
    """A short stable id for one configuration, used as its filename."""
    payload = json.dumps(normalise_config(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def label(cfg: dict[str, Any]) -> str:
    """A readable name for a configuration."""
    model = str(cfg["embedding_model"]).split("/")[-1]
    enc = str(cfg["encoding_mode"]).replace("header_values_", "").replace("header_", "")
    return f"{model} · {enc} · {cfg['sampling_mode']} · n={cfg['sampling_size']}"


def grid(
    embedding_models: Sequence[str],
    encoding_modes: Sequence[str] = ENCODING_MODES,
    sampling_modes: Sequence[str] = SAMPLING_MODES,
    sampling_sizes: Sequence[int] = DEFAULT_SIZES,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """
    The cross product without combinations that cannot differ (header_only is
    kept once per model). Model-major, so each model is loaded once.
    """
    out: list[dict[str, Any]] = []
    for model in embedding_models:
        for enc in encoding_modes:
            if enc in HEADER_ONLY_MODES:
                # A fixed representative, so two grids never embed the same
                # header_only text under two ids.
                out.append(dict(zip(CONFIG_FIELDS,
                                    (model, enc, SAMPLING_MODES[0],
                                     DEFAULT_SIZES[0], seed))))
                continue
            for sampling, size in product(sampling_modes, sampling_sizes):
                out.append(dict(zip(CONFIG_FIELDS,
                                    (model, enc, sampling, size, seed))))
    return out


def profile_digest(*profile_sets: dict[str, str]) -> str:
    """A digest of the serialised text alone, for sharing embedding passes."""
    h = hashlib.sha256()
    for profiles in profile_sets:
        for name, text in profiles.items():
            h.update(f"{len(str(name))}:{name}\n{len(text)}:{text}\n".encode("utf-8"))
        h.update(b"--\n")
    return h.hexdigest()[:32]


def ablation_path(results: Path, corpus: str, cfg: dict[str, Any]) -> Path:
    """Where one configuration's rankings are stored."""
    return (Path(results) / "ablation" /
            f"{cache.fs_safe(corpus)}__{config_id(cfg)}.json")


def _encode(rankings: dict[str, list[tuple[str, float]]],
            targets: Sequence[str]) -> dict[str, list]:
    """Rankings as (index, score) into `targets`."""
    at = {name: i for i, name in enumerate(targets)}
    return {col: [[at[t], round(float(s), _SCORE_DP)] for t, s in ranked]
            for col, ranked in rankings.items()}


def _decode(encoded: dict[str, list],
            targets: Sequence[str]) -> dict[str, list[tuple[str, float]]]:
    """The inverse of `_encode`."""
    return {col: [(targets[int(i)], float(s)) for i, s in ranked]
            for col, ranked in encoded.items()}


def build_config(
    pairs: Sequence[dict],
    cfg: dict[str, Any],
    *,
    vector_store: Any = None,
    batch_size: int = 64,
    token_budget: int | None = None,
) -> dict[str, Any]:
    """Embed every pair under `cfg` and return the complete orderings."""
    # Imported here so reading a finished ablation needs no torch.
    from schemaoutis.retrieval import BiEncoderIndex

    started = time.perf_counter()
    out: dict[str, Any] = {}
    for pair in pairs:
        index = BiEncoderIndex(
            source=pair["source_df"], target=pair["target_df"],
            model_name=cfg["embedding_model"],
            encoding_mode=cfg["encoding_mode"],
            sampling_mode=cfg["sampling_mode"],
            n_samples=cfg["sampling_size"], seed=cfg["seed"],
            vector_store=vector_store, pair_name=pair["name"],
            batch_size=batch_size,
            **({} if token_budget is None else {"token_budget": token_budget}),
        )
        rankings = {col: index.full_ranking(col) for col in index.source_columns}
        out[pair["name"]] = {
            "targets": list(index.target_columns),
            "rankings": _encode(rankings, index.target_columns),
        }

    return {
        "format": FORMAT_VERSION,
        "config": normalise_config(cfg),
        "config_id": config_id(cfg),
        "label": label(cfg),
        "wall_s": round(time.perf_counter() - started, 2),
        "pairs": out,
    }


def load_config(results: Path, corpus: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """One configuration's payload, or None if missing or from an older format."""
    path = ablation_path(results, corpus, cfg)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT_VERSION or "pairs" not in payload:
        return None
    return payload


def cached_configs(results: Path, corpus: str) -> list[dict[str, Any]]:
    """
    Every configuration cached for this corpus, read from the payloads
    themselves (the filename holds only a digest).
    """
    out = []
    for path in sorted((Path(results) / "ablation")
                       .glob(f"{cache.fs_safe(corpus)}__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT_VERSION or "config" not in payload:
            continue
        out.append(normalise_config(payload["config"]))
    return out


def summarise_cache(pairs: Sequence[dict], *, results: Path, corpus: str,
                    ks: Iterable[int] = DEFAULT_KS, top_k: int = 20,
                    window: int = 20, coverage: float = 0.99,
                    configs: Sequence[dict[str, Any]] | None = None,
                    ) -> pd.DataFrame:
    """
    `summarise` over what is already cached. Embeds and writes nothing, so the
    summary CSV can be rebuilt in minutes.
    """
    frames = []
    for cfg in (configs if configs is not None
                else cached_configs(results, corpus)):
        payload = load_config(results, corpus, cfg)
        if payload is None:
            continue
        rows = gold_rank_rows(payload, pairs, window=window, top_k=top_k)
        if rows:
            frames.append(pd.DataFrame(rows).assign(
                **{k: cfg[k] for k in CONFIG_FIELDS},
                config_id=config_id(cfg), label=label(cfg)))
    if not frames:
        return pd.DataFrame()
    return summarise(pd.concat(frames, ignore_index=True), ks=ks,
                     top_k=top_k, coverage=coverage)


def pair_rankings(payload: dict[str, Any],
                  pair: str) -> dict[str, list[tuple[str, float]]] | None:
    """
    One pair's order from a payload already in memory, or None. Use this, not
    `rankings_for`, when walking a corpus.
    """
    block = payload.get("pairs", {}).get(pair)
    return None if block is None else _decode(block["rankings"],
                                              block["targets"])


def rankings_for(
    results: Path, corpus: str, cfg: dict[str, Any], pair: str
) -> dict[str, list[tuple[str, float]]]:
    """
    One pair's complete ranking under `cfg`, ready for `run_arm`:

        order = ablation.rankings_for(RESULTS, "gdc", best_cfg, "Cao")
        trace = run_arm("paged", pair["source_df"], order, targets, ...)
    """
    payload = load_config(results, corpus, cfg)
    if payload is None:
        raise FileNotFoundError(
            f"{ablation_path(results, corpus, cfg).name} is missing, so "
            f"{label(cfg)} has not been run. Sweep it first."
        )
    if pair not in payload["pairs"]:
        raise KeyError(
            f"{label(cfg)} covers {sorted(payload['pairs'])[:5]}…, not {pair!r}. "
            "The sweep was run over a different set of studies."
        )
    block = payload["pairs"][pair]
    return _decode(block["rankings"], block["targets"])


def gold_rank_rows(
    payload: dict[str, Any],
    pairs: Sequence[dict],
    *,
    window: int = 20,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Where every gold target sits in a stored run, one row per gold pair. `rank`
    is 1-based, or None when the gold names a column the target table lacks.
    """
    rows = []
    for pair in pairs:
        block = payload["pairs"].get(pair["name"])
        if block is None:
            continue
        targets = block["targets"]
        for source_col, golds in gold_by_source(pair["ground_truth"]).items():
            ranked = block["rankings"].get(source_col)
            if ranked is None:
                continue
            position = {targets[int(i)]: r for r, (i, _) in enumerate(ranked)}
            for target_col in sorted(golds):
                i = position.get(target_col)
                rows.append({
                    "pair": pair["name"],
                    "source_column": source_col,
                    "gold_target": target_col,
                    "rank": None if i is None else i + 1,
                    "block": None if i is None else i // window,
                    "within_top_k": None if i is None else i < top_k,
                    "n_targets": len(targets),
                })
    return rows


def run_config(
    pairs: Sequence[dict],
    cfg: dict[str, Any],
    *,
    results: Path,
    corpus: str,
    window: int = 20,
    top_k: int = 20,
    vector_store: Any = None,
    use_cache: bool = True,
    shared: dict[str, dict] | None = None,
    batch_size: int = 64,
    token_budget: int | None = None,
) -> tuple[pd.DataFrame, str, int]:
    """
    One configuration over every pair. Returns (gold rows, status, bytes), with
    status one of cached, extended, shared or embedded. Pass `shared` across
    calls to deduplicate identical text.
    """
    path = ablation_path(results, corpus, cfg)
    payload = load_config(results, corpus, cfg) if use_cache else None

    # A cached payload can cover fewer pairs than asked for; embed the rest.
    if payload is not None:
        missing = [p for p in pairs if p["name"] not in payload["pairs"]]
        if missing:
            extra = build_config(missing, cfg, vector_store=vector_store,
                                 batch_size=batch_size,
                                 token_budget=token_budget)
            payload["pairs"].update(extra["pairs"])
            cache.write(path, payload)
            status = "extended"
        else:
            status = "cached"
    else:
        # Serialising is cheap and tells whether the text is new before any
        # model is loaded.
        digest = profile_digest(*[
            profiles
            for pair in pairs
            for profiles in (
                build_column_profiles(pair["source_df"], cfg["encoding_mode"],
                                      cfg["sampling_mode"], cfg["sampling_size"],
                                      cfg["seed"]),
                build_column_profiles(pair["target_df"], cfg["encoding_mode"],
                                      cfg["sampling_mode"], cfg["sampling_size"],
                                      cfg["seed"]),
            )
        ])
        memo_key = f"{cfg['embedding_model']}|{digest}"

        if shared is not None and memo_key in shared:
            twin = shared[memo_key]
            payload = {**twin, "config": normalise_config(cfg),
                       "config_id": config_id(cfg), "label": label(cfg),
                       "shared_with": twin["config_id"]}
            status = "shared"
        else:
            payload = build_config(pairs, cfg, vector_store=vector_store,
                                   batch_size=batch_size,
                                   token_budget=token_budget)
            status = "embedded"
            if shared is not None:
                shared[memo_key] = payload

        payload["digest"] = digest
        payload["corpus"] = corpus
        cache.write(path, payload)

    rows = gold_rank_rows(payload, pairs, window=window, top_k=top_k)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["wall_s"] = payload.get("wall_s")
    return frame, status, path.stat().st_size


def sweep(
    pairs: Sequence[dict],
    configs: Sequence[dict[str, Any]],
    *,
    results: Path,
    corpus: str,
    window: int = 20,
    top_k: int = 20,
    vector_store: Any = None,
    use_cache: bool = True,
    free_between_models: bool = True,
    on_config=None,
    batch_size: int = 64,
    token_budget: int | None = None,
) -> pd.DataFrame:
    """
    Every configuration as one frame of gold ranks. The embedder is freed when
    the model changes, so only one is on the GPU at a time.
    """
    frames, current_model = [], None
    shared: dict[str, dict] = {}

    for i, cfg in enumerate(configs, start=1):
        if free_between_models and current_model not in (None, cfg["embedding_model"]):
            from schemaoutis.retrieval import free_embedder
            free_embedder()
            shared.clear()          # digests are memoised per model
        current_model = cfg["embedding_model"]

        rows, status, n_bytes = run_config(
            pairs, cfg, results=results, corpus=corpus, window=window,
            top_k=top_k, vector_store=vector_store, use_cache=use_cache,
            shared=shared, batch_size=batch_size, token_budget=token_budget)
        if len(rows):
            frames.append(rows.assign(**{k: cfg[k] for k in CONFIG_FIELDS},
                                      config_id=config_id(cfg),
                                      label=label(cfg)))
        if on_config is not None:
            on_config(i, len(configs), cfg, status, n_bytes)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def k_at_recall(ranks: pd.Series, coverage: float = 0.99,
                n_gold: int | None = None) -> float | None:
    """
    The smallest rank by which `coverage` of the gold is reached, or None if it
    never is (gold the retriever never placed still counts in `n_gold`).
    """
    ordered = sorted(float(v) for v in ranks.dropna())
    n = int(n_gold if n_gold is not None else len(ordered))
    if not n or not ordered:
        return None
    need = math.ceil(coverage * n)
    if need > len(ordered):
        return None
    return ordered[need - 1]


def summarise(ranks: pd.DataFrame, ks: Iterable[int] = DEFAULT_KS,
              top_k: int = 20, coverage: float = 0.99) -> pd.DataFrame:
    """
    One row per configuration: the retriever's recall@k and MRR over gold
    pairs, plus `beyond_top_k` (gold a top_k reranker cannot reach) and
    `mean_blocks` (what paging pays to reach it).
    """
    if ranks.empty:
        return pd.DataFrame()

    ks = list(ks)

    def per_group(g: pd.DataFrame) -> pd.Series:
        r = g["rank"].dropna()
        n = len(g)
        out = {"n_gold": n, "n_missing": int(n - len(r))}
        for k in ks:
            out[f"recall@{k}"] = float((r <= k).sum() / n) if n else 0.0
        out["mrr"] = float((1.0 / r).sum() / n) if n else 0.0
        out["median_rank"] = float(r.median()) if len(r) else float("nan")
        out["p90_rank"] = float(r.quantile(0.90)) if len(r) else float("nan")
        out[f"k{int(coverage * 100)}"] = k_at_recall(r, coverage, n_gold=n)
        out["beyond_top_k"] = float((r > top_k).sum() / n) if n else 0.0
        out["mean_blocks"] = (float(g["block"].dropna().add(1).mean())
                              if len(r) else float("nan"))
        if "wall_s" in g.columns:
            # Constant within a configuration.
            out["wall_s"] = float(g["wall_s"].mean())
        return pd.Series(out)

    keys = [k for k in ["label", *CONFIG_FIELDS, "config_id"]
            if k in ranks.columns]
    summary = ranks.groupby(keys, dropna=False).apply(
        per_group, include_groups=False).reset_index()
    return summary.sort_values("mrr", ascending=False).reset_index(drop=True)


def short_model(name: str) -> str:
    """sentence-transformers/all-mpnet-base-v2 -> all-mpnet-base-v2."""
    return str(name).split("/")[-1]


def best_per_model(summary: pd.DataFrame, by: str = "mrr") -> pd.DataFrame:
    """
    The best configuration for each embedding model, best first, with
    `n_configs` (how many it was chosen from, since a maximum over more draws
    sits higher) and the model's mean wall time over its whole grid.
    """
    if summary.empty:
        return summary
    per_model = summary.groupby("embedding_model")
    counts = per_model.size().rename("n_configs")
    ordered = summary.sort_values(by, ascending=False)
    best = (ordered.groupby("embedding_model", as_index=False, sort=False)
            .head(1).sort_values(by, ascending=False).reset_index(drop=True))
    best = best.merge(counts, on="embedding_model", how="left")
    if "wall_s" in summary.columns:
        best = best.merge(per_model["wall_s"].mean().rename("mean_wall_s"),
                          on="embedding_model", how="left")
    return best
