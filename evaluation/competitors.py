"""The published baselines: Magneto and Valentine's matchers.

They are run over the same pairs as the arms and scored by the same code
(`scoring.score_competitors`). The free ones make no LLM calls; Magneto with
its LLM reranker does, through OpenRouter. Every run is cached.

Results go to `results/competitors/` under flat names, never `results/runs/`,
where `inventory.scan` would read them as arms. Neither `valentine` nor
`magneto` is imported until a matcher is run.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from schemaoutis.cache import fs_safe


def _valentine(name: str, **kw) -> Callable:
    """One of Valentine's matchers, built and run in one call."""
    def match(source, target):
        import valentine.algorithms as alg
        from valentine import valentine_match

        return valentine_match([source, target], getattr(alg, name)(**kw),
                               ["source", "target"])
    return match


@contextmanager
def _bp_on_dicts():
    """
    Magneto 0.2.0's bipartite stage calls `.pop` and `.update` on a valentine
    1.0 `MatcherResults`, which is read-only. Run the same stage on plain dicts.
    """
    import magneto.bp_reranker as bp
    import magneto.magneto as mm

    arrange, results_type = mm.arrange_bipartite_matches, bp.MatcherResults
    mm.arrange_bipartite_matches = lambda m, *a: arrange(dict(m), *a)
    bp.MatcherResults = dict
    try:
        yield
    finally:
        mm.arrange_bipartite_matches, bp.MatcherResults = arrange, results_type


def _magneto(bp: bool = True) -> Callable:
    """
    Magneto, with its GPT reranker off (its default) so nothing is sent.
    `bp` is its bipartite reranker, part of the published method.
    """
    def match(source, target):
        from magneto import Magneto

        m = Magneto(use_bp_reranker=bp)
        if getattr(m, "params", Magneto.DEFAULT_PARAMS).get("use_gpt_reranker"):
            raise RuntimeError(
                "Magneto's GPT reranker is on, which makes API calls. This "
                "module runs baselines that cost nothing to repeat.")
        with _bp_on_dicts():
            return m.get_matches(source, target)
    return match


# name -> match(source_df, target_df). `coma` is COMA 3.0 on schema only and
# `coma-instances` reads values too. Magneto returns a full top-20 for every
# column, so like rerank it never declines.
MATCHERS: dict[str, Callable[[Any, Any], Any]] = {
    "coma": _valentine("Coma", use_instances=False),
    "coma-instances": _valentine("Coma", use_instances=True),
    "cupid": _valentine("Cupid"),
    "distribution-based": _valentine("DistributionBased"),
    "jaccard-levenshtein": _valentine("JaccardDistanceMatcher"),
    "similarity-flooding": _valentine("SimilarityFlooding"),
    "magneto": _magneto(bp=True),
    "magneto-no-bp": _magneto(bp=False),
}


def magneto_tag(retriever: str, bp: bool = True) -> str:
    """Free Magneto on one of our cached orders, e.g. `magneto__qwen`."""
    return f"magneto{'' if bp else '-no-bp'}__{retriever}"


def _parse_magneto_tag(matcher: str) -> tuple[str, bool] | None:
    head, sep, retriever = matcher.partition("__")
    if sep and head in ("magneto", "magneto-no-bp") and retriever:
        return retriever, head == "magneto"
    return None


def _magneto_shortlist(pair: dict, *, runs: Path, corpus: str, retriever: str,
                       bp: bool) -> Any:
    """
    Magneto's matches with its embedding step read from our cached order: its
    exact-name matches, top-20 and bipartite stage run unchanged.
    """
    from magneto import Magneto

    from schemaoutis import cache

    path = cache.retrieval_path(runs, corpus, pair["name"], retriever)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached {retriever} order for {pair['name']}; build it with "
            "run_arms first")
    order = json.loads(path.read_text(encoding="utf-8"))["rankings"]
    m = Magneto(use_bp_reranker=bp, use_gpt_reranker=False, topk=20)

    def from_order():
        for col in m.input_sim_map:
            for t, s in order.get(col, []):
                m.input_sim_map[col][t] = float(s)

    m.apply_embedding_matches = from_order
    with _bp_on_dicts():
        return m.get_matches(pair["source_df"], pair["target_df"])


def pair_columns(key: Any) -> tuple[str, str]:
    """
    (source column, target column) from a result key. Valentine's key is a flat
    `ColumnPair`, read by attribute (by position, `key[0][1]` is a letter of
    "source"); Magneto's is ((table, column), (table, column)).
    """
    if hasattr(key, "source_column"):
        return str(key.source_column), str(key.target_column)
    (_, source), (_, target) = key
    return str(source), str(target)


def rankings_of(result: Any, source_columns: Iterable[str]) -> dict[str, list]:
    """
    A matcher's pair scores as per-column rankings, best first. Every source
    column gets an entry, empty if the matcher proposed nothing for it.
    """
    out: dict[str, list[tuple[str, float]]] = {str(c): [] for c in source_columns}
    for key, score in dict(result).items():
        source, target = pair_columns(key)
        out.setdefault(source, []).append((target, float(score)))
    for col in out:
        out[col].sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def path_for(results: Path, corpus: str, pair: str, matcher: str) -> Path:
    """Where one matcher's run of one pair is stored."""
    return Path(results) / fs_safe(f"{corpus}__{pair}__{matcher}.json")


def run_pair(pair: dict, matcher: str, *, results: Path, corpus: str,
             runs: Path | None = None,
             use_cache: bool = True) -> tuple[dict, bool]:
    """
    One matcher over one pair. Returns (payload, was_cached). A `magneto__<order>`
    matcher reads that cached order from `runs`.
    """
    on_order = _parse_magneto_tag(matcher)
    if matcher not in MATCHERS and on_order is None:
        raise ValueError(f"unknown matcher {matcher!r}; have {sorted(MATCHERS)} "
                         "or magneto__<retriever>")
    if on_order is not None and runs is None:
        raise ValueError(f"{matcher} reads a cached order; pass runs=")

    path = path_for(results, corpus, pair["name"], matcher)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True

    source, target = pair["source_df"], pair["target_df"]
    started = time.perf_counter()
    result = (MATCHERS[matcher](source, target) if on_order is None else
              _magneto_shortlist(pair, runs=runs, corpus=corpus,
                                 retriever=on_order[0], bp=on_order[1]))

    payload = {
        "corpus": corpus, "pair": pair["name"], "matcher": matcher,
        "wall_s": round(time.perf_counter() - started, 2),
        "n_source_columns": int(source.shape[1]),
        "n_target_columns": int(target.shape[1]),
        "n_pairs_proposed": len(dict(result)),
        "rankings": {c: [[t, s] for t, s in r]
                     for c, r in rankings_of(result, source.columns).items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, False


def run(pairs: Sequence[dict], matchers: Sequence[str], *, results: Path,
        corpus: str, runs: Path | None = None, use_cache: bool = True,
        verbose: bool = True) -> list[dict]:
    """
    Every matcher over every pair, one row each, matcher by matcher so an
    interrupted sweep leaves whole matchers finished.
    """
    rows = []
    for matcher in matchers:
        for pair in pairs:
            payload, cached = run_pair(pair, matcher, results=results,
                                       corpus=corpus, runs=runs,
                                       use_cache=use_cache)
            row = {k: v for k, v in payload.items() if k != "rankings"}
            row["cached"] = cached
            row["n_answered"] = sum(1 for r in payload["rankings"].values() if r)
            rows.append(row)
            if verbose:
                print(f"{'cached' if cached else 'ran   '} {matcher:<20} "
                      f"{payload['pair']:<12} "
                      f"{row['n_answered']}/{payload['n_source_columns']} "
                      f"columns answered  {payload['wall_s']:>7.1f}s")
    return rows


def load(results: Path, corpus: str, pair: str, matcher: str) -> dict | None:
    """A finished cached run, or None. A run still in progress is None too."""
    path = path_for(results, corpus, pair, matcher)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("complete", True) else None


def magneto_llm_tag(retriever: str, llm: str, provider: str = "openrouter",
                    bp: bool = True) -> str:
    """The matcher name, which carries everything that changes the answer."""
    return (f"magneto{'' if bp else '-no-bp'}-llm__{retriever}__{provider}__"
            f"{llm.split('/')[-1]}")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def run_magneto_llm(pair: dict, *, results: Path, corpus: str, runs: Path,
                    retriever: str = "qwen",
                    llm: str = "deepseek/deepseek-v4.1-flash", bp: bool = True,
                    workers: int = 8, use_cache: bool = True,
                    verbose: bool = True) -> dict:
    """
    Magneto with its LLM reranker, its shortlist built on our cached order.

    Magneto's exact-name matches, top-20 and (with `bp`) bipartite stage run
    unchanged on the order's scores; its `LLMReranker` then reranks each column
    through OpenRouter, `workers` columns at a time, with its own prompt,
    retries and fallback. Every reply is kept and the cache is written as each
    column finishes, so a rerun resumes. An API error stops new columns and is
    raised once the finished ones are saved.
    """
    import magneto.llm_reranker as reranker_module
    from magneto.llm_reranker import LLMReranker
    from magneto.utils.dataframe_table import DataframeTable
    from magneto.utils.utils import get_samples

    from schemaoutis.llm import load_env

    load_env()
    matcher = magneto_llm_tag(retriever, llm, bp=bp)
    path = path_for(results, corpus, pair["name"], matcher)
    source, target = pair["source_df"], pair["target_df"]
    payload = (json.loads(path.read_text(encoding="utf-8"))
               if use_cache and path.exists() else None)
    if payload is None:
        shortlist = rankings_of(
            _magneto_shortlist(pair, runs=runs, corpus=corpus,
                               retriever=retriever, bp=bp), source.columns)
        payload = {
            "corpus": corpus, "pair": pair["name"], "matcher": matcher,
            "retriever": retriever, "provider": "openrouter", "llm": llm,
            "bp": bp,
            "n_source_columns": int(source.shape[1]),
            "n_target_columns": int(target.shape[1]),
            "candidates": {c: [[t, s] for t, s in r] for c, r in shortlist.items()},
            "llm_calls": {}, "rankings": {}, "fallback": [], "wall_s": 0.0,
            "complete": False,
        }

    src_df = DataframeTable(source, "source").get_df()
    tgt_df = DataframeTable(target, "target").get_df()
    src_values = {c: get_samples(src_df[c], 10) for c in src_df.columns}
    tgt_values = {c: get_samples(tgt_df[c], 10) for c in tgt_df.columns}
    original = reranker_module.completion
    local, lock, stop = threading.local(), threading.Lock(), threading.Event()

    def recording(**kw):
        resp = original(**kw)
        usage = getattr(resp, "usage", None)
        local.calls.append({"messages": kw.get("messages"),
                            "reply": resp.choices[0].message.content,
                            "tokens_in": getattr(usage, "prompt_tokens", None),
                            "tokens_out": getattr(usage, "completion_tokens", None)})
        return resp

    def one(col, cands):
        if stop.is_set():
            return None
        local.calls = []
        reranker = LLMReranker(llm_model=f"openrouter/{llm}")
        ranked = (reranker.rematch(src_df, tgt_df, src_values, tgt_values,
                                   {col: [(t, s) for t, s in cands]})[col]
                  if cands else [])
        parsed = any(reranker._parse_matches(c["reply"] or "") is not None
                     for c in local.calls)
        ranked = sorted(((str(t), float(s)) for t, s in ranked),
                        key=lambda kv: (-kv[1], kv[0]))
        with lock:
            if cands and not parsed:
                payload["fallback"].append(col)
            payload["llm_calls"][col] = local.calls
            payload["rankings"][col] = [[t, s] for t, s in ranked]
            _write(path, payload)
        if verbose:
            print(f"{col:<40} {len(local.calls)} call(s)"
                  + ("" if parsed or not cands else "  FALLBACK"))
        return col

    todo = [(c, cands) for c, cands in payload["candidates"].items()
            if c not in payload["rankings"]]
    started, failure = time.perf_counter(), None
    reranker_module.completion = recording
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for f in as_completed([pool.submit(one, c, cands)
                                   for c, cands in todo]):
                if f.exception() is not None and failure is None:
                    failure = f.exception()
                    stop.set()
    finally:
        reranker_module.completion = original
        with lock:
            payload["wall_s"] = round(payload["wall_s"]
                                      + time.perf_counter() - started, 2)
            _write(path, payload)
    if failure is not None:
        raise failure

    payload["complete"] = True
    payload["n_pairs_proposed"] = sum(len(r) for r in payload["rankings"].values())
    _write(path, payload)
    return payload


def magneto_llm_plan(pairs: Sequence[dict], *, results: Path, corpus: str,
                     runs: Path, retrievers: Sequence[str],
                     llms: Sequence[str], bp: bool = True) -> list[dict]:
    """One row per (retriever, llm, pair): columns done and left, and tokens spent."""
    from schemaoutis import cache

    rows = []
    for retriever in retrievers:
        for llm in llms:
            tag = magneto_llm_tag(retriever, llm, bp=bp)
            for pair in pairs:
                n = int(pair["source_df"].shape[1])
                path = path_for(results, corpus, pair["name"], tag)
                p = (json.loads(path.read_text(encoding="utf-8"))
                     if path.exists() else {})
                calls = [c for cs in (p.get("llm_calls") or {}).values()
                         for c in cs]
                done = len(p.get("rankings") or {})
                rows.append({
                    "retriever": retriever, "llm": llm, "matcher": tag,
                    "pair": pair["name"], "columns": n, "done": done,
                    "todo": n - done,
                    "order_on_disk": cache.retrieval_path(
                        runs, corpus, pair["name"], retriever).exists(),
                    "calls": len(calls),
                    "tokens_in": sum(c.get("tokens_in") or 0 for c in calls),
                    "tokens_out": sum(c.get("tokens_out") or 0 for c in calls),
                })
    return rows


def run_magneto_llm_all(pairs: Sequence[dict], *, results: Path, corpus: str,
                        runs: Path, retrievers: Sequence[str],
                        llms: Sequence[str], bp: bool = True, workers: int = 8,
                        verbose: bool = True) -> list[dict]:
    """`run_magneto_llm` over every pair, configuration by configuration."""
    rows = []
    for retriever in retrievers:
        for llm in llms:
            for pair in pairs:
                if verbose:
                    print(f"--- {magneto_llm_tag(retriever, llm, bp=bp)} / "
                          f"{pair['name']}")
                p = run_magneto_llm(pair, results=results, corpus=corpus,
                                    runs=runs, retriever=retriever, llm=llm,
                                    bp=bp, workers=workers, verbose=False)
                rows.append({"matcher": p["matcher"], "pair": p["pair"],
                             "columns": len(p["rankings"]),
                             "fallback": len(p["fallback"]),
                             "wall_s": p["wall_s"]})
    return rows
