"""A notebook's run configuration, and the setup that follows from it.

A retriever label is not a cache key: `run.retrieval_order` checks the cached
order's own configuration. The target-text policy rides in the arm variant,
except for `magneto`, so the ported runs keep their names. One client is built
per set of options, not per provider.

On Ollama, `think=False` and `num_ctx` both change the answer: without the
first a reasoning model can return empty content, and the default `num_ctx`
truncates a 20-candidate prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from . import pruning
from .llm import build_client, canonical_model, load_env

from . import cache, inventory, run

# Magneto's reranker target text, whatever the retriever used.
MAGNETO_TARGET = dict(encoding_mode="header_values_default",
                      sampling_mode="priority_sampling", sampling_size=10)

# The knob each arm sweeps, and the column its rows are labelled by.
LEVEL_KEY = {"rerank": "top_k", "gated": "top_k", "paged": "threshold"}


@dataclass
class Bench:
    """One notebook's configuration."""

    corpus: str
    pairs: Sequence[dict]
    results: Path
    # label -> the four knobs that decide the retrieval order.
    retrievers: dict[str, dict]
    # provider, model, and optionally variant / prompt_layout / client options.
    models: Sequence[dict]
    target_text: str = "match_retriever"
    seed: int = 0
    reps: int = 1
    window: int = 20
    top_k: int = 20
    # What a paged walk does with an unreadable reply ("stop" or "fallback").
    # The request is the same either way, so it is not in the filename.
    on_unparseable: str = "stop"
    # Re-ask an unreadable block up to this many times, with the format spelled
    # out. This changes the request, so it is tagged into the variant.
    max_parse_retries: int = 0
    # None, any of pruning.POLICIES, or a combination. In the run filename.
    prune: Any = None
    dry_run: bool = True
    resume: bool = True
    use_cache: bool = True
    verbose: bool = False
    # Per pair, not per sweep: the counter lives on the Pager, and one is built
    # for each pair.
    max_total_tokens: int = 50_000_000
    # Keep the model's reply on every block. Without it a parser bug can only
    # be repaired by asking every affected column again, which has already
    # cost a full rerun once.
    capture_raw: bool = True

    _clients: dict = field(default_factory=dict, repr=False)
    _orders: dict = field(default_factory=dict, repr=False)
    _targets: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.results = Path(self.results)
        self.results.mkdir(parents=True, exist_ok=True)

    @property
    def names(self) -> list[str]:
        return [p["name"] for p in self.pairs]

    def target_config(self, retriever: str) -> dict:
        """What the prompt shows for each target column."""
        if self.target_text == "magneto":
            return dict(MAGNETO_TARGET)
        cfg = self.retrievers[retriever]
        return {k: cfg[k] for k in
                ("encoding_mode", "sampling_mode", "sampling_size")}

    def arm(self, model: dict) -> str:
        """
        The filename identity of one `models` entry: provider, model, and a
        variant carrying the target-text policy and any `retry{n}`.
        """
        variant = model.get("variant", "")
        layout = model.get("prompt_layout", "standard")
        if (layout != "standard"
                and layout.replace("_", "-") not in variant.replace("_", "-")):
            raise ValueError(
                f"prompt_layout={layout!r} changes the paged prompt, so it must "
                f"be named in the model's variant (e.g. variant="
                f"{layout.replace('_', '-')!r}); without it the run would land "
                "on the standard layout's files.")
        if self.target_text != "magneto":
            tag = f"tgt-{self.target_text.replace('_', '-')}"
            variant = f"{variant}__{tag}" if variant else tag
        # `max_parse_retries` is deliberately not in the tag. A retry only
        # changes a request after a reply fails to parse, so the two runs are
        # the same measurement for every column that parsed. Splitting them
        # into separate files meant one had to be pre-filled from the other,
        # which is the whole reason `reseed` existed. The re-ask now lands in
        # this file, and `run._to_redo` decides which columns it covers.
        provider, name = self.recorded(model)
        return cache.arm_tag(provider, name, variant)

    @staticmethod
    def recorded(model: dict) -> tuple[str, str]:
        """
        The (provider, model) a run is filed under. The same as the transport
        unless `record_as` files a gateway call under the vendor's own name,
        which is only allowed when the gateway id names that vendor.
        """
        provider, name = model["provider"], model["model"]
        recorded = model.get("record_as")
        if not recorded:
            return provider, name

        if recorded not in cache.PROVIDERS:
            raise ValueError(
                f"record_as must be one of {cache.PROVIDERS}, got {recorded!r}")
        vendor = name.partition("/")[0] if "/" in name else ""
        if vendor and vendor.lower() != recorded:
            raise ValueError(
                f"{name!r} is served by {vendor!r}, so it cannot be recorded as "
                f"{recorded!r}. record_as is for one checkpoint reached two "
                "ways.")
        return recorded, canonical_model(name)

    def client(self, model: dict):
        opts = {"seed": self.seed, **model.get("client", {})}
        key = (model["provider"], tuple(sorted(opts.items())))
        if key not in self._clients:
            self._clients[key] = build_client(model["provider"], **opts)
        return self._clients[key]

    def orders(self, rebuild: bool = False) -> dict[tuple[str, str], dict]:
        """(retriever, pair) -> complete ranking, embedded once and cached."""
        for label, cfg in self.retrievers.items():
            for i, pair in enumerate(self.pairs, start=1):
                if (label, pair["name"]) in self._orders and not rebuild:
                    continue
                started = time.perf_counter()
                self._orders[label, pair["name"]] = run.retrieval_order(
                    pair, results=self.results, corpus=self.corpus,
                    retriever=label, rebuild=rebuild, **cfg)
                print(f"[{label} {i}/{len(self.pairs)}] {pair['name']}: "
                      f"{len(self._orders[label, pair['name']])} x "
                      f"{pair['target_df'].shape[1]} targets "
                      f"({time.perf_counter() - started:.1f}s)")
        return self._orders

    def targets(self) -> dict[tuple[str, str], dict]:
        """(retriever, pair) -> target column text."""
        if not self._targets:
            self._targets = {
                (label, p["name"]): run.target_text(
                    p["target_df"], seed=self.seed, **self.target_config(label))
                for label in self.retrievers for p in self.pairs}
        return self._targets

    def banner(self) -> None:
        keys = load_env()
        n_src = sum(p["source_df"].shape[1] for p in self.pairs)
        print(f"env        {keys or 'nothing loaded from .env'}")
        print(f"corpus     {self.corpus}: {len(self.pairs)} studies, "
              f"{n_src} source columns")
        print(f"retrievers {list(self.retrievers)}")
        print(f"models     {[self.arm(m) for m in self.models]}")
        print(f"target     {self.target_text}")
        print(f"prune      {pruning.tag(self.prune) or 'none'}")
        if self.max_parse_retries:
            print(f"retries    {self.max_parse_retries} on an unreadable reply"
                  "   (tagged into the variant)")
        print(f"repeats    {self.reps}   runs -> {self.results}")
        print(f"\nDRY_RUN = {self.dry_run}"
              + ("   (nothing will be sent)" if self.dry_run
                 else "   *** THIS WILL SEND REQUESTS ***"))

    def inventory(self, kind: str) -> pd.DataFrame:
        """
        Every run of `kind` on disk under this Bench's prune, including ones
        unsafe to score. This frame decides what still needs sending, so a
        dropped run would be quoted, and paid for, again.
        """
        inv = inventory.check(inventory.scan(
            self.results, corpus=self.corpus, pairs=self.names, kinds=(kind,)),
            max_fallback=1.0, drop_errored=False)
        if "prune" in inv.columns:
            inv = inv[inv["prune"] == pruning.tag(self.prune)]
        return inv.reset_index(drop=True)

    def coverage(self, kind: str, inv: pd.DataFrame | None = None) -> pd.DataFrame:
        inv = self.inventory(kind) if inv is None else inv
        print(f"{len(inv)} usable {kind} run(s) on disk\n")
        return inventory.coverage(inv, self.names, kind=kind, reps=self.reps)

    def cost(self, kind: str, inv: pd.DataFrame | None = None,
             levels: Iterable[float] = (0.0,)) -> pd.DataFrame:
        """
        What is left to run and what it would cost, in columns and LLM calls.
        For paged the worst case is common: a column with no match sees every
        block.
        """
        inv = self.inventory(kind) if inv is None else inv
        src = {cache.fs_safe(p["name"]): p["source_df"].shape[1]
               for p in self.pairs}
        tgt = {cache.fs_safe(p["name"]): p["target_df"].shape[1]
               for p in self.pairs}

        key = LEVEL_KEY[kind]
        rows = []
        for label in self.retrievers:
            for m in self.models:
                tag = self.arm(m)
                for level in (levels if kind == "paged" else [self.top_k]):
                    where = dict(retriever=label, provider=self.recorded(m)[0],
                                 arm=cache.split_arm(tag)[1],
                                 prune=pruning.tag(self.prune), **{key: level})
                    if kind == "paged":
                        where["window"] = self.window
                    left = inventory.todo(inv, self.names, kind=kind,
                                          reps=self.reps, **where)
                    # Runs on disk but missing columns.
                    part = inventory.short(inv, src, kind=kind,
                                           reps=self.reps, **where)
                    # Failed columns inside runs on disk; only those are resumed.
                    here = inv[inv["kind"] == kind]
                    for k, v in where.items():
                        if k in here.columns:
                            here = here[here[k] == v]
                    n_retry = int(here["n_retryable"].sum()) \
                        if "n_retryable" in here.columns else 0
                    n_stuck = int(here["n_stuck"].sum()) \
                        if "n_stuck" in here.columns else 0
                    # An unreadable reply is re-asked only with retries on:
                    # the same request returns the same reply, but a retry
                    # spells the answer format out. `run._to_redo` decides the
                    # same way, so the quote matches what the sweep sends.
                    if self.max_parse_retries:
                        n_retry, n_stuck = n_retry + n_stuck, 0
                    fresh = sum(src[p] for p, _ in left) + sum(part.values())
                    rows.append({
                        "retriever": label, "model": tag, key: level,
                        "runs_todo": len(left),
                        "runs_partial": len(part),
                        "columns_todo": fresh + n_retry,
                        # Unreadable replies left out, with retries off.
                        "columns_stuck": n_stuck,
                        "calls_best": fresh + n_retry,
                        "calls_worst": sum(
                            run.worst_case_calls(kind, src[p], tgt[p],
                                                 self.window)
                            for p, _ in left) + sum(
                            run.worst_case_calls(kind, n, tgt[p], self.window)
                            for (p, _), n in part.items()) + run.worst_case_calls(
                            kind, n_retry, max(tgt.values(), default=0),
                            self.window)})

        out = pd.DataFrame(rows)
        print(f"{out['runs_todo'].sum()} run(s) to do"
              + (f", {out['runs_partial'].sum()} partial run(s) to finish"
                 if out["runs_partial"].sum() else ""))
        print(f"  columns    {out['columns_todo'].sum():>9,}")
        print(f"  best case  {out['calls_best'].sum():>9,} LLM calls")
        print(f"  worst case {out['calls_worst'].sum():>9,} LLM calls")
        stuck = int(out["columns_stuck"].sum())
        if stuck:
            print(f"  {stuck} column(s) have an unreadable reply that a retry "
                  "would not change. They are not counted above, and `run` "
                  "reports their pairs as cached.")
        return out

    def run(self, kind: str, levels: Iterable[float] = (0.0,),
            on_run: Callable | None = None) -> pd.DataFrame:
        """
        Every retriever, model, level, study and repeat, one row per run.
        Study-major within a configuration. A poisoned cache entry is reported
        and skipped.
        """
        odd = [m["model"] for m in self.models
               if m.get("prompt_layout", "standard") != "standard"]
        if kind != "paged" and odd:
            raise ValueError(
                f"{kind} ignores prompt_layout, which {odd} set; running it "
                f"would repeat the standard {kind} requests under a new name. "
                "Run only paged for them.")
        levels = list(levels) if kind == "paged" else [self.top_k]
        orders, targets = self.orders(), self.targets()
        rows = []

        for label in self.retrievers:
            for m in self.models:
                tag, client = self.arm(m), self.client(m)
                for pair in self.pairs:
                    for level in levels:
                        for rep in range(1, self.reps + 1):
                            row = self._one(kind, pair, label, m, tag, client,
                                            level, rep, orders, targets)
                            if row is None:
                                continue
                            rows.append(row)
                            (on_run or self._print)(row)

        return pd.DataFrame(rows)

    def _one(self, kind, pair, label, m, tag, client, level, rep, orders,
             targets) -> dict | None:
        started = time.perf_counter()
        # Rerank takes no `on_unparseable`: its fallback already handles every
        # uncommitted column.
        if kind == "paged":
            knobs = {"threshold": level, "window": self.window,
                     "prompt_layout": m.get("prompt_layout", "standard"),
                     "on_unparseable": self.on_unparseable,
                     "max_parse_retries": self.max_parse_retries}
        elif kind == "gated":
            knobs = {"top_k": level, "on_unparseable": self.on_unparseable,
                     "max_parse_retries": self.max_parse_retries}
        else:
            knobs = {"top_k": level,
                     "max_parse_retries": self.max_parse_retries}
        try:
            payload, cached = run.run_pair(
                pair, kind=kind, client=client, model=m["model"],
                results=self.results, corpus=self.corpus, retriever=label,
                arm=tag, rep=rep, prune=self.prune, provider=m["provider"],
                rankings=orders[label, pair["name"]],
                targets=targets[label, pair["name"]],
                resume=self.resume, use_cache=self.use_cache,
                dry_run=self.dry_run, verbose=self.verbose,
                capture_raw=self.capture_raw,
                client_options=m.get("client", {}),
                seed=self.seed,
                sampling_size=self.target_config(label)["sampling_size"],
                max_total_tokens=self.max_total_tokens, **knobs)
        except cache.PoisonedCache as exc:
            print(f"!! {tag} {pair['name']}\n{exc}\n")
            return None

        # The prune is in the filename but not in the arm tag.
        return run.summarise(pair["name"], kind, payload, cached=cached,
                             wall_s=time.perf_counter() - started,
                             retriever=label, model=tag, rep=rep,
                             prune=pruning.tag(self.prune),
                             **{LEVEL_KEY[kind]: level})

    @staticmethod
    def _print(r: dict) -> None:
        key = LEVEL_KEY[r["arm"]]
        prune = r.get("prune") or ""
        print(f"{'cached' if r['cached'] else 'ran   '} "
              f"{r['retriever']:<8} {r['model']:<34} {r['pair']:<10} "
              f"r{r['rep']} {key}={r[key]:<5} {r['calls']:>6,} calls  "
              f"{r['committed']}/{r['columns']} committed  "
              f"{r['abstained']} abstained"
              + (f"  p{prune}" if prune else "")
              + (f"  {r['errored']} ERRORED" if r["errored"] else ""))

    def health(self, sweep: pd.DataFrame) -> pd.DataFrame | None:
        """Errored and unanswered runs in a sweep, then totals per configuration."""
        if not len(sweep):
            print("nothing ran")
            return None

        bad = sweep[sweep["errored"] > 0]
        if len(bad):
            print(f"{len(bad)} run(s) have errored columns. Keep RESUME = True "
                  "and re-run the sweep to retry just those:\n")
            print(bad[["retriever", "model", "pair", "rep", "errored",
                       "columns"]].to_string(index=False))
        else:
            print("no errored columns")

        short = sweep[~sweep["complete"]]
        if len(short):
            print(f"\n{len(short)} run(s) have unanswered columns:")
            print(short[["model", "pair", "columns", "committed", "abstained",
                         "errored"]].to_string(index=False))

        # Arm and prune join the keys only when they vary. Each arm has its own
        # level column, so a mixed sweep is grouped by both.
        keys = ["arm"] if sweep["arm"].nunique() > 1 else []
        keys += ["retriever", "model"]
        keys += [k for k in dict.fromkeys(LEVEL_KEY[a] for a in sweep["arm"])
                 if k in sweep.columns]
        if "prune" in sweep.columns and sweep["prune"].nunique() > 1:
            keys.append("prune")
        per = sweep.groupby(keys, dropna=False).agg(
            runs=("pair", "count"), columns=("columns", "sum"),
            calls=("calls", "sum"), committed=("committed", "sum"),
            abstained=("abstained", "sum"), wall_s=("wall_s", "sum"))
        per["calls_per_column"] = (per["calls"]
                                   / per["columns"].clip(lower=1)).round(2)
        return per
