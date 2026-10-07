"""Bench: filenames, clients, cost estimates and the dry-run guard."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from conftest import ScriptedClient, score_first
from schemaoutis import cache, inventory
from schemaoutis.bench import Bench

CORPUS = "gdc"

QWEN = dict(embedding_model="Qwen/Qwen3-Embedding-4B",
            encoding_mode="header_values_xml",
            sampling_mode="mixed", sampling_size=10, seed=0)


@pytest.fixture
def pairs(targets, source):
    target_df = pd.DataFrame({t: [1] for t in targets})
    return [{"name": "Cao", "source_df": source, "target_df": target_df,
             "ground_truth": [("age", "t_0")]}]


@pytest.fixture
def bench(pairs, tmp_path, ranking, monkeypatch):
    b = Bench(corpus=CORPUS, pairs=pairs, results=tmp_path,
              retrievers={"qwen": QWEN},
              models=[dict(provider="anthropic", model="claude-sonnet-5")])
    # No GPU and no network: the order is handed over rather than embedded, and
    # every model gets the same scripted backend.
    monkeypatch.setattr(Bench, "orders",
                        lambda self, rebuild=False: {
                            ("qwen", p["name"]): {c: ranking
                                                  for c in p["source_df"].columns}
                            for p in self.pairs})
    monkeypatch.setattr(Bench, "client",
                        lambda self, m: ScriptedClient(score_first(0.9)))
    return b


# Identity: what a run is called, and what it was shown

def test_target_text_policy_rides_in_the_variant(bench):
    """
    It is not part of the filename grammar, so without this a run showing the
    model different text lands on an existing filename and is served it.
    """
    m = dict(provider="anthropic", model="claude-sonnet-5")
    assert bench.arm(m) == "anthropic__claude-sonnet-5__tgt-match-retriever"

    bench.target_text = "magneto"
    assert bench.arm(m) == "anthropic__claude-sonnet-5"


def test_an_existing_variant_keeps_its_place_before_the_policy_tag(bench):
    m = dict(provider="anthropic", model="claude-sonnet-5",
             variant="oneshot-none")
    assert bench.arm(m).endswith("oneshot-none__tgt-match-retriever")


def test_the_unparseable_policy_does_not_change_the_filename(bench):
    """
    The stop and fallback policies send the same requests, so neither is in the
    filename.
    """
    m = dict(provider="anthropic", model="claude-sonnet-5")
    before = bench.arm(m)
    bench.on_unparseable = "fallback"
    assert bench.arm(m) == before


def test_switching_the_policy_still_serves_the_cache(bench):
    bench.dry_run = False
    first = bench.run("paged", levels=[0.0])
    assert not first["cached"].any()

    bench.on_unparseable = "fallback"
    again = bench.run("paged", levels=[0.0])
    assert again["cached"].all()          # nothing re-sent
    assert len({p.name for p in bench.results.glob("*__paged.json")}) == len(first)


def test_match_retriever_shows_the_text_the_ordering_was_built_from(bench):
    assert bench.target_config("qwen") == dict(
        encoding_mode="header_values_xml", sampling_mode="mixed",
        sampling_size=10)


def test_magneto_target_text_ignores_the_retriever(bench):
    """Magneto's reranker reads its own fixed text whatever ordered the block."""
    bench.target_text = "magneto"
    assert bench.target_config("qwen")["sampling_mode"] == "priority_sampling"


def test_one_client_per_set_of_options_not_per_provider(pairs, tmp_path):
    """
    Two Ollama entries can want different `think` or `num_ctx`. Reusing the
    first model's client for the second sends the wrong ones.
    """
    built = []
    b = Bench(corpus=CORPUS, pairs=pairs, results=tmp_path,
              retrievers={"qwen": QWEN},
              models=[dict(provider="ollama", model="a",
                           client=dict(think=False)),
                      dict(provider="ollama", model="b",
                           client=dict(think=True))])
    b._clients = {}

    def fake_build(provider, **opts):
        built.append((provider, tuple(sorted(opts.items()))))
        return object()

    import schemaoutis.bench as mod
    original, mod.build_client = mod.build_client, fake_build
    try:
        first = b.client(b.models[0])
        again = b.client(b.models[0])
        other = b.client(b.models[1])
    finally:
        mod.build_client = original

    assert first is again            # same options, one client
    assert other is not first        # different `think`, its own client
    assert len(built) == 2


# The sweep

def test_a_sweep_sends_nothing_while_dry_run_is_on(bench):
    """`dry_run` defaults to True so a runner fails noisily, not expensively."""
    with pytest.raises(RuntimeError, match="dry_run is True"):
        bench.run("rerank")


def test_a_rerank_row_is_labelled_by_top_k_and_a_paged_row_by_threshold(bench):
    bench.dry_run = False

    rerank = bench.run("rerank")
    assert "top_k" in rerank.columns and "threshold" not in rerank.columns
    assert set(rerank["top_k"]) == {20}

    paged = bench.run("paged", levels=[0.0, 0.5])
    assert "threshold" in paged.columns and "top_k" not in paged.columns
    assert set(paged["threshold"]) == {0.0, 0.5}


def test_the_sweep_covers_every_model_study_and_repeat(bench, pairs):
    bench.dry_run = False
    bench.reps = 2
    bench.models = [dict(provider="anthropic", model="claude-sonnet-5"),
                    dict(provider="anthropic", model="other")]

    sweep = bench.run("rerank")
    assert len(sweep) == len(bench.models) * len(pairs) * bench.reps
    assert set(sweep["rep"]) == {1, 2}
    assert sweep["model"].nunique() == 2


def test_a_second_sweep_is_served_from_the_cache(bench):
    bench.dry_run = False
    assert not bench.run("rerank")["cached"].any()
    assert bench.run("rerank")["cached"].all()


def test_health_reports_errored_columns_rather_than_scoring_them(bench):
    """
    An errored column is a request that raised, not a model that declined, and
    a run holding one writes a file that otherwise looks healthy.
    """
    bench.dry_run = False
    sweep = bench.run("rerank")
    per = bench.health(sweep)
    assert per is not None
    assert per["calls"].sum() == sweep["calls"].sum()


def test_health_keeps_two_arms_apart(bench):
    """
    Each arm sweeps its own knob, `threshold` for paged and `top_k` for the
    one-call arms, so a frame holding both has both columns. Grouping by the
    first row's key would pool a rerank run and a paged run into one number.
    """
    import pandas as pd

    bench.dry_run = False
    sweep = pd.concat([bench.run("rerank"), bench.run("paged", levels=[0.0])],
                      ignore_index=True)
    per = bench.health(sweep)

    assert "arm" in per.index.names
    assert {"top_k", "threshold"} <= set(per.index.names)
    assert set(per.index.get_level_values("arm")) == {"rerank", "paged"}
    assert per["calls"].sum() == sweep["calls"].sum()


def test_health_of_one_arm_groups_exactly_as_before(bench):
    """The extra keys appear only when they vary."""
    bench.dry_run = False
    per = bench.health(bench.run("rerank"))
    assert list(per.index.names) == ["retriever", "model", "top_k"]


def test_the_filter_is_on_the_row_it_produced(bench):
    """
    `prune` is in the filename but not in the arm tag, so without it two sweeps
    under different filters give rows identical in every printed field.
    """
    bench.dry_run = False
    assert set(bench.run("rerank")["prune"]) == {""}

    bench.prune = "mutual"
    assert set(bench.run("rerank")["prune"]) == {"mutual"}


def test_a_matrix_of_arms_and_filters_stays_separable(bench, pairs, tmp_path,
                                                      ranking, monkeypatch):
    """
    The ARMS x PRUNES matrix of run_arms.ipynb: one Bench per filter, and every
    cell its own row and file.
    """
    benches = {}
    for prune in (None, "mutual"):
        b = Bench(corpus=CORPUS, pairs=pairs, results=tmp_path,
                  retrievers={"qwen": QWEN}, prune=prune, dry_run=False,
                  models=[dict(provider="anthropic", model="claude-sonnet-5")])
        monkeypatch.setattr(type(b), "orders",
                            lambda self, rebuild=False: {
                                ("qwen", p["name"]): {
                                    c: ranking for c in p["source_df"].columns}
                                for p in self.pairs})
        monkeypatch.setattr(type(b), "client",
                            lambda self, m: ScriptedClient(score_first(0.9)))
        benches[b.prune or ""] = b

    jobs = [(arm, tag, benches[tag])
            for arm in ("rerank", "paged") for tag in benches]
    assert len(jobs) == 4

    sweep = pd.concat(
        [b.run(arm, levels=[0.0]) for arm, _, b in jobs], ignore_index=True)

    assert set(zip(sweep["arm"], sweep["prune"])) == {
        ("rerank", ""), ("rerank", "mutual"),
        ("paged", ""), ("paged", "mutual")}

    per = bench.health(sweep)
    assert len(per) == 4, "four configurations, four rows"
    assert set(per.index.names) >= {"arm", "prune", "top_k", "threshold"}
    # Four separate files, so nothing was served another cell's cache.
    assert len(list(tmp_path.glob("*.json"))) == 4

    # Each Bench sees only its own filter. Otherwise `coverage` prints the same
    # table once per filter and reports work this Bench will never do.
    for tag, b in benches.items():
        inv = b.inventory("rerank")
        assert set(inv["prune"]) == {tag}, tag
        assert len(inv) == 1


def test_cost_asks_the_inventory_what_is_already_done(bench):
    todo = bench.cost("rerank")
    assert int(todo["runs_todo"].sum()) == 1

    bench.dry_run = False
    bench.run("rerank")
    assert int(bench.cost("rerank")["runs_todo"].sum()) == 0


def test_cost_prices_a_run_that_is_present_but_short(bench, pairs, tmp_path):
    """
    Presence is not completeness. A payload seeded for a re-ask is on disk with
    columns missing, so pricing by absence alone quoted nothing while `run`
    went on to spend on every missing column.
    """
    import json

    from schemaoutis import cache, inventory

    bench.dry_run = False
    bench.run("rerank")
    assert int(bench.cost("rerank")["columns_todo"].sum()) == 0

    # Drop a column, as the seeding does.
    path = next(tmp_path.glob("*__rerank.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    n_before = len(payload["trace"]["columns"])
    payload["trace"]["columns"] = payload["trace"]["columns"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    quoted = bench.cost("rerank")
    assert int(quoted["runs_todo"].sum()) == 0        # the file is there
    assert int(quoted["runs_partial"].sum()) == 1     # but it is short
    assert int(quoted["columns_todo"].sum()) == 1
    assert n_before == pairs[0]["source_df"].shape[1]



def test_a_fallback_heavy_run_is_still_counted_as_done(bench, monkeypatch):
    """
    The sending inventory keeps fallback-heavy runs, so they are not quoted and
    paid for again.
    """
    monkeypatch.setattr(Bench, "client",
                        lambda self, m: ScriptedClient(lambda p: "no scores"))
    bench.dry_run = False
    bench.run("rerank")

    inv = bench.inventory("rerank")
    assert len(inv) == 1
    assert inv.iloc[0]["n_fallback"] > 0
    assert int(bench.cost("rerank")["runs_todo"].sum()) == 0



def test_run_paths_use_the_arm_tag_the_bench_built(bench):
    bench.dry_run = False
    bench.run("rerank")
    written = {p.name for p in bench.results.glob("*.json")}
    tag = cache.split_arm(bench.arm(bench.models[0]))[1]
    assert any(tag in name and name.endswith("__rerank.json")
               for name in written)


# The same checkpoint through a gateway

def test_a_gateway_run_can_be_filed_under_the_vendor_it_reaches(bench):
    """
    OpenRouter pinned to Anthropic sends the request `provider="anthropic"`
    would have sent. Filing it separately would split one measurement across
    two rows of every design and leave both short of the models they need.
    """
    direct = dict(provider="anthropic", model="claude-sonnet-5")
    gateway = dict(provider="openrouter", model="anthropic/claude-sonnet-5",
                   record_as="anthropic")
    assert bench.arm(gateway) == bench.arm(direct)


def test_without_record_as_the_gateway_keeps_its_own_identity(bench):
    """The default stays what `CLAUDE.md` asks for: the transport is in the name."""
    m = dict(provider="openrouter", model="anthropic/claude-sonnet-5")
    assert bench.arm(m).startswith("openrouter__")


def test_record_as_refuses_to_relabel_a_different_vendor(bench):
    m = dict(provider="openrouter", model="openai/gpt-5", record_as="anthropic")
    with pytest.raises(ValueError, match="served by 'openai'"):
        bench.arm(m)


def test_record_as_refuses_a_provider_no_filename_can_carry(bench):
    """
    `cache.split_arm` reads the head back only when it is a known provider, so
    an invented one would make the whole tag read as a model name.
    """
    m = dict(provider="openrouter", model="anthropic/claude-sonnet-5",
             record_as="Anthropic")
    with pytest.raises(ValueError, match="record_as must be one of"):
        bench.arm(m)


def test_the_transport_survives_in_the_payload_when_it_leaves_the_filename(bench):
    """
    The whole argument for allowing this: the provider moves from the name into
    the metadata rather than being lost, so a merged run can still be taken
    apart afterwards.
    """
    bench.models = [dict(provider="echo", model="anthropic/claude-sonnet-5",
                         record_as="anthropic")]
    bench.dry_run = False
    bench.run("paged", levels=[0.0])

    tag = cache.split_arm(bench.arm(bench.models[0]))[1]
    path = cache.run_path(bench.results, bench.corpus, bench.names[0], "qwen",
                          "paged", arm=bench.arm(bench.models[0]),
                          window=bench.window, threshold=0.0)
    meta = json.loads(path.read_text(encoding="utf-8"))["trace"]["metadata"]
    assert path.name.startswith(f"{bench.corpus}__")
    assert tag.startswith("claude-sonnet-5")
    assert meta["provider"] == "echo"
    assert meta["wire_model"] == "anthropic/claude-sonnet-5"



def test_cost_looks_up_the_name_the_run_is_filed_under(bench):
    """
    `cost` filtered the inventory by the transport while `arm` wrote the
    recorded provider into the filename, so a fully cached corpus reached
    through a gateway read as entirely still to do.
    """
    bench.models = [dict(provider="echo", model="claude-sonnet-5")]
    bench.dry_run = False
    bench.run("paged", levels=[0.0])
    assert bench.cost("paged")["runs_todo"].sum() == 0

    # Same files, reached the other way round. `cost` only reads the inventory,
    # so no gateway client is built and no key is needed.
    bench.models = [dict(provider="openrouter", model="claude-sonnet-5",
                         record_as="echo")]
    assert bench.cost("paged")["runs_todo"].sum() == 0


# The middle arm through the runner

def test_bench_runs_the_gated_arm_and_files_it_apart_from_rerank(bench):
    """
    Both arms sweep top_k and carry the same fields, so only the suffix keeps
    them apart. If it did not, one would be served the other's payload.
    """
    bench.dry_run = False
    rerank = bench.run("rerank")
    gated = bench.run("gated")

    assert len(rerank) == len(gated) == 1
    assert not gated["cached"].any()          # not served the rerank payload
    names = sorted(p.name for p in bench.results.glob("*.json"))
    assert any(n.endswith("__rerank.json") for n in names)
    assert any(n.endswith("__gated.json") for n in names)


def test_gated_costs_one_call_per_column_like_rerank(bench):
    """A second block would make it a different arm, not a stricter one."""
    assert (bench.cost("rerank")["calls_worst"].sum()
            == bench.cost("gated")["calls_worst"].sum())


# A column that failed inside a run already on disk

def _break_one(path, stop_reason):
    """Mark one column of a finished run as having ended without an answer."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trace"]["columns"][0]["stop_reason"] = stop_reason
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["trace"]["columns"][0]["source_column"]


def test_a_failed_column_is_priced_as_one_column_not_as_its_whole_study(
        bench, tmp_path):
    """
    `run_pair` resumes exactly the failed column, so quoting the study it sits
    in over-quotes by two orders of magnitude. It used to: the run was dropped
    from the frame for being errored, which made it look absent.
    """
    bench.dry_run = False
    bench.run("rerank")
    path = next(tmp_path.glob("*__rerank.json"))
    _break_one(path, "error")

    out = bench.cost("rerank")
    n_source = bench.pairs[0]["source_df"].shape[1]
    assert int(out["columns_todo"].sum()) == 1
    assert int(out["runs_todo"].sum()) == 0
    assert n_source > 1                       # the study is bigger than the fix
    assert int(out["columns_stuck"].sum()) == 0


def test_an_unreadable_column_is_reported_as_stuck_rather_than_as_work(
        bench, tmp_path):
    """
    An unreadable column is not retried, so cost reports it as stuck rather
    than as work to do.
    """
    bench.dry_run = False
    bench.run("rerank")
    path = next(tmp_path.glob("*__rerank.json"))
    _break_one(path, "unparseable")

    out = bench.cost("rerank")
    assert int(out["columns_todo"].sum()) == 0
    assert int(out["calls_best"].sum()) == 0
    assert int(out["columns_stuck"].sum()) == 1


def test_the_sweep_frame_keeps_a_run_the_scorer_would_drop(bench, tmp_path):
    """
    The two frames answer different questions. `Bench.inventory` decides what
    to send and must see the run; `inventory.check`'s default is for whoever
    plots and must not.
    """
    from schemaoutis import inventory

    bench.dry_run = False
    bench.run("rerank")
    path = next(tmp_path.glob("*__rerank.json"))
    _break_one(path, "error")

    assert len(bench.inventory("rerank")) == 1
    raw = inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",))
    assert len(inventory.check(raw, verbose=False)) == 0



def test_scan_keeps_only_the_pairs_asked_for(bench, tmp_path):
    bench.dry_run = False
    bench.run("rerank")
    other = next(tmp_path.glob("*__rerank.json"))
    other.with_name(other.name.replace("__Cao__", "__Elsewhere__")).write_text(
        other.read_text(encoding="utf-8"), encoding="utf-8")
    everything = inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",))
    asked = inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",),
                           pairs=["Cao"])
    assert set(everything["pair"]) == {"Cao", "Elsewhere"}
    assert set(asked["pair"]) == {"Cao"}


def test_a_non_standard_layout_must_be_named_in_the_variant(bench):
    m = dict(provider="anthropic", model="claude-sonnet-5",
             prompt_layout="oneshot_none")
    with pytest.raises(ValueError, match="named in the model's variant"):
        bench.arm(m)
    assert "oneshot-none" in bench.arm({**m, "variant": "oneshot-none"})
    assert bench.arm(dict(provider="anthropic", model="claude-sonnet-5",
                          prompt_layout="oneshot_none",
                          variant="oneshot-none")).endswith(
        "oneshot-none__tgt-match-retriever")


def test_only_paged_runs_a_model_with_its_own_layout(bench):
    bench.dry_run = False
    bench.models = [dict(provider="anthropic", model="claude-sonnet-5",
                         prompt_layout="oneshot_none", variant="oneshot-none")]
    with pytest.raises(ValueError, match="ignores prompt_layout"):
        bench.run("gated")
    assert len(bench.run("paged", levels=[0.0]))



def _payload(columns, *, config=None):
    return {"trace": {"metadata": {"config": config or {}}, "columns": columns}}


def _col(name, stop="accepted"):
    return {"source_column": name, "stop_reason": stop, "api_calls": 1,
            "committed": stop == "accepted", "blocks": []}



def test_a_cached_run_made_with_other_client_options_is_not_served():
    from schemaoutis.resume import ConfigMismatch, check_cached

    on_disk = _payload([], config={"prompt_style": "figure5",
                                   "on_unparseable": "stop",
                                   "client": {"think": False}})
    same = {"prompt_style": "figure5", "on_unparseable": "stop",
            "client": {"think": False}}
    check_cached(on_disk, same)

    with pytest.raises(ConfigMismatch, match="client"):
        check_cached(on_disk, {**same, "client": {"think": True}})

    # `on_unparseable` is not compared: both surviving values send the same
    # bytes and differ only in what is recorded afterwards.
    check_cached(on_disk, {**same, "on_unparseable": "fallback"})


def test_a_cached_run_recording_a_retired_field_is_still_served():
    """
    Every non-jev run on disk records `prompt_style: "figure5"`, which the
    running code no longer emits. The comparison skips a field the stored run
    never had, but a field it HAS against a request that does not read as
    drift, so retiring one without dropping it from `UNKEYED_FIELDS` refuses
    all 12,428 of them.
    """
    from schemaoutis.resume import check_cached

    on_disk = _payload([], config={"prompt_style": "figure5",
                                   "on_unparseable": "stop",
                                   "client": {"think": False}})
    check_cached(on_disk, {"on_unparseable": "stop", "client": {"think": False}})


def test_a_run_that_recorded_no_client_options_still_resolves():
    from schemaoutis.resume import check_cached

    # Written before the field existed. Unknown is not the same as different,
    # and treating it as different would reject every older run on disk.
    old = _payload([], config={"prompt_style": "figure5"})
    check_cached(old, {"prompt_style": "figure5", "on_unparseable": "stop",
                       "client": {"think": False}})


def test_a_retry_is_a_policy_not_a_filename(tmp_path):
    """
    A retry only changes a request after a reply fails to parse, so a run with
    retries on is the same measurement as one without for every column that
    parsed. Putting it in the filename split one run across two files, and
    something then had to copy the columns from one to the other.
    """
    b = Bench(corpus="gdc", pairs=[], results=tmp_path, retrievers={}, models=[])
    m = dict(provider="anthropic", model="claude-sonnet-5")
    plain = b.arm(m)
    b.max_parse_retries = 3
    assert b.arm(m) == plain
    assert "retry" not in b.arm(m)


def test_an_unreadable_column_is_asked_again_only_when_retries_are_on():
    """
    At temperature 0 the same request returns the same reply, so re-sending it
    changes nothing; a retry spells the answer format out and can. The re-ask
    lands in the run's own file, which is why no seeding step is needed.
    """
    from schemaoutis.run import _to_redo

    payload = _payload([_col("a"), _col("b", "unparseable"),
                        _col("c", "error")])
    rankings = {"a": [], "b": [], "c": [], "d": []}

    # Without retries: the API error and the column never reached, not `b`.
    assert sorted(_to_redo(payload, rankings)) == ["c", "d"]
    # With retries: `b` joins them, in the same file.
    assert sorted(_to_redo(payload, rankings, 3)) == ["b", "c", "d"]


def test_cost_prices_an_unreadable_column_only_when_retries_are_on(bench, tmp_path):
    """
    `run._to_redo` re-asks an unreadable column when retries are on, so the
    quote has to include it. Quoting zero and then sending it is how the two
    counters came to disagree.
    """
    bench.dry_run = False
    bench.run("rerank", levels=[20])
    path = next(tmp_path.glob("*__rerank.json"))
    _break_one(path, "unparseable")

    bench.max_parse_retries = 0
    assert int(bench.cost("rerank", levels=[20])["columns_todo"].sum()) == 0

    bench.max_parse_retries = 3
    assert int(bench.cost("rerank", levels=[20])["columns_todo"].sum()) == 1
