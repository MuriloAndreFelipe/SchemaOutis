"""
The retrieval ablation's bookkeeping. `build_config` is stubbed, so nothing is
embedded.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation import ablation


@pytest.fixture
def cfg():
    return {"embedding_model": "m", "encoding_mode": "header_values_verbose",
            "sampling_mode": "frequent", "sampling_size": 10, "seed": 0}


def fake_pair(name, source_cols, target_cols, gold):
    return {"name": name,
            "source_df": pd.DataFrame({c: [1, 2] for c in source_cols}),
            "target_df": pd.DataFrame({c: [1, 2] for c in target_cols}),
            "ground_truth": gold}


def fake_payload(cfg, pairs, order_by_pair):
    """A payload shaped exactly as `build_config` returns one."""
    return {
        "format": ablation.FORMAT_VERSION,
        "config": ablation.normalise_config(cfg),
        "config_id": ablation.config_id(cfg),
        "label": ablation.label(cfg),
        "pairs": {
            p["name"]: {
                "targets": list(order_by_pair[p["name"]]["targets"]),
                "rankings": {
                    col: [[i, 1.0 - i / 100] for i in idxs]
                    for col, idxs in order_by_pair[p["name"]]["rankings"].items()
                },
            }
            for p in pairs
        },
    }


# The grid

def test_header_only_collapses_its_redundant_axes():
    """
    It emits no values, so the sampler and the sample size cannot change its
    output. Keeping all 18 would embed the same text 18 times.
    """
    g = ablation.grid(["m"], ["header_only"],
                      ["frequent", "random", "mixed"], [5, 10, 20])
    assert len(g) == 1


def test_a_value_bearing_serializer_keeps_every_combination():
    g = ablation.grid(["m"], ["header_values_verbose"],
                      ["frequent", "random", "mixed"], [5, 10, 20])
    assert len(g) == 9


def test_the_grid_is_model_major_so_each_model_loads_once():
    g = ablation.grid(["a", "b"], ["header_values_verbose"], ["frequent"], [5])
    assert [c["embedding_model"] for c in g] == ["a", "b"]

    g = ablation.grid(["a", "b"], ["header_values_verbose"],
                      ["frequent", "mixed"], [5, 10])
    models = [c["embedding_model"] for c in g]
    assert models == ["a"] * 4 + ["b"] * 4


def test_the_full_default_grid_is_the_documented_size():
    """4 value-bearing serializers x 6 samplers x 3 sizes, plus header_only."""
    assert len(ablation.grid(["m"])) == 4 * 6 * 3 + 1 == 73


# Config identity

def test_a_config_rebuilt_from_a_dataframe_row_gets_the_same_id(cfg):
    """
    Section 8 of the notebook rebuilds the winning config from the summary
    frame, where pandas has made `sampling_size` a numpy int64. If that hashed
    differently the same configuration would be embedded a second time.
    """
    row = pd.DataFrame([cfg]).iloc[0]
    assert isinstance(row["sampling_size"], np.integer)
    assert ablation.config_id(row) == ablation.config_id(cfg)


def test_the_id_is_stable_across_key_order(cfg):
    assert ablation.config_id(dict(reversed(list(cfg.items())))) == \
        ablation.config_id(cfg)


def test_changing_any_knob_changes_the_id(cfg):
    ids = {ablation.config_id(cfg)}
    for field, other in [("embedding_model", "n"), ("encoding_mode", "header_only"),
                         ("sampling_mode", "mixed"), ("sampling_size", 5),
                         ("seed", 1)]:
        ids.add(ablation.config_id({**cfg, field: other}))
    assert len(ids) == 6


def test_a_normalised_config_is_json_serialisable(cfg):
    row = pd.DataFrame([cfg]).iloc[0]
    json.dumps(ablation.normalise_config(row))   # must not raise


# Deduplication

def test_identical_text_digests_the_same_however_it_was_configured():
    a = {"col": "Column: age"}
    assert ablation.profile_digest(a) == ablation.profile_digest(dict(a))


def test_the_digest_separates_the_sides():
    """`(src={'a'}, tgt={}) ` and `(src={}, tgt={'a'})` are different runs."""
    assert ablation.profile_digest({"c": "x"}, {}) != \
        ablation.profile_digest({}, {"c": "x"})


# The stored ranking

def test_a_ranking_survives_the_index_encoding_unchanged():
    targets = ["t_a", "t_b", "t_c"]
    ranking = {"age": [("t_c", 0.9), ("t_a", 0.5), ("t_b", 0.25)]}
    back = ablation._decode(ablation._encode(ranking, targets), targets)
    assert back == ranking


def test_gold_ranks_are_read_out_of_a_stored_run(cfg):
    targets = [f"t_{i}" for i in range(60)]
    pair = fake_pair("P", ["age"], targets, [("age", "t_34")])
    payload = fake_payload(cfg, [pair], {
        "P": {"targets": targets, "rankings": {"age": list(range(60))}}})

    rows = ablation.gold_rank_rows(payload, [pair], window=20, top_k=20)
    assert len(rows) == 1
    assert rows[0]["rank"] == 35            # 1-based
    assert rows[0]["block"] == 1            # ranks 20-39
    assert rows[0]["within_top_k"] is False


def test_a_gold_target_missing_from_the_schema_reads_as_none_not_as_worst(cfg):
    """
    A ground-truth problem, not a retrieval one. Scoring it as rank-last would
    quietly blame the retriever for a broken mapping.
    """
    targets = ["t_0", "t_1"]
    pair = fake_pair("P", ["age"], targets, [("age", "not_a_column")])
    payload = fake_payload(cfg, [pair], {
        "P": {"targets": targets, "rankings": {"age": [0, 1]}}})

    rows = ablation.gold_rank_rows(payload, [pair])
    assert rows[0]["rank"] is None and rows[0]["block"] is None


# Caching

def test_a_stale_layout_reads_as_absent_rather_than_raising(tmp_path, cfg):
    path = ablation.ablation_path(tmp_path, "gdc", cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": cfg, "rows": []}), encoding="utf-8")

    assert ablation.load_config(tmp_path, "gdc", cfg) is None


def test_a_current_payload_reads_back(tmp_path, cfg):
    targets = ["t_0", "t_1"]
    pair = fake_pair("P", ["age"], targets, [("age", "t_1")])
    path = ablation.ablation_path(tmp_path, "gdc", cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fake_payload(cfg, [pair], {
        "P": {"targets": targets, "rankings": {"age": [1, 0]}}})),
        encoding="utf-8")

    assert ablation.load_config(tmp_path, "gdc", cfg) is not None
    order = ablation.rankings_for(tmp_path, "gdc", cfg, "P")
    assert [t for t, _ in order["age"]] == ["t_1", "t_0"]


def test_asking_for_a_config_that_was_never_run_says_so(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="has not been run"):
        ablation.rankings_for(tmp_path, "gdc", cfg, "P")


def test_asking_for_a_pair_the_sweep_did_not_cover_says_so(tmp_path, cfg):
    targets = ["t_0"]
    pair = fake_pair("P", ["age"], targets, [("age", "t_0")])
    path = ablation.ablation_path(tmp_path, "gdc", cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fake_payload(cfg, [pair], {
        "P": {"targets": targets, "rankings": {"age": [0]}}})), encoding="utf-8")

    with pytest.raises(KeyError, match="not 'Q'"):
        ablation.rankings_for(tmp_path, "gdc", cfg, "Q")


def test_a_cache_covering_fewer_studies_is_extended_not_served_short(
        tmp_path, cfg, monkeypatch):
    """
    The cache is keyed by the knobs, not by which studies were swept. Sweeping
    two studies and then asking for three must embed the third, not hand back a
    two-study answer that looks complete.
    """
    targets = ["t_0", "t_1"]
    pairs = [fake_pair(n, ["age"], targets, [("age", "t_0")]) for n in "PQR"]

    def stub(subset, config, **kw):
        stub.built.append([p["name"] for p in subset])
        return fake_payload(config, subset, {
            p["name"]: {"targets": targets, "rankings": {"age": [0, 1]}}
            for p in subset})

    stub.built = []
    monkeypatch.setattr(ablation, "build_config", stub)

    rows, status, _ = ablation.run_config(pairs[:2], cfg, results=tmp_path,
                                          corpus="gdc")
    assert status == "embedded" and sorted(rows["pair"].unique()) == ["P", "Q"]

    rows, status, _ = ablation.run_config(pairs, cfg, results=tmp_path,
                                          corpus="gdc")
    assert status == "extended"
    assert sorted(rows["pair"].unique()) == ["P", "Q", "R"]
    assert stub.built == [["P", "Q"], ["R"]], "the third study was re-embedded alone"

    rows, status, _ = ablation.run_config(pairs, cfg, results=tmp_path,
                                          corpus="gdc")
    assert status == "cached" and len(stub.built) == 2


# The summary

def test_the_summary_reports_what_each_arm_can_reach():
    ranks = pd.DataFrame([
        {"label": "c", "config_id": "x", "rank": 1, "block": 0,
         **{f: v for f, v in zip(ablation.CONFIG_FIELDS, ("m", "e", "s", 10, 0))}},
        {"label": "c", "config_id": "x", "rank": 35, "block": 1,
         **{f: v for f, v in zip(ablation.CONFIG_FIELDS, ("m", "e", "s", 10, 0))}},
    ])
    row = ablation.summarise(ranks, ks=(1, 20), top_k=20).iloc[0]

    assert row["n_gold"] == 2
    assert row["recall@1"] == 0.5
    assert row["recall@20"] == 0.5
    assert row["beyond_top_k"] == 0.5       # the half rerank cannot reach
    assert row["mean_blocks"] == 1.5
    assert row["mrr"] == pytest.approx((1 + 1 / 35) / 2)


def test_summarising_nothing_returns_nothing_rather_than_raising():
    assert ablation.summarise(pd.DataFrame()).empty


# k at a coverage level

def test_k99_is_the_depth_the_tail_reaches_not_the_middle():
    """
    99 gold pairs inside rank 5 and one at 400: the median says 2, and a
    shortlist of 20 still throws one away. k99 is what says so.
    """
    r = pd.Series([1] * 50 + [2] * 30 + [5] * 19 + [400])
    assert ablation.k_at_recall(r, 0.99) == 5.0
    assert ablation.k_at_recall(r, 1.00) == 400.0


def test_k99_is_undefined_when_more_than_one_percent_is_unreachable():
    """
    Gold the retriever never placed can never be reached at any depth.
    Returning the deepest rank would report a tail that does not end.
    """
    r = pd.Series([1.0] * 98 + [float("nan")] * 2)
    assert ablation.k_at_recall(r, 0.99, n_gold=100) is None
    assert ablation.k_at_recall(r, 0.95, n_gold=100) == 1.0


def test_k_at_recall_on_nothing_is_none_rather_than_an_error():
    assert ablation.k_at_recall(pd.Series([], dtype="float64")) is None


def test_the_summary_carries_k99():
    ranks = pd.DataFrame([
        {"label": "c", "config_id": "x", "rank": float(r), "block": 0,
         **dict(zip(ablation.CONFIG_FIELDS, ("m", "e", "s", 10, 0)))}
        for r in ([1] * 99 + [400])
    ])
    row = ablation.summarise(ranks, ks=(1, 20), top_k=20).iloc[0]
    assert row["k99"] == 1.0
    assert row["median_rank"] == 1.0


# The LaTeX table

def summary_frame():
    rows = []
    for model, mrr, r1, r20, k in [
        ("Qwen/Qwen3-Embedding-4B", 0.634, 0.512, 0.834, 412.0),
        ("vida-nyu/magneto-schema-retriever-gdc", 0.571, 0.44, 0.79, None),
    ]:
        for j, enc in enumerate(["header_values_verbose", "header_only"]):
            rows.append({"embedding_model": model, "encoding_mode": enc,
                         "sampling_mode": "priority_sampling", "sampling_size": 10,
                         "mrr": mrr - 0.1 * j, "recall@1": r1,
                         "recall@20": r20, "k99": k})
    return pd.DataFrame(rows)


def test_only_the_best_configuration_of_each_model_is_tabulated():
    best = ablation.best_per_model(summary_frame())
    assert len(best) == 2
    assert list(best["encoding_mode"]) == ["header_values_verbose"] * 2
    assert list(best["mrr"]) == sorted(best["mrr"], reverse=True)


def test_the_best_row_says_how_many_configurations_it_beat():
    """
    A max over 73 draws sits above a max over 2. Without this the table would
    rank an unfinished sweep against finished ones and show nothing amiss.
    """
    best = ablation.best_per_model(summary_frame())
    assert set(best["n_configs"]) == {2}


def test_header_only_gets_the_same_id_whatever_axes_were_passed():
    """
    Its text ignores the sampler, so two grids that name different samplers
    first must not produce two ids for one run. That embeds it twice, and the
    dedup never fires because the ids differ.
    """
    a = ablation.grid(["m"], ["header_only"], ["random", "frequent"], [5, 10])[0]
    b = ablation.grid(["m"], ["header_only"], ["frequent", "random"], [10, 5])[0]
    assert ablation.config_id(a) == ablation.config_id(b)


def test_a_longer_profile_gets_a_smaller_batch():
    """
    A transformer batch is padded to its longest member, so cost is
    `batch x longest`. Holding the batch fixed while `n_samples` grows is what
    turns a working sweep into an out-of-memory error partway through.
    """
    from schemaoracle.retrieval import plan_batch_size

    short = plan_batch_size(["x" * 300])
    long = plan_batch_size(["x" * 2100])
    assert long < short


def test_the_planned_batch_never_exceeds_the_ceiling_or_drops_below_one():
    from schemaoracle.retrieval import plan_batch_size

    assert plan_batch_size(["x" * 10], batch_size=64) == 64
    assert plan_batch_size(["x" * 5_000_000], batch_size=64) == 1
    assert plan_batch_size([], batch_size=64) == 64


def test_the_batch_is_sized_by_the_longest_not_the_average():
    """One 2,000-character profile among short ones still sets the cost."""
    from schemaoracle.retrieval import plan_batch_size

    mixed = ["x" * 20] * 99 + ["x" * 2100]
    assert plan_batch_size(mixed) == plan_batch_size(["x" * 2100])


def test_a_short_max_sequence_length_caps_the_estimate():
    """A model that truncates at 128 tokens cannot be charged for 8,000."""
    from schemaoracle.retrieval import plan_batch_size

    assert plan_batch_size(["x" * 8000], max_seq_length=128) > \
        plan_batch_size(["x" * 8000], max_seq_length=8192)


def test_an_out_of_memory_error_halves_the_batch_and_retries(monkeypatch):
    """
    The failure lands hours into a sweep, on one corner of the grid. Halving
    and retrying turns it into a slower configuration rather than a lost run.
    """
    import numpy as np
    import torch

    from schemaoracle import retrieval

    tried = []

    class Model:
        max_seq_length = 512

        def encode(self, texts, batch_size, **kw):
            tried.append(batch_size)
            if batch_size > 8:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return np.zeros((len(texts), 4), dtype="float32")

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    out = retrieval.embed_profiles({"a": "x" * 100, "b": "y" * 100}, Model())

    assert len(tried) > 1, "it gave up without retrying"
    assert tried == sorted(tried, reverse=True), "the batch must shrink"
    assert tried[-1] <= 8 and set(out) == {"a", "b"}


def test_an_out_of_memory_error_at_batch_one_is_raised(monkeypatch):
    """Nothing left to halve: the card genuinely cannot hold the model."""
    import torch

    from schemaoracle import retrieval

    class Model:
        max_seq_length = 512

        def encode(self, texts, batch_size, **kw):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        retrieval.embed_profiles({"a": "x"}, Model())


def test_pair_rankings_reads_one_pair_out_of_a_loaded_payload(cfg):
    """
    The in-memory counterpart of `rankings_for`. Walking a corpus through the
    file-reading version re-parses a multi-megabyte payload per pair.
    """
    targets = ["t_0", "t_1", "t_2"]
    pair = fake_pair("P", ["age"], targets, [("age", "t_1")])
    payload = fake_payload(cfg, [pair], {
        "P": {"targets": targets, "rankings": {"age": [2, 0, 1]}}})

    ranked = ablation.pair_rankings(payload, "P")
    assert [t for t, _ in ranked["age"]] == ["t_2", "t_0", "t_1"]
    assert ablation.pair_rankings(payload, "Q") is None


# The summary, rebuilt from the cache alone

def _cache(tmp_path, cfgs, pair, order):
    """Write one payload per config, the way `build_config` would."""
    for c in cfgs:
        path = ablation.ablation_path(tmp_path, "gdc", c)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fake_payload(c, [pair], order)),
                        encoding="utf-8")


def test_the_cached_configs_are_recovered_from_inside_the_payloads(tmp_path):
    """
    The filename carries only a digest, so a directory of these files has to be
    self-describing or the cache is unreadable without the CSV beside it.
    """
    cfgs = ablation.grid(["m"], ["header_values_verbose"], ["frequent"],
                         [5, 10])
    targets = ["t_0", "t_1", "t_2"]
    pair = fake_pair("P", ["age"], targets, [("age", "t_1")])
    _cache(tmp_path, cfgs, pair, {"P": {"targets": targets,
                                        "rankings": {"age": [1, 0, 2]}}})

    found = ablation.cached_configs(tmp_path, "gdc")
    assert sorted(c["sampling_size"] for c in found) == [5, 10]
    assert all(set(c) == set(ablation.CONFIG_FIELDS) for c in found)


def test_a_stale_payload_is_not_reported_as_a_cached_configuration(tmp_path,
                                                                   cfg):
    path = ablation.ablation_path(tmp_path, "gdc", cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": ablation.FORMAT_VERSION - 1,
                                "config": cfg}), encoding="utf-8")
    assert ablation.cached_configs(tmp_path, "gdc") == []


def test_summarising_the_cache_matches_summarising_the_sweep(tmp_path):
    """
    The cached summary embeds nothing and agrees with `sweep` on the same
    payloads.
    """
    cfgs = ablation.grid(["m"], ["header_values_verbose"], ["frequent"],
                         [5, 10])
    targets = ["t_0", "t_1", "t_2"]
    pair = fake_pair("P", ["age"], targets, [("age", "t_1")])
    order = {"P": {"targets": targets, "rankings": {"age": [1, 0, 2]}}}
    _cache(tmp_path, cfgs, pair, order)

    got = ablation.summarise_cache([pair], results=tmp_path, corpus="gdc")
    assert len(got) == 2

    rows = []
    for c in cfgs:
        payload = ablation.load_config(tmp_path, "gdc", c)
        rows.append(pd.DataFrame(ablation.gold_rank_rows(payload, [pair]))
                    .assign(**{k: c[k] for k in ablation.CONFIG_FIELDS},
                            config_id=ablation.config_id(c),
                            label=ablation.label(c)))
    want = ablation.summarise(pd.concat(rows, ignore_index=True))

    keys = ["embedding_model", "sampling_size", "mrr", "recall@1", "recall@20"]
    assert (got.sort_values("sampling_size")[keys].reset_index(drop=True)
            .equals(want.sort_values("sampling_size")[keys]
                    .reset_index(drop=True)))
    # The gold is at rank 1 of a three-target order, so this is a real number
    # rather than an empty frame that happens to match another empty frame.
    assert got["mrr"].iloc[0] == pytest.approx(1.0)


def test_summarising_an_empty_cache_returns_an_empty_frame(tmp_path):
    (tmp_path / "ablation").mkdir()
    assert ablation.summarise_cache([], results=tmp_path,
                                    corpus="gdc").empty
