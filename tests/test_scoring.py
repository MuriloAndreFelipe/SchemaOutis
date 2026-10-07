"""
Scoring runs into the paper's numbers: pooling studies, the ceiling row, and
model names.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from conftest import ScriptedClient, score_first
from evaluation import scoring as report
from schemaoutis import cache, inventory
from schemaoutis.run import run_pair

CORPUS, RETRIEVER, ARM = "gdc", "qwen", "anthropic__claude-sonnet-5"


@pytest.fixture
def rankings(ranking, source):
    return {col: ranking for col in source.columns}


@pytest.fixture
def two_studies(targets, source):
    """Two pairs sharing a column name, which is the case that pools wrongly."""
    target_df = pd.DataFrame({t: [1] for t in targets})
    return [
        {"name": "Cao", "source_df": source, "target_df": target_df,
         "ground_truth": [("age", "t_0")]},
        {"name": "Wang", "source_df": source, "target_df": target_df,
         "ground_truth": [("age", "t_1")]},
    ]


def _run(pairs, rankings, target_text, tmp_path, arm=ARM, rep=1):
    for p in pairs:
        run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
                 retriever=RETRIEVER, arm=arm, rep=rep, rankings=rankings,
                 targets=target_text, client=ScriptedClient(score_first(0.9)),
                 model="fake", dry_run=False)
    return inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",))


def test_two_studies_sharing_a_column_name_do_not_collide(
        two_studies, rankings, target_text, tmp_path):
    """
    `age` exists in both, with different gold. Pooling on the bare name keeps
    one of them and scores half the corpus against the wrong target.
    """
    inv = _run(two_studies, rankings, target_text, tmp_path)
    pooled, gold, committed, abstained = report.pool(inv.itertuples(),
                                                     two_studies)

    assert len(pooled) == 4          # 2 columns x 2 studies, none overwritten
    assert len(gold) == 2
    assert len({s for s, _ in gold}) == 2


def test_a_study_the_run_skipped_is_not_credited(
        two_studies, rankings, target_text, tmp_path):
    """Scoring what is there is right. Counting what is not is not."""
    inv = _run(two_studies[:1], rankings, target_text, tmp_path)
    scored = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                          ceiling=False)
    assert scored.loc[0, "n_studies"] == 1
    assert scored.loc[0, "n_gold_pairs"] == 1


def test_one_row_per_configuration_and_repeat(
        two_studies, rankings, target_text, tmp_path):
    for rep in (1, 2, 3):
        _run(two_studies, rankings, target_text, tmp_path, rep=rep)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",))
    scored = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                          ceiling=False)
    assert sorted(scored["rep"]) == [1, 2, 3]
    assert scored["model"].nunique() == 1


def test_the_spread_needs_more_than_one_repeat(
        two_studies, rankings, target_text, tmp_path):
    for rep in (1, 2):
        _run(two_studies, rankings, target_text, tmp_path, rep=rep)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",))
    spread = report.across_repeats(
        report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                     ceiling=False))
    assert spread.loc[0, "reps"] == 2
    assert "mrr_std" in spread.columns


# The table

def test_the_variant_survives_into_the_model_name():
    """
    Two prompts, two measurements. A table printing both as `claude-sonnet-5`
    puts them under one name and the reader cannot tell which is which.
    """
    plain = report.short_model("anthropic__claude-sonnet-5")
    variant = report.short_model("anthropic__claude-sonnet-5__tgt-match-retriever")
    assert plain == "claude-sonnet-5"
    assert plain != variant
    assert "tgt-match-retriever" in variant


def test_the_provider_is_dropped_from_the_model_name():
    """The transport is not a property of the matcher."""
    assert report.short_model("ollama__qwen3.5:9b") == "qwen3.5:9b"


def accepts_only(column, target):
    """Accept the block holding `target`, but only for `column`."""
    def reply(prompt):
        if f"Column: {column}," in prompt and f"{target}," in prompt:
            return f"{target}(0.90)"
        return "NONE"
    return reply


@pytest.fixture
def paged(two_studies, rankings, target_text, tmp_path):
    """`age` accepts its first block, `nomatch` declines every one."""
    for p in two_studies:
        run_pair(p, kind="paged", results=tmp_path, corpus=CORPUS,
                 retriever=RETRIEVER, arm=ARM, rankings=rankings,
                 targets=target_text, window=20, threshold=0.0,
                 client=ScriptedClient(accepts_only("age", "t_0")),
                 model="fake", dry_run=False)
    return inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))


def test_paged_is_keyed_on_its_own_settings_not_the_shortlist(
        paged, two_studies, tmp_path):
    """
    top_k means nothing to a walk. Grouping paged rows by it would collapse two
    thresholds into one row and average measurements of different arms.
    """
    scored = report.score(paged, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="paged", ceiling=False)
    assert "threshold" in scored.columns and "window" in scored.columns
    assert scored.loc[0, "window"] == 20


def test_the_two_precisions_separate_once_an_arm_can_decline(
        paged, two_studies, tmp_path):
    """
    On rerank they are equal by construction. Here `nomatch` is declined, which
    keeps it out of the committed denominator and not the evaluated one, and
    that gap is the paper's argument.
    """
    scored = report.score(paged, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="paged", ceiling=False)
    r = scored.iloc[0]
    assert r["n_declined"] > 0
    assert r["precision_committed"] > r["precision_evaluated"]


def test_declining_is_scored_as_its_own_decision(paged, two_studies, tmp_path):
    """
    No ranking metric registers it: an arm that declines correctly and one that
    guesses get the same recall@1.
    """
    scored = report.score(paged, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="paged", ceiling=False)
    r = scored.iloc[0]
    # `nomatch` has no gold in either study and was declined in both.
    assert r["n_without_gold"] == 2
    assert r["abstain_precision"] == 1.0
    assert r["abstain_recall"] == 1.0


def test_abstention_is_undefined_rather_than_zero_when_nothing_declined():
    """A precision over an empty denominator is not 0, and printing 0 would
    read as an arm that declined badly rather than one that never declined."""
    out = report.abstention({"tn_declined": 0, "fn_abstained": 0,
                             "fp_invented": 7})
    assert out["n_declined"] == 0
    assert pd.isna(out["abstain_precision"])
    assert out["abstain_recall"] == 0.0


def test_the_cost_of_a_column_is_a_distribution_not_a_mean(
        paged, two_studies, tmp_path):
    """
    A column that accepts its first block costs 1 and one that declines every
    block costs the whole schema. The mean sits between two numbers almost no
    column pays, so the median has to be reported beside it.
    """
    cols = report.per_column(paged.itertuples(), two_studies)
    assert len(cols) == 4                       # 2 columns x 2 studies
    assert set(cols["stop_reason"]) == {"accepted", "blocks_exhausted"}

    accepted = cols[cols["stop_reason"] == "accepted"]["api_calls"]
    declined = cols[cols["stop_reason"] == "blocks_exhausted"]["api_calls"]
    assert accepted.max() < declined.min()

    shape = report.cost_shape(cols)
    assert shape["calls_with_gold"] < shape["calls_without_gold"]
    assert shape["calls_max"] == declined.max()


def _write_order(tmp_path, pairs, ranking, retriever=RETRIEVER):
    """The retriever's order on disk, which is what blocks are cut from."""
    for p in pairs:
        cache.write(
            cache.retrieval_path(tmp_path, CORPUS, p["name"], retriever),
            {"corpus": CORPUS, "pair": p["name"], "retriever": retriever,
             "rankings": {c: [[t, float(s)] for t, s in ranking]
                          for c in p["source_df"].columns}})


def test_pair_completeness_is_reported_twice(
        two_studies, rankings, ranking, target_text, tmp_path):
    """
    Gold that was shown and then declined is not gold that was never reached.
    One pooled number would average an arm that keeps what it sees with one
    that throws it away.
    """
    _write_order(tmp_path, two_studies, ranking)
    for p in two_studies:
        run_pair(p, kind="paged", results=tmp_path, corpus=CORPUS,
                 retriever=RETRIEVER, arm=ARM, rankings=rankings,
                 targets=target_text, window=20, threshold=0.0,
                 client=ScriptedClient(lambda prompt: "NONE"),
                 model="fake", dry_run=False)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))
    scored = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="paged", ceiling=False)

    r = scored.iloc[0]
    # Declined everything, so the walk saw all the gold and kept none of it.
    assert r["pc_traversed"] == 1.0
    assert r["pc_accepted"] == 0.0


def test_the_walk_that_keeps_what_it_sees_shows_no_gap(
        paged, two_studies, ranking, tmp_path):
    _write_order(tmp_path, two_studies, ranking)
    scored = report.score(paged, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="paged", ceiling=False)
    r = scored.iloc[0]
    assert r["pc_traversed"] == r["pc_accepted"]


def test_reduction_ratio_comes_with_its_oracle_baseline(
        paged, two_studies, ranking, tmp_path):
    """
    Against brute force any blocking looks excellent. The stop-at-gold oracle
    is the ceiling for a policy that cannot see the answer in advance, and it
    is what `rr` has to be read against.
    """
    _write_order(tmp_path, two_studies, ranking)
    scored = report.score(paged, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="paged", ceiling=False)
    r = scored.iloc[0]
    assert {"rr", "rr_stop_at_gold"} <= set(scored.columns)
    assert r["n_comparisons"] <= r["n_comparisons_brute"]


def test_the_rerank_arm_is_not_scored_as_blocking(
        two_studies, rankings, target_text, tmp_path):
    """One block is not a walk, and RR over a fixed shortlist says nothing."""
    _run(two_studies, rankings, target_text, tmp_path)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",))
    scored = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="rerank", ceiling=False)
    assert "pc_traversed" not in scored.columns


# The findings, as prose

def test_grouping_splits_the_score_without_averaging_the_pairs(
        two_studies, rankings, target_text, tmp_path):
    """
    Valentine has 546 pairs. A row per pair is not a table and one row pooling
    five unrelated domains describes none of them, so a group is scored as its
    own pool rather than as a mean of its members.
    """
    inv = _run(two_studies, rankings, target_text, tmp_path)
    inv["group"] = inv["pair"]                    # one group per study here

    combined = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                            kind="rerank", ceiling=False)
    grouped = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                           kind="rerank", ceiling=False, group_by="group")

    assert len(combined) == 1
    assert len(grouped) == 2
    assert set(grouped["group"]) == {"Cao", "Wang"}
    # The pooled row covers every column; each group covers only its own.
    assert combined["n_columns"].iloc[0] == grouped["n_columns"].sum()
    assert combined["n_gold_pairs"].iloc[0] == grouped["n_gold_pairs"].sum()


def test_grouping_refuses_a_column_the_inventory_does_not_have(
        two_studies, rankings, target_text, tmp_path):
    """Silently pooling everything would look like a result over one group."""
    inv = _run(two_studies, rankings, target_text, tmp_path)
    with pytest.raises(KeyError, match="subset"):
        report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                     kind="rerank", ceiling=False, group_by="subset")


# A prune is not a decline

def test_a_pruned_column_is_counted_apart_from_a_declined_one(
        two_studies, rankings, target_text, tmp_path):
    """Pruned and model-declined columns are counted separately."""
    for p in two_studies:
        run_pair(p, kind="gated", results=tmp_path, corpus=CORPUS,
                 retriever=RETRIEVER, arm=ARM, rankings=rankings,
                 targets=target_text, prune=None, model="fake",
                 client=ScriptedClient(lambda prompt: "NONE"), dry_run=False)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("gated",))
    scored = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="gated").iloc[0]
    assert scored["n_model_declined"] > 0
    assert scored["n_pruned"] == 0
    # `abstention`'s own n_declined is every abstention of any kind, so the two
    # have to partition it exactly. Counting the literal stop reason "declined"
    # instead broke this: the paged arm says no as `blocks_exhausted` and was
    # scored as never declining while it declined 71 times.
    assert (scored["n_pruned"] + scored["n_model_declined"]
            == scored["n_declined"])


def test_a_filtered_column_is_charged_to_the_filter_not_the_model(
        two_studies, rankings, target_text, tmp_path):
    """The partition has to hold with a prune in play, which is its whole use."""
    for p in two_studies:
        run_pair(p, kind="gated", results=tmp_path, corpus=CORPUS,
                 retriever=RETRIEVER, arm=ARM, rankings=rankings,
                 targets=target_text, prune="mutual", model="fake",
                 client=ScriptedClient(lambda prompt: "NONE"), dry_run=False)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("gated",))
    scored = report.score(inv, two_studies, results=tmp_path, corpus=CORPUS,
                          kind="gated").iloc[0]
    assert scored["n_pruned"] > 0
    assert (scored["n_pruned"] + scored["n_model_declined"]
            == scored["n_declined"])


# Pair names that a filename has to sanitise

@pytest.fixture
def nested_studies(targets, source):
    """
    Valentine pair names are paths, which filenames store with the separators
    replaced.
    """
    target_df = pd.DataFrame({t: [1] for t in targets})
    return [
        {"name": "OpenData/Joinable/miller2_ac1", "source_df": source,
         "target_df": target_df, "ground_truth": [("age", "t_0")]},
        {"name": "OpenData/Joinable/miller2_ac2", "source_df": source,
         "target_df": target_df, "ground_truth": [("age", "t_1")]},
    ]


def test_a_nested_pair_name_is_still_found_when_scoring(
        nested_studies, rankings, target_text, tmp_path):
    """Scoring finds a pair whose name is a path."""
    inv = _run(nested_studies, rankings, target_text, tmp_path)
    assert len(inv) == 2
    # The inventory spells the pair with separators replaced.
    assert all("/" not in p for p in inv["pair"])

    scored = report.score(inv, nested_studies, results=tmp_path,
                          corpus=CORPUS, kind="rerank")
    assert len(scored) == 1
    assert scored.iloc[0]["n_studies"] == 2
    assert scored.iloc[0]["n_gold_pairs"] == 2


def test_the_pair_index_is_unchanged_for_a_flat_corpus(two_studies):
    """`fs_safe` is a no-op on GDC's names, so nothing moves there."""
    assert set(report.by_pair_name(two_studies)) == {"Cao", "Wang"}


def test_a_declined_column_scores_zero_even_if_it_carries_a_ranking():
    rankings = {"a": [("GOLD", 0.9)], "b": [("GOLD_B", 0.9)]}
    gt = [("a", "GOLD"), ("b", "GOLD_B")]
    row = report.metrics(rankings, gt, {"a": True, "b": False}, ks=(1,))
    assert row["mrr"] == pytest.approx(0.5)
    assert row["recall@1"] == pytest.approx(0.5)


def test_evaluated_precision_fills_a_declined_column_from_the_retriever():
    rankings = {"a": [("GOLD", 0.9)], "b": []}
    gt = [("a", "GOLD"), ("b", "GOLD_B")]
    fallback = {"b": [("GOLD_B", 0.5)]}
    committed = {"a": True, "b": False}
    bare = report.metrics(rankings, gt, committed, ks=(1,))
    filled = report.metrics(rankings, gt, committed, ks=(1,), fallback=fallback)
    assert bare["precision_evaluated"] == pytest.approx(0.5)
    assert filled["precision_evaluated"] == pytest.approx(1.0)
    assert filled["precision_committed"] == bare["precision_committed"] == 1.0
    assert filled["mrr"] == bare["mrr"] == pytest.approx(0.5)


def test_the_retrieval_order_is_keyed_as_pool_keys_a_slashed_pair(tmp_path):
    """Valentine names hold slashes; the filename, and `pool`, do not."""
    pair = {"name": "OpenData/Joinable/x", "ground_truth": [("a", "t")]}
    path = cache.retrieval_path(tmp_path, "valentine", pair["name"], RETRIEVER)
    path.write_text(json.dumps({"rankings": {"a": [["t", 0.9]]}}),
                    encoding="utf-8")
    safe = cache.fs_safe(pair["name"])
    rankings, gt = report.retrieval_rankings(
        [pair], results=tmp_path, corpus="valentine", retriever=RETRIEVER,
        covered=[safe])
    assert list(rankings) == [report.namespaced(safe, "a")]
    assert gt == [(report.namespaced(safe, "a"), "t")]


def test_mrr_abstain_credits_a_decline_only_where_there_is_no_gold():
    from evaluation.matching import mrr_abstain

    rankings = {"g1": [("x", 0.9), ("G", 0.8)], "g2": [("G2", 0.9)],
                "free_declined": [], "free_answered": [("y", 0.9)]}
    gt = [("g1", "G"), ("g2", "G2")]
    committed = {"g1": True, "g2": False, "free_declined": False,
                 "free_answered": True}
    # g1 at rank 2, g2 declined, one correct decline, one invented answer.
    assert mrr_abstain(rankings, gt, committed) == pytest.approx(
        (0.5 + 0 + 1 + 0) / 4)


def test_mrr_abstain_is_in_every_scored_row():
    rankings = {"a": [("GOLD", 0.9)], "b": []}
    row = report.metrics(rankings, [("a", "GOLD")], {"a": True, "b": False},
                         ks=(1,))
    assert row["mrr_abstain"] == pytest.approx(1.0)


def test_an_unreadable_reply_is_not_an_abstention():
    """
    A column whose reply could not be read asserted nothing, but it did not
    decline either. Scoring it as a decline pays a model for being unreadable,
    and `abstain_precision` is the number that carries the paper's argument.
    """
    rankings = {"free_declined": [], "free_unreadable": []}
    committed = {"free_declined": False, "free_unreadable": False}
    abstained = {"free_declined": True, "free_unreadable": False}
    gt = [("elsewhere", "G")]

    row = report.metrics(rankings, gt, committed, ks=(1,), abstained=abstained)
    assert row["tn_declined"] == 1 and row["no_answer_free"] == 1
    # One of the two gold-free columns was declined, and the unreadable one is
    # in the recall's denominator but not its numerator or the precision's.
    assert row["abstain_precision"] == pytest.approx(1.0)
    assert row["abstain_recall"] == pytest.approx(0.5)
    assert row["mrr_abstain"] == pytest.approx(0.5)

    # Without the flag the two are indistinguishable, which is the old bug.
    bare = report.metrics(rankings, gt, committed, ks=(1,))
    assert bare["abstain_recall"] == pytest.approx(1.0)
    assert bare["mrr_abstain"] == pytest.approx(1.0)


def test_reduction_ratio_counts_only_pairs_that_could_exist():
    """
    Two studies with different target widths. Against the widest one the
    denominator counts comparisons no matcher could ever make, which reads
    reduction ratio high by 25% to 44% on every Valentine corpus.
    """
    from evaluation.blocking import blocking_report

    columns = [{"source_column": "a", "blocks": [{"start": 0, "end": 10}],
                "accepted_block": 0, "stop_reason": "accepted"}]
    rep = blocking_report(columns, {"a": [f"t{i}" for i in range(10)]}, [],
                          window=10, n_source_columns=3, n_target_columns=100,
                          n_pairs_brute=3 * 40)
    assert rep.n_comparisons_brute == 120
    # Falling back to the product is what a caller that knows no better gets.
    wide = blocking_report(columns, {"a": [f"t{i}" for i in range(10)]}, [],
                           window=10, n_source_columns=3, n_target_columns=100)
    assert wide.n_comparisons_brute == 300
