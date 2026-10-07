"""
Metrics, where two reasonable definitions differ: pairs against columns, MRR on
1:N gold, and declining as silence.
"""

from __future__ import annotations

import pytest

from evaluation import (
    blocking_report,
    classify_columns,
    gold_beyond_k,
    mrr,
    mrr_forfeited,
    precision_recall,
    recall_at_k,
    top1_accuracy,
)

ORDER = [f"t_{i}" for i in range(60)]


def ranking_with(gold_at):
    """A ranking placing `gold` at 0-based position `gold_at`."""
    names = [f"x_{i}" for i in range(60)]
    names[gold_at] = "GOLD"
    return [(n, 1.0 - i / 100) for i, n in enumerate(names)]


# Ranking quality

def test_recall_at_1_finds_a_top_ranked_gold():
    r = {"a": ranking_with(0)}
    assert recall_at_k(r, [("a", "GOLD")], k=1) == 1.0


def test_recall_at_k_widens_the_window():
    r = {"a": ranking_with(4)}
    assert recall_at_k(r, [("a", "GOLD")], k=1) == 0.0
    assert recall_at_k(r, [("a", "GOLD")], k=5) == 1.0


def test_recall_is_pair_denominated_and_a_1_to_n_gold_caps_it():
    """
    A column with two gold targets can contribute at most one of them at k=1, so
    recall@1 is structurally below 1.0 no matter how good the matcher is. That
    cap is a property of the corpus, and it is why top1_accuracy exists.
    """
    r = {"a": [("g1", 0.9), ("z", 0.5)]}
    gt = [("a", "g1"), ("a", "g2")]
    assert recall_at_k(r, gt, k=1) == 0.5
    assert top1_accuracy(r, gt) == 1.0


def test_mrr_credits_the_first_correct_target_and_stops():
    assert mrr({"a": ranking_with(0)}, [("a", "GOLD")]) == 1.0
    assert mrr({"a": ranking_with(3)}, [("a", "GOLD")]) == pytest.approx(0.25)


def test_mrr_counts_an_unanswered_gold_column_as_zero():
    """
    The strict denominator. Dividing by the columns that got a prediction would
    let an arm raise its MRR by declining to answer.
    """
    r = {"a": ranking_with(0), "b": []}
    assert mrr(r, [("a", "GOLD"), ("b", "GOLD")]) == pytest.approx(0.5)


# Precision and recall, with abstention taken seriously

def test_the_five_outcomes_partition_every_column():
    r = {"a": ranking_with(0), "b": ranking_with(9), "c": [("x", 0.4)],
         "d": [("y", 0.3)]}
    gt = [("a", "GOLD"), ("b", "GOLD")]
    committed = {"a": True, "b": True, "c": True, "d": False}
    cls = classify_columns(r, gt, committed)
    assert cls == {"a": "TP", "b": "FP_FN_wrong",
                   "c": "FP_invented", "d": "TN_declined"}


def test_declining_a_column_that_had_a_match_is_a_false_negative():
    r = {"a": ranking_with(0)}
    cls = classify_columns(r, [("a", "GOLD")], {"a": False})
    assert cls == {"a": "FN_abstained"}


def test_abstention_only_helps_precision_when_it_counts_as_silence():
    """
    Declining raises precision only when a declined column leaves the
    denominator.
    """
    r = {"a": ranking_with(0), "no_match": [("x", 0.4)]}
    gt = [("a", "GOLD")]

    guessing = precision_recall(r, gt, {"a": True, "no_match": True})
    refusing = precision_recall(r, gt, {"a": True, "no_match": False})

    assert guessing["precision_committed"] == pytest.approx(0.5)
    assert refusing["precision_committed"] == 1.0
    # Recall cannot tell them apart.
    assert guessing["recall_committed"] == refusing["recall_committed"] == 1.0
    # And under the fallback-filled scoring neither can precision.
    assert (guessing["precision_evaluated"]
            == refusing["precision_evaluated"] == pytest.approx(0.5))


# The run as a blocking scheme

def column(name, n_blocks, accepted, available=3):
    return {"source_column": name, "n_blocks_available": available,
            "accepted_block": accepted,
            "blocks": [{"index": i} for i in range(n_blocks)]}


def test_accepting_the_block_holding_the_gold_is_a_true_positive():
    """
    Gold at rank 30 is in block 1. Declining the empty block 0 is not an error:
    a declined block is a false negative only if it held gold.
    """
    rep = blocking_report([column("a", 2, 1)], {"a": ORDER},
                          [("a", "t_30")], window=20)
    assert (rep.tp_gold, rep.fp_gold, rep.fn_gold) == (1, 0, 0)
    assert rep.pc_accepted == 1.0


def test_a_declined_block_that_held_the_gold_is_a_false_negative():
    """
    Accepting block k returns only block k's targets, so gold in a declined
    block leaves the answer entirely rather than slipping a few ranks.
    """
    rep = blocking_report([column("a", 3, 2)], {"a": ORDER},
                          [("a", "t_5")], window=20)
    assert rep.fn_gold == 1 and rep.tp_gold == 0
    assert rep.pc_traversed == 1.0      # it was shown
    assert rep.pc_accepted == 0.0       # and then thrown away


def test_pair_completeness_separates_what_was_seen_from_what_was_kept():
    """Averaging the two would hide the arm's characteristic failure."""
    rep = blocking_report([column("a", 3, 0)], {"a": ORDER},
                          [("a", "t_0"), ("a", "t_45")], window=20)
    assert rep.pc_traversed == 1.0
    assert rep.pc_accepted == 0.5


def test_reduction_ratio_is_measured_against_brute_force():
    rep = blocking_report([column("a", 1, 0)], {"a": ORDER},
                          [("a", "t_0")], window=20,
                          n_source_columns=1, n_target_columns=60)
    assert rep.n_comparisons == 20
    assert rep.n_comparisons_brute == 60
    assert rep.rr == pytest.approx(1 - 20 / 60)


def test_the_scanning_oracle_still_pays_to_prove_a_negative():
    """
    A column with no gold cannot be skipped by any policy that does not already
    know the answer, so the honest ceiling charges for walking it to the end.
    """
    cols = [column("a", 1, 0), column("nomatch", 3, None)]
    rep = blocking_report(cols, {"a": ORDER, "nomatch": ORDER},
                          [("a", "t_0")], window=20)
    # 20 to reach the gold in "a", then the whole 60-column order for "nomatch".
    assert rep.n_comparisons_oracle_scan == 20 + 60


def test_gold_only_precision_ignores_unanswerable_columns():
    """
    Precision pooled over every column is dominated by the ones where accepting
    anything is wrong by construction; that marks the model on a question with
    no right answer. Only the gold-bearing columns are scored, for that reason.
    """
    cols = [column("a", 1, 0), column("nomatch", 1, 0)]
    rep = blocking_report(cols, {"a": ORDER, "nomatch": ORDER},
                          [("a", "t_0")], window=20)
    assert rep.block_precision_gold == 1.0


# What a source prune gives up, and the tail no shortlist reaches

def test_the_forfeited_mrr_is_the_drop_the_prune_actually_produces():
    """Removing a column lowers MRR by exactly its reciprocal rank."""
    rankings = {"a": [("x", 0.9), ("y", 0.2)],
                "b": [("y", 0.8), ("x", 0.3)],
                "c": [("z", 0.7), ("x", 0.1)]}
    gt = [("a", "x"), ("b", "x"), ("c", "x")]
    before = mrr(rankings, gt)

    pruned = {**rankings, "b": []}
    after = mrr(pruned, gt)

    forfeit = mrr_forfeited(rankings, gt, ["b"])
    assert forfeit["n_gold_columns"] == 3
    assert forfeit["n_gold_columns_removed"] == 1
    assert forfeit["mrr_forfeited"] == pytest.approx(before - after)


def test_removing_a_column_with_no_gold_forfeits_nothing():
    rankings = {"a": [("x", 0.9)], "b": [("x", 0.1)]}
    gt = [("a", "x")]
    assert mrr_forfeited(rankings, gt, ["b"])["mrr_forfeited"] == 0.0


def test_the_tail_is_counted_per_pair_not_per_column():
    """
    A column whose first gold target is shallow can carry a second that is
    not, and only the second is beyond a shortlist's reach.
    """
    rankings = {"a": [(f"t{i}", 1.0 - i / 100) for i in range(30)]}
    gt = [("a", "t0"), ("a", "t25")]
    beyond = gold_beyond_k(rankings, gt, k=20)
    assert beyond == [("a", "t25", 26)]


def test_a_gold_target_missing_from_the_order_is_beyond_every_depth():
    """Absent, not rank-worst: the two need different explanations."""
    rankings = {"a": [("x", 0.9)]}
    assert gold_beyond_k(rankings, [("a", "gone")], k=20) == [("a", "gone", None)]
