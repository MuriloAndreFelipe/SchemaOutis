"""
Source and target pruning. The rules never read labels.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from schemaoracle import pruning
from schemaoracle.pruning import (csls, masks as shipped_masks,
                                  score_matrix)
from evaluation.matching import (recall_at_ground_truth,
                                 recall_at_ground_truth_bounds)


def order(**cols):
    """{source: [(target, score), ...]} from {source: {target: score}}."""
    return {s: sorted(v.items(), key=lambda kv: -kv[1]) for s, v in cols.items()}


def planted(n_sources=12, n_targets=40, n_gold=8, seed=0):
    """A matrix where the first `n_gold` sources have a planted match."""
    rng = np.random.default_rng(seed)
    M = rng.random((n_sources, n_targets)) * 0.3 + 0.4
    gold = []
    for i in range(n_gold):
        M[i, i] = 0.95 - 0.01 * i
        gold.append((f"s{i}", f"t{i}"))
    rankings = {f"s{i}": [(f"t{j}", float(M[i, j]))
                          for j in np.argsort(-M[i])]
                for i in range(n_sources)}
    return rankings, gold


# The matrix and the correction

def test_the_matrix_keeps_the_first_rows_target_order():
    M, sources, targets = score_matrix(
        order(a={"t1": 0.9, "t2": 0.4}, b={"t1": 0.1, "t2": 0.8}))
    assert sources == ["a", "b"] and targets == ["t1", "t2"]
    assert M[1].tolist() == pytest.approx([0.1, 0.8])


def test_a_partial_ranking_raises_rather_than_leaving_a_hole():
    """A short row would corrupt the transpose, and silently."""
    bad = {"a": [("t1", 0.9), ("t2", 0.4)], "b": [("t1", 0.1)]}
    with pytest.raises(ValueError):
        score_matrix(bad)


def test_csls_demotes_a_hub_that_is_close_to_everyone():
    """
    A hub at 0.80 to all six sources beats each one's own 0.78 target on raw
    cosine, and CSLS demotes it. A hub needs several sources to show.
    """
    rankings = {f"s{i}": [("hub", 0.80)] +
                         [(f"own{j}", 0.78 if i == j else 0.05)
                          for j in range(6)]
                for i in range(6)}
    M, sources, targets = score_matrix(rankings)
    C = csls(M, k=3)
    hub = targets.index("hub")
    for i in range(6):
        own = targets.index(f"own{i}")
        assert M[i, hub] > M[i, own]            # raw cosine: the hub wins
        assert C[i, own] > C[i, hub]            # csls: the specific target does


def test_normalise_reads_back_its_own_tag_spelling():
    """
    `tag` joins with a hyphen, so anything rebuilding a configuration from a
    filename hands that string straight back. Without the round trip
    `normalise(tag(x))` raised on the one rule set with two rules in it.
    """
    for n in range(len(pruning.POLICIES) + 1):
        for rules in combinations(pruning.POLICIES, n):
            assert pruning.normalise(pruning.tag(rules)) == rules
    assert pruning.tag("mutual-zero") == "mutual-zero"
    assert pruning.tag("zerosrc-mutual") == "mutual-zerosrc"
    with pytest.raises(ValueError, match="unknown prune rule"):
        pruning.normalise("mutual-nonsense")


def test_recall_at_gt_bounds_bracket_the_reported_number():
    """
    The budget is global, so the cut can fall inside a run of equal scores and
    the reported number then depends on sort order. The bounds say how much.
    """
    for seed in range(4):
        rankings, gold = planted(seed=seed)
        lo, hi = recall_at_ground_truth_bounds(rankings, gold)
        assert lo <= recall_at_ground_truth(rankings, gold) <= hi


def test_a_coarse_score_makes_the_tie_break_decide_recall_at_gt():
    """
    Every pair scores the same, so the budget of two is filled arbitrarily and
    the interval is the whole range. A single number here would be a coin toss
    reported to three decimals.
    """
    rankings = {"a": [("x", 0.5), ("y", 0.5)], "b": [("x", 0.5), ("y", 0.5)]}
    gold = [("a", "x"), ("b", "y")]
    assert recall_at_ground_truth_bounds(rankings, gold) == (0.0, 1.0)


def contested_matrix():
    """
    Six sources with their own target, one contesting a taken target, and two
    weak targets whose CSLS is negative.
    """
    targets = [f"own{j}" for j in range(6)] + ["nobody0", "nobody1"]
    rankings = {}
    for i in range(6):
        row = {t: 0.05 for t in targets}
        row[f"own{i}"] = 0.78
        rankings[f"s{i}"] = row
    contested = {t: 0.05 for t in targets}
    contested["own0"] = 0.60
    rankings["contested"] = contested
    return {s: sorted(v.items(), key=lambda kv: -kv[1])
            for s, v in rankings.items()}


def stranded_matrix():
    """
    `contested_matrix` plus a source column that is slightly further from
    every target than any other source is, so nothing clears its own
    neighbourhood and `zerosrc` is the only rule that sees it. 0.04 against
    0.05 rather than an equal score: at equality the margin lands on floating
    point either side of the `<= 0` cut.
    """
    rankings = {s: dict(v) for s, v in contested_matrix().items()}
    targets = list(next(iter(rankings.values())))
    rankings["stranger"] = {t: 0.04 for t in targets}
    return {s: sorted(v.items(), key=lambda kv: -kv[1])
            for s, v in rankings.items()}


def test_mutual_drops_only_the_column_whose_preference_is_not_returned():
    """
    The exact mask, not its size. These two rules name the paper's numbers, so
    a silent drift in either invalidates every replay.
    """
    M, sources, _ = score_matrix(contested_matrix())
    drop_s, _ = shipped_masks(M, ("mutual",))
    assert {s for s, d in zip(sources, drop_s) if d} == {"contested"}


def test_zero_drops_only_the_targets_no_source_is_close_to():
    M, _, targets = score_matrix(contested_matrix())
    _, drop_t = shipped_masks(M, ("zero",))
    assert ({t for t, d in zip(targets, drop_t) if d}
            == {"nobody0", "nobody1"})


def test_composing_the_rules_uses_one_correction_for_both_sides():
    """
    Two corrections would measure the two rules under different hubness
    estimates, and the composition would not be the two rules it claims.
    """
    M, _, _ = score_matrix(contested_matrix())
    both_s, both_t = shipped_masks(M, ("mutual", "zero"))
    only_s, _ = shipped_masks(M, ("mutual",))
    _, only_t = shipped_masks(M, ("zero",))
    assert (both_s == only_s).all() and (both_t == only_t).all()


def test_a_prune_request_has_one_canonical_spelling():
    """The tuple names the run file, so order must not make two of one."""
    assert pruning.normalise(("zero", "mutual")) == ("mutual", "zero")
    assert pruning.tag(("zero", "mutual")) == "mutual-zero"
    assert pruning.tag(None) == "" and pruning.tag([]) == ""


# `zerosrc`: the same test as `zero`, read on the other margin

def test_zerosrc_drops_only_the_column_with_nothing_above_its_neighbourhood():
    M, sources, _ = score_matrix(stranded_matrix())
    drop_s, drop_t = shipped_masks(M, ("zerosrc",))
    assert {s for s, d in zip(sources, drop_s) if d} == {"stranger"}
    assert not drop_t.any()


def test_the_two_source_rules_see_different_columns():
    """
    They are not two spellings of one rule. `mutual` asks whether the best
    target returns the preference, `zerosrc` whether there is a best target at
    all, and on this matrix each fires where the other does not.
    """
    M, sources, _ = score_matrix(stranded_matrix())
    mut, _ = shipped_masks(M, ("mutual",))
    zsrc, _ = shipped_masks(M, ("zerosrc",))
    named = lambda m: {s for s, d in zip(sources, m) if d}
    assert named(mut) == {"contested"} and named(zsrc) == {"stranger"}


def test_the_three_rules_decompose_into_their_parts():
    """
    One correction of the unfiltered matrix feeds every mask, so stacking the
    rules is the union of running them alone and cannot depend on order.
    """
    M, _, _ = score_matrix(stranded_matrix())
    all_s, all_t = shipped_masks(M, pruning.POLICIES)
    want_s = np.zeros_like(all_s)
    want_t = np.zeros_like(all_t)
    for rule in pruning.POLICIES:
        s, t = shipped_masks(M, (rule,))
        want_s |= s
        want_t |= t
    assert (all_s == want_s).all() and (all_t == want_t).all()


def test_zerosrc_leaves_the_prompts_alone():
    """
    It removes a column, never a candidate, so a run with it sends the same
    prompt as an unfiltered run for every column it still asks. That is what
    `cuts_targets` is read for, and it is the difference from `zero`.
    """
    rankings = stranded_matrix()
    plan = pruning.plan(rankings, "zerosrc")
    assert plan.active and not plan.cuts_targets
    assert plan.skip() == ("stranger",) and plan.targets == ()
    kept = plan.apply(rankings)
    assert kept == {k: list(v) for k, v in rankings.items()}
    assert pruning.plan(rankings, "zero").cuts_targets


def test_an_inert_plan_leaves_the_order_untouched():
    rankings = contested_matrix()
    plan = pruning.plan(rankings, None)
    assert not plan.active and plan.skip() == ()
    assert plan.apply(rankings) == {k: list(v) for k, v in rankings.items()}


def test_a_column_left_with_no_candidates_is_pruned_not_exhausted():
    """
    A column left with no candidates is recorded as pruned, not as a model that
    declined everything.
    """
    # One target, and it is a hub nothing is specifically close to, so `zero`
    # takes the entire schema.
    rankings = {f"s{i}": [("only", 0.5)] for i in range(4)}
    plan = pruning.plan(rankings, "zero")
    assert set(plan.targets) == {"only"}
    assert set(plan.emptied) == set(rankings) - set(plan.sources)
    assert set(plan.skip()) == set(rankings)


# The paper's predictor, depth and bootstrap helpers
