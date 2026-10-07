"""
The arms. The key case is gold at rank 34: paged reaches it, a top-20 shortlist
cannot.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import ScriptedClient, accept_at, score_first
from schemaoracle import ARMS, run_arm
from schemaoracle.matcher import (
    SchemaOracle,
    convert_to_valentine_format,
    get_top_k_matches,
)
from schemaoracle.trace import (
    STOP_ACCEPTED,
    STOP_DECLINED,
    STOP_ERROR,
    STOP_FALLBACK,
    STOP_PRUNED,
    STOP_RETRIEVAL,
)

#: The gold sits at rank 34: reachable by walking, not by a top-20 shortlist.
GOLD = "t_34"


@pytest.fixture
def source():
    return pd.DataFrame({"age": [1, 2, 3]})


def run(arm, ranking, target_text, source, reply=None, **kw):
    client = ScriptedClient(reply) if reply else None
    return run_arm(arm, source, {"age": ranking}, target_text, client=client,
                   model="fake", top_k=20, window=20, **kw).columns[0]


# The ceiling each arm inherits

def test_retrieval_returns_the_cosine_order_and_makes_no_calls(
        ranking, target_text, source):
    rec = run("retrieval", ranking, target_text, source)
    assert rec.api_calls == 0
    assert rec.stop_reason == STOP_RETRIEVAL
    assert [t for t, _ in rec.ranking] == [t for t, _ in ranking[:20]]


def test_rerank_spends_exactly_one_call_per_column(ranking, target_text, source):
    rec = run("rerank", ranking, target_text, source, reply=score_first(0.60))
    assert rec.api_calls == 1


def test_rerank_cannot_see_past_its_shortlist(ranking, target_text, source):
    """
    A model that would pick the gold is never shown it, so the miss is the
    retriever's recall@k, not a model failure.
    """
    client = ScriptedClient(accept_at(GOLD))
    rec = run_arm("rerank", source, {"age": ranking}, target_text,
                  client=client, model="fake", top_k=20).columns[0]

    sent = "".join(c["messages"][0]["content"] for c in client.calls)
    assert len(client.calls) == 1
    assert GOLD not in sent, "the gold was shown despite sitting past rank 20"
    assert "t_19" in sent and "t_20" not in sent
    assert GOLD not in [t for t, _ in rec.ranking]


def test_paged_walks_until_it_reaches_the_gold(ranking, target_text, source):
    rec = run("paged", ranking, target_text, source, reply=accept_at(GOLD))
    assert rec.stop_reason == STOP_ACCEPTED
    assert rec.accepted_block == 1            # ranks 20-39
    assert rec.api_calls == 2
    assert rec.ranking[0][0] == GOLD


def test_only_paging_recovers_a_gold_below_the_shortlist(
        ranking, target_text, source):
    """The three arms side by side, which is the whole experiment."""
    found = {
        "retrieval": run("retrieval", ranking, target_text, source),
        "rerank": run("rerank", ranking, target_text, source,
                      reply=score_first(0.60)),
        "paged": run("paged", ranking, target_text, source,
                     reply=accept_at(GOLD)),
    }
    tops = {arm: (r.ranking[0][0] if r.ranking else None)
            for arm, r in found.items()}
    assert tops["retrieval"] != GOLD
    assert tops["rerank"] != GOLD
    assert tops["paged"] == GOLD


# Committing, declining, and falling back

def test_retrieval_always_commits(ranking, target_text, source):
    """Cosine ranking cannot abstain, so there is no retrieval run that declined."""
    assert run("retrieval", ranking, target_text, source).committed


def test_rerank_falls_back_to_retrieval_when_the_reply_is_unusable(
        ranking, target_text, source):
    """
    Magneto keeps the retrieval scores after its parse retries fail, so the
    rerank arm cannot abstain either. The trace still says what happened.
    """
    rec = run("rerank", ranking, target_text, source,
              reply=lambda prompt: "I cannot help with that.")
    assert rec.committed
    assert rec.stop_reason == STOP_FALLBACK
    assert rec.stop_reason != STOP_ACCEPTED   # not mistaken for a real answer
    assert [t for t, _ in rec.ranking] == [t for t, _ in ranking[:20]]


def test_a_paged_column_that_declines_everything_does_not_commit(
        ranking, target_text, source):
    rec = run("paged", ranking, target_text, source,
              reply=lambda prompt: "NONE")
    assert not rec.committed
    assert rec.abstained and rec.answered


def test_an_errored_column_is_never_given_a_fallback(ranking, target_text, source):
    """
    A column whose request raised was never asked. Handing it the retrieval
    order would turn a failed run into one that looks answered.
    """
    def boom(prompt):
        raise RuntimeError("api is down")

    rec = run("rerank", ranking, target_text, source, reply=boom)
    assert rec.stop_reason == STOP_ERROR
    assert not rec.committed and rec.ranking == []


# The shared ranking, and the API shape

def test_every_arm_reads_the_same_ranking_object(ranking, target_text, source):
    rankings = {"age": ranking}
    before = list(rankings["age"])
    for arm in ARMS:
        client = ScriptedClient(score_first(0.6)) if arm != "retrieval" else None
        run_arm(arm, source, rankings, target_text, client=client, model="fake",
                top_k=20, window=20)
    assert rankings["age"] == before, "an arm mutated the shared ranking"


def test_an_llm_arm_without_a_client_says_so(ranking, target_text, source):
    for arm in ("rerank", "paged"):
        with pytest.raises(ValueError, match="needs a client"):
            run_arm(arm, source, {"age": ranking}, target_text, model="fake")


def test_an_unknown_arm_raises(ranking, target_text, source):
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm("magic", source, {"age": ranking}, target_text)


def test_valentine_format_is_the_shape_magneto_returns():
    matches = convert_to_valentine_format({"age": [("years", 0.9)]}, "src", "tgt")
    assert matches == {(("src", "age"), ("tgt", "years")): 0.9}


def test_top_k_matches_breaks_ties_by_name_so_runs_are_reproducible():
    tied = {"b": 0.5, "a": 0.5, "c": 0.9}
    assert get_top_k_matches(tied, 3) == [("c", 0.9), ("a", 0.5), ("b", 0.5)]


def test_schemaoracle_rejects_an_unknown_parameter():
    """A silently-ignored kwarg is a run configured differently than you think."""
    with pytest.raises(TypeError, match="unknown parameter"):
        SchemaOracle(windwo=20)


def test_schemaoracle_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        SchemaOracle(arm="magic")


# The middle arm: one block, and permission to refuse it

def test_gated_sees_the_same_block_as_rerank_and_costs_the_same(
        ranking, target_text, source):
    """
    The two arms must differ in exactly one thing. If gated saw a different
    number of candidates or made a different number of calls, the comparison
    between them would no longer isolate declining.
    """
    seen = {}
    for arm in ("rerank", "gated"):
        client = ScriptedClient(accept_at("t_0"))
        run_arm(arm, source, {"age": ranking}, target_text, client=client,
                model="fake", top_k=20, window=20)
        seen[arm] = client.calls
    assert len(seen["rerank"]) == len(seen["gated"]) == 1

    def targets_in(call):
        text = call["messages"][0]["content"]
        return sum(text.count(f"t_{i},") for i in range(len(ranking)))

    assert targets_in(seen["rerank"][0]) == targets_in(seen["gated"][0])


def test_gated_declines_instead_of_falling_back(ranking, target_text, source):
    """
    The whole point of the arm. rerank answers NONE by handing back the
    retriever's shortlist and committing to it; gated records an abstention.
    """
    rec = run("gated", ranking, target_text, source,
              reply=lambda prompt: "NONE")
    assert rec.abstained
    assert not rec.committed
    assert rec.stop_reason == STOP_DECLINED
    assert rec.stop_reason != STOP_FALLBACK
    assert not rec.ranking


def test_gated_still_commits_when_the_model_answers(ranking, target_text, source):
    rec = run("gated", ranking, target_text, source, reply=accept_at("t_0"))
    assert rec.committed and not rec.abstained
    assert rec.stop_reason == STOP_ACCEPTED
    assert rec.ranking[0][0] == "t_0"


def test_an_unreadable_gated_reply_is_not_an_abstention(
        ranking, target_text, source):
    """
    The conservative rule the abstention numbers rest on: a parse failure is not
    a judgement. Counting one as a decline would credit the arm for reasoning it
    never did.
    """
    rec = run("gated", ranking, target_text, source,
              reply=lambda prompt: "I cannot help with that.")
    assert not rec.abstained
    assert rec.stop_reason != STOP_DECLINED


def test_gated_cannot_reach_gold_below_the_shortlist(
        ranking, target_text, source):
    """
    It shares rerank's ceiling, not paged's. Declining buys precision, not
    recall: the gold at rank 34 is outside the one block either arm is shown.
    """
    rec = run("gated", ranking, target_text, source, reply=accept_at("t_0"))
    assert GOLD not in [t for t, _ in rec.ranking]


# get_matches, the entry point a caller actually uses

@pytest.fixture
def oracle_order():
    """A hand-typed order, so no encoder is needed to exercise get_matches."""
    return {
        "age": [("t_0", 0.95), ("t_1", 0.40), ("t_2", 0.35)],
        "sex": [("t_1", 0.90), ("t_0", 0.41), ("t_2", 0.36)],
        "nomatch": [("t_2", 0.50), ("t_0", 0.49), ("t_1", 0.48)],
    }


@pytest.fixture
def frames(oracle_order):
    src = pd.DataFrame({c: [1, 2] for c in oracle_order})
    tgt = pd.DataFrame({c: [1, 2] for c in ("t_0", "t_1", "t_2")})
    return src, tgt


@pytest.fixture
def oracle(monkeypatch, oracle_order):
    """SchemaOracle with the retriever stubbed out."""
    monkeypatch.setattr(SchemaOracle, "get_retrieval_order",
                        lambda self, s, t: {k: list(v)
                                            for k, v in oracle_order.items()})
    return SchemaOracle


@pytest.mark.parametrize("arm", ARMS)
def test_get_matches_returns_valentine_format_for_every_arm(
        oracle, frames, arm):
    """
    Zero coverage until now, and the class did not run at all: `pager_kw` was
    assembled above the guard that consumes it, so every arm paid for the
    embedding pass and then raised KeyError on a pager parameter.
    """
    src, tgt = frames
    kw = {} if arm == "retrieval" else {
        "client": ScriptedClient(lambda p: "t_0(0.90)"), "llm_model": "fake"}
    matches = oracle(arm=arm, **kw).get_matches(src, tgt)
    assert matches, f"{arm} returned nothing"
    for (left, right), score in matches.items():
        assert left[0] == "source" and right[0] == "target"
        assert isinstance(score, float)


def test_get_matches_keeps_the_arms_own_terminal_setting(oracle, frames):
    """
    `terminal` defaults to None; False would override gated's True and leave
    its declines unresolved.
    """
    src, tgt = frames
    o = oracle(arm="gated", client=ScriptedClient(lambda p: "NONE"),
               llm_model="fake")
    o.get_matches(src, tgt)
    assert o.trace.metadata["config"]["prompt_layout"] == "oneshot_none"
    assert all(c.stop_reason == STOP_DECLINED for c in o.trace.columns)
    assert all(c.abstained and c.answered for c in o.trace.columns)


# Pruning, from the class down

def test_a_pruned_column_is_not_resurrected_by_the_rerank_fallback(
        ranking, target_text, source):
    """
    The rerank fallback does not give a pruned column the retriever's answer.
    """
    trace = run_arm("rerank", source, {"age": ranking, "nomatch": ranking},
                    target_text, client=ScriptedClient(score_first(0.60)),
                    model="fake", top_k=20, skip=["nomatch"])
    rec = {c.source_column: c for c in trace.columns}["nomatch"]
    assert rec.stop_reason == STOP_PRUNED
    assert not rec.committed and rec.ranking == [] and rec.api_calls == 0
    assert trace.metadata["n_pruned"] == 1
    assert trace.metadata["n_fallback"] == 0


def test_a_source_prune_is_recorded_but_costs_the_retrieval_arm_nothing(
        oracle, frames):
    """
    A source prune reaches the arm as `skip`, which the retrieval arm never
    reads, so the ceiling is unchanged.
    """
    src, tgt = frames
    o = oracle(arm="retrieval", prune="mutual")
    o.get_matches(src, tgt)
    assert o.prune_plan.rules == ("mutual",)
    assert o.trace.metadata["prune"]["rules"] == ["mutual"]
    assert all(c.stop_reason == STOP_RETRIEVAL for c in o.trace.columns)


def test_an_unknown_prune_rule_is_refused_before_anything_is_embedded(oracle):
    with pytest.raises(ValueError, match="unknown prune rule"):
        oracle(prune="otsu")


def test_a_target_prune_removes_the_candidate_from_the_block(
        target_text, source):
    """
    The target side is the half that changes what the model is *asked*, so the
    proof has to be the prompt itself and not the trace.
    """
    from schemaoracle import pruning

    # `t_1` and `t_2` are weakly similar and specifically close to nothing,
    # which is what a non-positive locally scaled score means.
    order = {"age": [("t_0", 0.80), ("t_1", 0.05), ("t_2", 0.05)]}
    plan = pruning.plan(order, "zero")
    assert "t_1" in plan.targets, plan.targets

    client = ScriptedClient(lambda p: "t_0(0.90)")
    run_arm("gated", source, plan.apply(order), target_text, client=client,
            model="fake", top_k=20, skip=plan.skip())
    sent = "".join(c["messages"][0]["content"] for c in client.calls)
    assert "t_1" not in sent
    assert "t_0" in sent


def test_a_target_prune_shortens_the_walk(target_text, source, ranking):
    """Fewer candidates is fewer windows, so paged pays for fewer calls."""
    from schemaoracle import pruning

    order = {"age": list(ranking)}
    full = run_arm("paged", source, order, target_text,
                   client=ScriptedClient(lambda p: "NONE"), model="fake",
                   window=10).columns[0]
    # Drop half the schema by hand: the rule's own choice is not what is
    # under test here, only that the walk follows the order it was given.
    half = {"age": order["age"][:30]}
    cut = run_arm("paged", source, half, target_text,
                  client=ScriptedClient(lambda p: "NONE"), model="fake",
                  window=10).columns[0]
    assert cut.n_blocks_available < full.n_blocks_available
    assert cut.api_calls < full.api_calls
