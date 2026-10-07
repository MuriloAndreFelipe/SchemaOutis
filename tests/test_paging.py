"""
The traversal: windows, declining, and what an unreadable or truncated walk
records.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import ScriptedClient, accept_at, score_first
from schemaoracle import Pager, block_spans, run_paging
from schemaoracle.trace import (
    STOP_ACCEPTED,
    STOP_CAPPED,
    STOP_ERROR,
    STOP_EXHAUSTED,
    STOP_UNPARSEABLE,
    STOP_UNPARSEABLE_FALLBACK,
)


def pager(reply, **kw):
    kw.setdefault("window", 20)
    kw.setdefault("accept_threshold", 0.0)
    return Pager(ScriptedClient(reply), "fake-model", **kw)


def walk(p, ranking, target_text, column="age"):
    src = pd.DataFrame({column: [1, 2, 3]})
    return run_paging(src, {column: ranking}, target_text, p).columns[0]


# Windows

def test_blocks_tile_the_ranking_exactly():
    """Every target is in exactly one block."""
    spans, available = block_spans(45, 20)
    assert spans == [(0, 20), (20, 40), (40, 45)]
    assert available == 3
    assert sum(e - s for s, e in spans) == 45


def test_a_short_final_block_is_still_a_block():
    spans, available = block_spans(21, 20)
    assert spans == [(0, 20), (20, 21)] and available == 2


def test_max_blocks_truncates_but_reports_what_was_available():
    spans, available = block_spans(100, 20, max_blocks=2)
    assert len(spans) == 2 and available == 5


def test_a_zero_window_is_refused_rather_than_looping_forever():
    with pytest.raises(ValueError, match="window"):
        block_spans(40, 0)


# Accepting and declining

def test_it_walks_until_the_window_holding_the_answer(ranking, target_text):
    rec = walk(pager(accept_at("t_30")), ranking, target_text)
    assert rec.accepted_block == 1
    assert rec.api_calls == 2
    assert rec.stop_reason == STOP_ACCEPTED
    assert rec.ranking[0][0] == "t_30"


def test_accepting_the_first_window_costs_one_call(ranking, target_text):
    rec = walk(pager(accept_at("t_5")), ranking, target_text)
    assert rec.accepted_block == 0 and rec.api_calls == 1


def test_declining_everything_is_an_answer_not_a_failure(ranking, target_text):
    """
    The distinction the whole arm rests on. A column the model saw in full and
    declined has been answered; scoring it as a failure would make the arm's
    most common correct action on a sparse corpus look like a broken run.
    """
    rec = walk(pager(lambda _: "NONE"), ranking, target_text)
    assert rec.stop_reason == STOP_EXHAUSTED
    assert rec.abstained and rec.answered and not rec.accepted
    assert rec.api_calls == 3


def test_the_threshold_declines_a_window_the_model_scored_low(ranking, target_text):
    """The second rejection signal: a model that will not say NONE but scores an
    obviously-wrong window 0.15 is still declining it."""
    rec = walk(pager(score_first(0.15), accept_threshold=0.5),
               ranking, target_text)
    assert rec.accepted_block is None
    assert rec.stop_reason == STOP_EXHAUSTED
    assert rec.blocks[0].verdict == "reject"
    assert "top_score<0.50" in rec.blocks[0].reason


def test_at_threshold_zero_only_an_explicit_none_declines(ranking, target_text):
    rec = walk(pager(score_first(0.15), accept_threshold=0.0),
               ranking, target_text)
    assert rec.accepted_block == 0
    assert "top_score>=0.00" in rec.blocks[0].reason


def test_an_out_of_range_threshold_is_refused():
    with pytest.raises(ValueError, match="accept_threshold"):
        Pager(ScriptedClient(lambda _: ""), "m", accept_threshold=1.5)


# Not advancing past something the model never answered

def test_an_unparseable_reply_stops_the_column(ranking, target_text):
    """
    Not a rejection: advancing past an unanswered window could skip the gold
    and still report a clean traversal.
    """
    rec = walk(pager(lambda _: "I am not going to answer that"),
               ranking, target_text)
    assert rec.stop_reason == STOP_UNPARSEABLE
    assert not rec.answered
    assert rec.api_calls == 1


def test_a_reply_spent_in_the_thinking_channel_says_so(ranking, target_text):
    """
    Empty content with a full thinking channel is recorded as such, not as
    unparseable: a setting fixes it, not the model.
    """
    rec = walk(pager(lambda _: ("", "Let me consider each candidate...")),
               ranking, target_text)
    assert rec.stop_reason == STOP_UNPARSEABLE
    block = rec.blocks[0]
    assert block.parse_note == "answered_in_thinking"
    assert block.n_chars == 0
    assert block.n_thinking_chars > 0


def test_an_unreadable_reply_is_not_blamed_on_the_thinking_channel(
        ranking, target_text):
    rec = walk(pager(lambda _: ("mostly prose, no scores", "some thinking")),
               ranking, target_text)
    block = rec.blocks[0]
    assert block.parse_note == "unparseable"
    assert block.n_chars > 0
    assert block.n_thinking_chars > 0


def test_a_retry_is_still_one_call_but_not_one_request(ranking, target_text):
    """
    A retry stays one call in `api_calls`; the attempts and their tokens are
    recorded.
    """
    seen = []

    def reply(prompt):
        seen.append(prompt)
        return "nothing usable" if len(seen) < 3 else "t_0(0.90)"

    p = pager(reply, max_parse_retries=3)
    rec = walk(p, ranking, target_text)

    assert rec.api_calls == 1                 # one block, one decision
    assert rec.blocks[0].n_attempts == 3      # three requests to get it
    assert len(p.client.calls) == 3
    assert rec.stop_reason == STOP_ACCEPTED
    # Tokens accumulate over the attempts, or the cost tables understate the
    # bill by up to the retry count. The fake bills 10 in and 5 out per call.
    assert rec.blocks[0].tokens_in == 30
    assert rec.blocks[0].tokens_out == 15


def test_the_retry_spells_the_format_out_and_the_first_ask_does_not(
        ranking, target_text):
    """
    A retry that changes nothing returns the same reply at temperature 0. The
    correction goes in the system turn because that is the only placement
    measured to work on a real failure.
    """
    p = pager(lambda _: "nothing usable", max_parse_retries=2)
    walk(p, ranking, target_text)

    systems = [c["system"] for c in p.client.calls]
    assert len(systems) == 3
    assert "NAME ONLY" not in systems[0]
    assert all("NAME ONLY" in s for s in systems[1:])


def test_retries_are_off_by_default_so_existing_runs_are_unchanged(
        ranking, target_text):
    p = pager(lambda _: "nothing usable")
    rec = walk(p, ranking, target_text)
    assert len(p.client.calls) == 1
    assert rec.blocks[0].n_attempts == 1
    assert rec.stop_reason == STOP_UNPARSEABLE


def test_a_reply_that_parsed_is_never_asked_again(ranking, target_text):
    p = pager(score_first(0.9), max_parse_retries=3)
    rec = walk(p, ranking, target_text)
    assert len(p.client.calls) == 1
    assert rec.blocks[0].n_attempts == 1


def test_an_explicit_none_is_an_answer_and_is_not_retried(ranking, target_text):
    """NONE is the model declining, not failing to be read."""
    p = pager(lambda _: "NONE", max_parse_retries=3, max_blocks=1,
              terminal=True)
    walk(p, ranking, target_text)
    assert len(p.client.calls) == 1


def test_a_turn_spent_in_the_thinking_channel_is_not_retried(
        ranking, target_text):
    """
    Asking again buys the same failure at the same price: that one needs a
    bigger budget or `think=False`, which a retry cannot supply.
    """
    p = pager(lambda _: ("", "deliberating at length"), max_parse_retries=3)
    rec = walk(p, ranking, target_text)
    assert len(p.client.calls) == 1
    assert rec.blocks[0].parse_note == "answered_in_thinking"


def test_the_retry_count_cannot_exceed_the_ceiling():
    from schemaoracle.paging import MAX_PARSE_RETRIES

    assert MAX_PARSE_RETRIES == 3
    with pytest.raises(ValueError, match="max_parse_retries"):
        pager(lambda _: "x", max_parse_retries=MAX_PARSE_RETRIES + 1)


def test_a_retry_setting_is_recorded_but_is_not_the_run_identity():
    """
    A retry changes a request only after a reply fails to parse, so a column
    that parsed was sent the same bytes either way and the two runs are the
    same measurement. It is recorded, so what happened stays visible, but it
    must not block a resume: refusing that is what forced runs to be split
    across two filenames and copied between them.
    """
    from schemaoracle.resume import IDENTITY_FIELDS

    assert "max_parse_retries" not in IDENTITY_FIELDS
    assert "max_parse_retries" in pager(lambda _: "x").config()


def test_an_unknown_unparseable_policy_is_refused(ranking, target_text):
    """
    "skip" was one of these and walked on to the next window, which is the one
    policy that changed the requests sent. No run on disk used it.
    """
    with pytest.raises(ValueError, match="on_unparseable"):
        pager(lambda _: "x", on_unparseable="skip")


def test_fallback_policy_hands_the_column_the_retrievers_window(
        ranking, target_text, capsys):
    """
    As rerank does: the walk stops at the unreadable block, and the retriever's
    window stands as the answer instead of leaving a hole.
    """
    rec = walk(pager(lambda _: "I am not going to answer that",
                     on_unparseable="fallback", window=20),
               ranking, target_text)

    assert rec.stop_reason == STOP_UNPARSEABLE_FALLBACK
    assert rec.api_calls == 1                  # stopped, same as the stop policy
    assert [t for t, _ in rec.ranking] == [t for t, _ in ranking[:20]]
    assert not rec.error                       # a fallback is not a failure
    assert "falling back" in capsys.readouterr().out


def test_a_fallback_is_answered_and_committed_not_an_abstention(
        ranking, target_text):
    """
    The conservative reading: a parse failure is not a decline. Counting it as
    one would credit the arm for judgement it never exercised and inflate the
    abstention precision the paper's argument rests on.
    """
    rec = walk(pager(lambda _: "unreadable", on_unparseable="fallback"),
               ranking, target_text)
    assert rec.answered and rec.committed and not rec.abstained


def test_a_fallback_is_not_relabelled_blocks_exhausted(ranking, target_text):
    """
    A fallback has no accepted block, which is the same shape as a walk that
    declined everything. Without an explicit guard the end-of-walk branch
    overwrites both the stop reason and the retriever's window.
    """
    rec = walk(pager(lambda _: "unreadable", on_unparseable="fallback"),
               ranking, target_text)
    assert rec.stop_reason != STOP_EXHAUSTED
    assert len(rec.ranking) == 20


def test_an_api_error_is_recorded_and_stops_the_column(ranking, target_text):
    def boom(_):
        raise RuntimeError("credit balance is too low")
    rec = walk(pager(boom), ranking, target_text)
    assert rec.stop_reason == STOP_ERROR
    assert "credit balance" in rec.error
    assert not rec.answered


# What gets reported

def test_a_capped_walk_is_not_reported_as_exhaustive(ranking, target_text):
    """A run that stopped early because of `max_blocks` is a different
    measurement from one that saw the whole schema, and must say so."""
    rec = walk(pager(lambda _: "NONE", max_blocks=2), ranking, target_text)
    assert rec.stop_reason == STOP_CAPPED
    assert rec.n_blocks_available == 3 and rec.api_calls == 2


def test_declining_everything_still_keeps_the_models_own_ordering(ranking,
                                                                 target_text):
    """It scored what it saw; reporting the cosine order instead would discard a
    real result. `stop_reason` keeps it filterable."""
    rec = walk(pager(score_first(0.30), accept_threshold=0.9),
               ranking, target_text)
    assert rec.ranking, "pooled scores were dropped"
    assert rec.stop_reason == STOP_EXHAUSTED


def test_api_calls_is_derived_and_cannot_disagree_with_the_blocks(ranking,
                                                                 target_text):
    rec = walk(pager(accept_at("t_45")), ranking, target_text)
    assert rec.api_calls == len(rec.blocks) == 3


def test_only_columns_restricts_the_walk(source, ranking, target_text):
    """How a resume avoids re-paying for columns that already succeeded."""
    p = pager(accept_at("t_5"))
    trace = run_paging(source, {"age": ranking, "nomatch": ranking},
                       target_text, p, only_columns=["nomatch"])
    assert [c.source_column for c in trace.columns] == ["nomatch"]
    assert trace.metadata["only_columns"] == ["nomatch"]


def test_a_column_with_no_ranking_raises_rather_than_being_skipped(source,
                                                                  target_text):
    with pytest.raises(KeyError, match="ranking"):
        run_paging(source, {}, target_text, pager(lambda _: "NONE"),
                   only_columns=["age"])


def test_the_config_records_everything_that_changes_the_question(ranking,
                                                                 target_text):
    cfg = pager(lambda _: "NONE", accept_threshold=0.5, window=10).config()
    for field in ("model", "window", "accept_threshold",
                  "prompt_layout", "on_unparseable", "seed"):
        assert field in cfg


def test_the_trace_records_a_parameter_the_backend_refused(
        ranking, target_text, source):
    """
    `config()` lists what the run asked for. A run whose temperature was
    refused should not record one, or the trace claims a setting that never
    reached the model.
    """
    from schemaoracle.llm import DropsRejectedParams
    from schemaoracle.paging import Pager, run_paging

    class Rejecting:
        """Refuses temperature the first time, like a newer checkpoint."""

        def __init__(self, inner):
            self.messages = inner

    inner = ScriptedClient(score_first(0.9))
    real_create = inner.create

    def create(**kwargs):
        if "temperature" in kwargs:
            import anthropic
            import httpx
            raise anthropic.BadRequestError(
                "`temperature` is deprecated for this model.",
                response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                body=None)
        return real_create(**kwargs)

    inner.create = create
    client = DropsRejectedParams(Rejecting(inner))

    pager = Pager(client, "fake", window=20, max_blocks=1)
    trace = run_paging(source, {c: ranking for c in source.columns},
                       target_text, pager)

    assert trace.metadata["dropped_params"] == ["temperature"]
    assert trace.completeness()["n_errored"] == 0
