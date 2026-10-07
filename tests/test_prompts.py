"""
The prompts, pinned byte for byte against `fixtures/prompts.json`. Regenerate
the fixture only to change what the model is asked: every cached run then
becomes a different experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemaoutis import prompts
from schemaoutis.prompts import (
    MAGNETO_PROMPT,
    PROMPT_LAYOUTS,
    TEMPLATES,
    build_prompt,
)

CAND = "Column: age, Sample values: [1,2,3]"
TARGETS = ["Column: t_a, Sample values: [x]", "Column: t_b, Sample values: [y]"]
BLOCK, N_BLOCKS = 2, 37

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "prompts.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("layout", sorted(GOLDEN))
def test_composed_template_is_byte_identical_to_the_snapshot(layout):
    assert build_prompt(CAND, TARGETS, BLOCK, N_BLOCKS,
                        layout=layout) == GOLDEN[layout]


def test_every_layout_has_a_template():
    assert set(TEMPLATES) == set(PROMPT_LAYOUTS)


def test_the_published_prompt_is_the_oneshot_layout():
    """`rerank` is faithful to Magneto only if it sends Magneto's own text."""
    assert MAGNETO_PROMPT is TEMPLATES["oneshot"]


def test_oneshot_never_mentions_blocks_or_offers_an_exit():
    """
    The two sentences that make paging possible are the entire prompt-level
    difference between the arms. If they leak into `oneshot`, the rerank arm is
    no longer the published baseline and the comparison means nothing.
    """
    text = build_prompt(CAND, TARGETS, 0, 1, layout="oneshot")
    assert "block" not in text.lower()
    assert "NONE" not in text
    assert "next block" not in text


def test_the_paged_layout_says_which_block_and_permits_declining():
    text = build_prompt(CAND, TARGETS, BLOCK, N_BLOCKS, layout="standard")
    assert "block" in text.lower()
    assert "next block" in text
    assert "NONE" in text


def test_the_block_header_is_one_based():
    text = build_prompt(CAND, TARGETS, 0, 37, layout="standard")
    assert "block 1 of 37" in text


def test_an_unknown_layout_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="no template"):
        build_prompt(CAND, TARGETS, 0, 1, layout="nonesuch")


def test_only_the_gated_layout_offers_a_final_none():
    """
    rerank's prompt must never mention NONE, or it stops being the published
    baseline; paged's offers NONE as a defer; gated's offers it as an ending.
    """
    def render(layout):
        return build_prompt("Column: age", ["Column: t"], layout=layout)

    assert "NONE" not in render("oneshot")
    assert "NONE" in render("oneshot_none")
    assert "NONE" in render("standard")

    # The distinction that makes the arm mean anything. paged promises another
    # block after a refusal; gated has none to give and says so. Test the
    # promise, not the words: "there is no next block" contains "next block".
    promise = "you will be shown the next block"
    assert promise in render("standard")
    assert promise not in render("oneshot_none")
    assert "there is no next block" in render("oneshot_none")


def test_a_name_holding_parentheses_keeps_its_score():
    from schemaoutis.prompts import _norm, parse_scores

    valid = {_norm(n): n for n in
             ["miller2_Biodiversity(marker)", "SectorID", "Amount (USD)"]}
    pairs, unmatched = parse_scores(
        "miller2_Biodiversity(marker)(0.95); SectorID(0.30); Amount (USD) (0.10)",
        valid)
    assert pairs == [("miller2_Biodiversity(marker)", 0.95), ("SectorID", 0.30),
                     ("Amount (USD)", 0.10)]
    assert unmatched == 0


def test_plain_replies_parse_as_before():
    from schemaoutis.prompts import parse_scores

    valid = {"workerid": "WorkerID", "empcode": "EmpCode"}
    pairs, _ = parse_scores("WorkerID(0.95); EmpCode(0.30)", valid)
    assert pairs == [("WorkerID", 0.95), ("EmpCode", 0.30)]
    pairs, _ = parse_scores("WorkerID(0.95)\nEmpCode(0.30)\n", valid)
    assert pairs == [("WorkerID", 0.95), ("EmpCode", 0.30)]


def test_a_name_echoed_with_its_type_and_values_counts():
    from schemaoutis.prompts import parse_response

    texts = {"alcohol_drinks_per_day": "alcohol drinks per day unknown ",
             "alcohol_frequency":
                 "alcohol frequency categorical Daily Drinker Weekly Drinker",
             "age": "age numerical 34 51"}
    valid = {"alcoholdrinksperday": "alcohol_drinks_per_day",
             "alcoholfrequency": "alcohol_frequency", "age": "age"}
    reply = ("alcohol drinks per day unknown(0.90); "
             "alcohol frequency categorical Daily Drinker(0.60); "
             "age at diagnosis numerical(0.20)")
    pairs, unmatched, _ = parse_response(reply, valid, texts=texts)
    assert pairs == [("alcohol_drinks_per_day", 0.90), ("alcohol_frequency", 0.60)]
    # Starts with `age` but is not an echo of age's line, so it stays unknown.
    assert unmatched == 1

    pairs, unmatched, note = parse_response(reply, valid)
    assert (pairs, note) == ([], "unparseable")


def test_a_name_answered_as_the_prompt_showed_it_counts():
    from schemaoutis.prompts import _norm, parse_response

    names = ["miller2_Biodiversity(marker)", "miller2_ICTAsAToolForDevelopment(marker)",
             "miller2_SectorID"]
    valid = {_norm(n): n for n in names}
    reply = ("miller2 biodiversity marker(0.95); "
             "miller2 ictas atool for development marker(0.40); "
             "miller2 sector id(0.10)")
    pairs, unmatched, note = parse_response(reply, valid)
    assert [n for n, _ in pairs] == names
    assert (unmatched, note) == (0, "parsed")


def test_a_reply_that_drops_the_shared_table_prefix_still_matches():
    """
    Every candidate in a Valentine block carries the same table prefix and the
    model often answers without it. An unambiguous tail matches; an ambiguous
    one does not, so a bare `percent` against several `...Percent` targets is
    left unmatched rather than guessed.
    """
    from schemaoutis.prompts import _norm, parse_response

    names = ["miller2_CountryRegionPercent", "miller2_SectorPercent",
             "miller2_AmountSpent"]
    valid = {_norm(n): n for n in names}

    pairs, unmatched, note = parse_response(
        "country region percent(0.85); sector percent(0.80); "
        "amount spent(0.05)", valid)
    assert [n for n, _ in pairs] == names
    assert (unmatched, note) == (0, "parsed")

    # `percent` alone ends two of the three, so it names neither.
    pairs, _, _ = parse_response("percent(0.9)", valid)
    assert pairs == []

    # A tail shorter than four characters is never enough on its own, however
    # unambiguous it looks.
    short = {_norm(n): n for n in ["miller2_Age", "table_Gender"]}
    assert parse_response("gender(0.9)", short)[0] == [("table_Gender", 0.9)]
    assert parse_response("age(0.9)", short)[0] == []
