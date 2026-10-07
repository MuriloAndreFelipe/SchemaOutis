"""Replaying decision rules on a recorded walk."""

from __future__ import annotations

from evaluation.walk_rules import replay


def col(name, block, target):
    blocks = [{"index": i, "verdict": "reject"} for i in range(block)]
    blocks.append({"index": block, "verdict": "accept"})
    return {"source_column": name, "committed": True, "accepted_block": block,
            "ranking": [(target, 0.9)], "api_calls": block + 1,
            "blocks": blocks}


def declined(columns):
    return [c["source_column"] for c in columns if not c["committed"]]


def test_no_rule_changes_nothing():
    cols = [col("a", 0, "t"), col("b", 2, "u")]
    assert replay(cols) == cols


def test_cap_declines_acceptances_at_or_past_the_cap():
    cols = [col("a", 0, "t"), col("b", 1, "u"), col("c", 3, "v")]
    assert declined(replay(cols, cap=1)) == ["b", "c"]
    assert declined(replay(cols, cap=2)) == ["c"]


def test_a_late_acceptance_of_a_taken_target_is_declined():
    cols = [col("owner", 0, "t"), col("thief", 2, "t"), col("free", 2, "u")]
    assert declined(replay(cols, target_free=True)) == ["thief"]


def test_a_declined_acceptance_loses_its_ranking():
    """The column must read as an abstention, not as a commitment with no answer."""
    out = replay([col("a", 2, "t")], cap=1)
    assert out[0]["ranking"] == [] and out[0]["abstained"]
    assert out[0]["stop_reason"] == "replay_declined"
