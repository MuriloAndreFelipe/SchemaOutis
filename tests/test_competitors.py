"""
The published matchers, run and scored like the arms. Neither `valentine` nor
`magneto` is imported.
"""

from __future__ import annotations

import json

from typing import NamedTuple

import pandas as pd
import pytest

from evaluation import scoring as report
from evaluation import competitors
from schemaoutis import cache, inventory


class ColumnPair(NamedTuple):
    """Valentine's key: flat, four fields, and a tuple."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str


def flat(source, target, score):
    return {ColumnPair("source", source, "target", target): score}


def nested(source, target, score):
    """The shape Magneto and `convert_to_valentine_format` return."""
    return {(("source", source), ("target", target)): score}


@pytest.fixture
def pair():
    return {"name": "Cao",
            "source_df": pd.DataFrame({"age": [1], "sex": [2], "zip": [3]}),
            "target_df": pd.DataFrame({"age_at_index": [1], "gender": [2]}),
            "ground_truth": [("age", "age_at_index"), ("sex", "gender")]}


# Reading a result

def test_the_flat_key_is_read_by_name_not_by_position():
    """
    Valentine's ColumnPair is a 4-tuple, so `key[0][1]` is the second letter
    of "source". That reads as a ranking over single characters and every
    check downstream accepts it.
    """
    key = ColumnPair("source", "age", "target", "age_at_index")
    assert competitors.pair_columns(key) == ("age", "age_at_index")
    assert key[0][1] == "o", "the positional read this guards against"


def test_the_nested_key_is_read_too():
    key = (("source", "age"), ("target", "age_at_index"))
    assert competitors.pair_columns(key) == ("age", "age_at_index")


def test_every_source_column_gets_an_entry(pair):
    """
    A matcher proposed nothing for `zip`. Dropping the column would let a
    matcher raise its own MRR by answering less, which is the behaviour under
    study and not a detail of the format.
    """
    got = competitors.rankings_of(flat("age", "age_at_index", 0.9),
                                  pair["source_df"].columns)
    assert set(got) == {"age", "sex", "zip"}
    assert got["age"] == [("age_at_index", 0.9)]
    assert got["sex"] == [] and got["zip"] == []


def test_a_ranking_is_best_first():
    result = {ColumnPair("source", "age", "target", "gender"): 0.2,
              ColumnPair("source", "age", "target", "age_at_index"): 0.8}
    got = competitors.rankings_of(result, ["age"])
    assert [t for t, _ in got["age"]] == ["age_at_index", "gender"]


# On disk

def test_a_competitor_file_is_not_an_arm_run(tmp_path):
    """
    It has no retriever, no shortlist and no threshold, so it must not parse
    as one. A file in `results/runs/` that did would be scanned, scored and
    reported as an arm of this pipeline.
    """
    path = competitors.path_for(tmp_path, "gdc", "Cao", "coma")
    assert cache.parse_stem(path.stem) is None
    assert path.name == "gdc__Cao__coma.json"


def test_a_run_is_cached_and_the_second_is_served(pair, tmp_path, monkeypatch):
    calls = []

    def fake(source, target):
        calls.append((source.shape, target.shape))
        return flat("age", "age_at_index", 0.9)

    monkeypatch.setitem(competitors.MATCHERS, "fake", fake)

    payload, cached = competitors.run_pair(pair, "fake", results=tmp_path,
                                           corpus="gdc")
    assert cached is False and len(calls) == 1
    assert payload["n_source_columns"] == 3
    assert payload["n_pairs_proposed"] == 1
    assert payload["rankings"]["age"] == [["age_at_index", 0.9]]

    again, cached = competitors.run_pair(pair, "fake", results=tmp_path,
                                         corpus="gdc")
    assert cached is True and len(calls) == 1, "the matcher ran twice"
    assert again == payload


def test_an_unknown_matcher_is_refused(pair, tmp_path):
    with pytest.raises(ValueError, match="nope"):
        competitors.run_pair(pair, "nope", results=tmp_path, corpus="gdc")


def test_a_competitor_directory_scans_as_nothing(pair, tmp_path, monkeypatch):
    """
    The guarantee the flat name buys, end to end: `inventory.scan` over a
    directory of competitor runs finds no arm.
    """
    monkeypatch.setitem(competitors.MATCHERS, "fake",
                        lambda s, t: flat("age", "age_at_index", 0.9))
    competitors.run_pair(pair, "fake", results=tmp_path, corpus="gdc")
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert not len(inventory.scan(tmp_path, corpus="gdc"))


# Scored by the same code as an arm

def test_scoring_pools_and_counts_silence_as_a_decline(pair, tmp_path,
                                                       monkeypatch):
    """
    `zip` was never proposed, so the matcher committed on two columns of three.
    Precision as committed reads that as an abstention, which is the same
    treatment `gated` gets and the reason this goes through `metrics`.
    """
    monkeypatch.setitem(
        competitors.MATCHERS, "fake",
        lambda s, t: {**flat("age", "age_at_index", 0.9),
                      **flat("sex", "age_at_index", 0.4)})
    competitors.run_pair(pair, "fake", results=tmp_path, corpus="gdc")

    out = report.score_competitors([pair], ["fake"], results=tmp_path,
                                   corpus="gdc")
    row = out.iloc[0].to_dict()
    assert row["matcher"] == "fake"
    assert row["n_studies"] == 1 and row["n_columns"] == 3
    assert row["n_gold_pairs"] == 2
    assert row["calls"] == 0
    # age is right, sex is wrong, zip was declined.
    assert row["tp"] == 1 and row["recall@1"] == 0.5
    assert row["precision_committed"] == pytest.approx(0.5)
    assert row["n_declined"] == 1


def test_scoring_namespaces_by_study(tmp_path, monkeypatch):
    """
    `age` exists in several studies with different gold, so pooling without
    the pair name overwrites one study's column with another's.
    """
    def make(name, gold):
        return {"name": name,
                "source_df": pd.DataFrame({"age": [1]}),
                "target_df": pd.DataFrame({gold: [1], "other": [2]}),
                "ground_truth": [("age", gold)]}

    pairs = [make("Cao", "age_at_index"), make("Dou", "days_to_birth")]
    monkeypatch.setitem(
        competitors.MATCHERS, "fake",
        lambda s, t: flat("age", list(t.columns)[0], 0.9))
    for p in pairs:
        competitors.run_pair(p, "fake", results=tmp_path, corpus="gdc")

    row = report.score_competitors(pairs, ["fake"], results=tmp_path,
                                   corpus="gdc").iloc[0].to_dict()
    assert row["n_studies"] == 2
    assert row["n_columns"] == 2, "both studies' `age` survived the pooling"
    assert row["recall@1"] == 1.0


def test_the_calls_a_matcher_made_are_counted(pair, tmp_path, monkeypatch):
    """
    A matcher that calls a model records them per column under `llm_calls`.
    Reading a flat `n_calls` key, which no payload carries, reported every
    matcher at zero calls including the one that spends a call per column, so
    the cost column of the comparison table was always empty.
    """
    monkeypatch.setitem(competitors.MATCHERS, "fake",
                        lambda s, t: flat("age", "age_at_index", 0.9))
    competitors.run_pair(pair, "fake", results=tmp_path, corpus="gdc")

    path = next(tmp_path.glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["llm_calls"] = {"age": [{"tokens_in": 30}, {"tokens_in": 20}],
                            "sex": [{"tokens_in": 10}]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    row = report.score_competitors([pair], ["fake"], results=tmp_path,
                                   corpus="gdc").iloc[0]
    assert row["calls"] == 3
    # Three source columns, so the 60 input tokens are 20 each.
    assert row["tokens_in_per_column"] == pytest.approx(20.0)


def test_a_matcher_with_nothing_on_disk_is_left_out(pair, tmp_path):
    out = report.score_competitors([pair], ["coma"], results=tmp_path,
                                   corpus="gdc")
    assert not len(out)


def test_magneto_without_its_reranker_is_a_separate_name():
    """
    The bipartite reranker is part of the published method. Registering the
    version without it under `magneto` would report a matcher missing a stage
    under the name of the one that has it.
    """
    assert {"magneto", "magneto-no-bp"} <= set(competitors.MATCHERS)
    assert competitors.MATCHERS["magneto"] is not competitors.MATCHERS["magneto-no-bp"]


def test_listing_the_matchers_imports_neither_library():
    """
    Every entry builds its matcher on call, so reading a cached result, or
    just asking what is available, needs neither library installed. Checked in
    a subprocess because pytest has imported plenty by now.
    """
    import subprocess
    import sys
    from pathlib import Path

    code = ("import sys; from evaluation import competitors as c; "
            "assert sorted(c.MATCHERS), 'empty registry'; "
            "assert 'valentine' not in sys.modules, 'valentine imported'; "
            "assert 'magneto' not in sys.modules, 'magneto imported'; "
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code],
                         cwd=str(Path(__file__).resolve().parent.parent),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
