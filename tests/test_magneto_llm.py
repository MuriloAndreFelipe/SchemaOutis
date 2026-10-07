"""Magneto's LLM reranker on our order, with the LLM faked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("magneto")

import magneto.llm_reranker as reranker_module  # noqa: E402

from evaluation import competitors  # noqa: E402
from schemaoutis import cache  # noqa: E402

CORPUS = "gdc"


@pytest.fixture
def pair():
    return {"name": "Cao",
            "source_df": pd.DataFrame({"age": [1, 2], "sex": ["m", "f"]}),
            "target_df": pd.DataFrame({"t_age": [1, 2], "t_sex": ["m", "f"],
                                       "t_zip": [9, 8]}),
            "ground_truth": [("age", "t_age")]}


@pytest.fixture
def runs(tmp_path, pair):
    runs = tmp_path / "runs"
    runs.mkdir()
    order = {"age": [["t_zip", 0.9], ["t_age", 0.8], ["t_sex", 0.1]],
             "sex": [["t_sex", 0.7], ["t_age", 0.2], ["t_zip", 0.1]]}
    cache.retrieval_path(runs, CORPUS, pair["name"], "qwen").write_text(
        json.dumps({"rankings": order}), encoding="utf-8")
    return runs


def reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=text))], usage=SimpleNamespace(prompt_tokens=10,
                                               completion_tokens=5))


@pytest.fixture
def llm(monkeypatch):
    sent = []

    def fake(**kw):
        sent.append(kw)
        cand = kw["messages"][1]["content"].split("Candidate Column: ")[-1]
        best = "t_age" if cand.startswith("Column: age") else "t_sex"
        return reply(json.dumps([{"column": best, "score": 0.9}]))

    monkeypatch.setattr(reranker_module, "completion", fake)
    monkeypatch.setattr("schemaoutis.llm.load_env", lambda *a, **k: [])
    return sent


def test_the_shortlist_is_our_order_and_the_reply_is_the_ranking(
        pair, runs, tmp_path, llm):
    out = competitors.run_magneto_llm(pair, results=tmp_path / "c",
                                      corpus=CORPUS, runs=runs, verbose=False)
    assert [t for t, _ in out["candidates"]["age"]][:2] == ["t_zip", "t_age"]
    assert out["rankings"]["age"][0][0] == "t_age"
    assert out["complete"] and out["fallback"] == []
    assert out["llm_calls"]["age"][0]["reply"]
    assert llm[0]["model"] == "openrouter/deepseek/deepseek-v4.1-flash"


def test_a_second_run_sends_nothing(pair, runs, tmp_path, llm):
    kw = dict(results=tmp_path / "c", corpus=CORPUS, runs=runs, verbose=False)
    competitors.run_magneto_llm(pair, **kw)
    n = len(llm)
    competitors.run_magneto_llm(pair, **kw)
    assert len(llm) == n


def test_an_api_error_keeps_the_finished_columns(pair, runs, tmp_path,
                                                 monkeypatch):
    monkeypatch.setattr("schemaoutis.llm.load_env", lambda *a, **k: [])
    calls = []

    def fails_second(**kw):
        calls.append(kw)
        if len(calls) > 1:
            raise RuntimeError("401")
        return reply(json.dumps([{"column": "t_age", "score": 0.9}]))

    monkeypatch.setattr(reranker_module, "completion", fails_second)
    kw = dict(results=tmp_path / "c", corpus=CORPUS, runs=runs, verbose=False,
              workers=1)
    with pytest.raises(RuntimeError):
        competitors.run_magneto_llm(pair, **kw)
    tag = competitors.magneto_llm_tag("qwen", "deepseek/deepseek-v4.1-flash")
    on_disk = json.loads(competitors.path_for(tmp_path / "c", CORPUS, "Cao", tag)
                         .read_text(encoding="utf-8"))
    assert list(on_disk["rankings"]) == ["age"] and not on_disk["complete"]
    assert competitors.load(tmp_path / "c", CORPUS, "Cao", tag) is None


def test_the_bipartite_stage_runs_on_valentine_1(pair):
    """It mutates its input, which valentine 1.0 returns read-only."""
    from valentine import MatcherResults
    import magneto.magneto as mm

    m = MatcherResults({(("s", "age"), ("t", "t_age")): 0.9,
                        (("s", "age"), ("t", "t_sex")): 0.5,
                        (("s", "sex"), ("t", "t_age")): 0.8,
                        (("s", "sex"), ("t", "t_sex")): 0.7})
    with competitors._bp_on_dicts():
        out = mm.arrange_bipartite_matches(m, pair["source_df"], "s",
                                           pair["target_df"], "t")
    best = {s[1]: max((v, t[1]) for (s2, t), v in out.items() if s2 == s)[1]
            for s, _ in out}
    assert best == {"age": "t_age", "sex": "t_sex"}
    assert mm.arrange_bipartite_matches.__module__ == "magneto.bp_reranker"


def test_the_tag_says_whether_the_bipartite_stage_ran():
    llm = "deepseek/deepseek-v4.1-flash"
    assert competitors.magneto_llm_tag("qwen", llm).startswith("magneto-llm__")
    assert competitors.magneto_llm_tag("qwen", llm, bp=False).startswith(
        "magneto-no-bp-llm__")


def test_free_magneto_on_our_order_uses_that_order(pair, runs, tmp_path):
    out, cached = competitors.run_pair(pair, "magneto__qwen",
                                       results=tmp_path / "c", corpus=CORPUS,
                                       runs=runs)
    assert not cached
    assert {t for t, _ in out["rankings"]["age"]} == {"t_zip", "t_age", "t_sex"}
    assert competitors.run_pair(pair, "magneto__qwen", results=tmp_path / "c",
                                corpus=CORPUS, runs=runs)[1]


def test_free_magneto_on_an_order_needs_the_runs_directory(pair, tmp_path):
    with pytest.raises(ValueError, match="runs="):
        competitors.run_pair(pair, "magneto__qwen", results=tmp_path / "c",
                             corpus=CORPUS)
