"""
The runner: cache keys, the dry-run guard, resuming, and reading runs back.
"""

from __future__ import annotations

import json
from itertools import combinations

import pandas as pd
import pytest

from conftest import ScriptedClient, score_first
from schemaoracle import cache, inventory, pruning
from schemaoracle import run
from schemaoracle.run import run_pair
from schemaoracle.resume import errored_columns

CORPUS, RETRIEVER, ARM = "gdc", "qwen", "claude-sonnet-5__block_cached"


@pytest.fixture
def pair(targets, source):
    return {"name": "Cao", "source_df": source,
            "target_df": pd.DataFrame({t: [1] for t in targets}),
            "ground_truth": [("age", "t_3")]}


@pytest.fixture
def rankings(ranking, source):
    return {col: ranking for col in source.columns}


def do(pair, rankings, target_text, tmp_path, kind, **kw):
    kw.setdefault("dry_run", False)
    kw.setdefault("arm", ARM)
    return run_pair(pair, kind=kind, results=tmp_path, corpus=CORPUS,
                    retriever=RETRIEVER, rankings=rankings,
                    targets=target_text, client=ScriptedClient(score_first(0.6)),
                    model="fake", **kw)


# Cache keys

def test_each_arm_gets_its_own_filename(tmp_path):
    paths = {
        kind: cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, kind,
                             arm=ARM, window=20, threshold=0.0, top_k=20).name
        for kind in ("retrieval", "rerank", "gated", "paged")
    }
    assert len(set(paths.values())) == 4
    assert paths["retrieval"].endswith("__retrieval.json")
    assert paths["rerank"].endswith("__rerank.json")
    assert paths["gated"].endswith("__gated.json")
    assert paths["paged"].endswith("__paged.json")


def test_retrieval_key_carries_no_model_and_no_threshold():
    """
    It contains no LLM and cannot differ between them, so one embedding pass
    serves every arm and every threshold. Putting a model in the key would
    re-embed a GDC study once per arm for nothing.
    """
    name = cache.run_path("r", CORPUS, "Cao", RETRIEVER, "retrieval").name
    assert ARM not in name and "th0" not in name


def test_rerank_key_carries_no_threshold():
    """The arm always commits, so a threshold would split one run into many."""
    name = cache.run_path("r", CORPUS, "Cao", RETRIEVER, "rerank",
                          arm=ARM, top_k=20).name
    assert "th" not in name and "k20" in name


def test_paged_key_carries_the_window_and_the_threshold():
    """Both change what the model was asked, so both must split the cache."""
    a = cache.run_path("r", CORPUS, "Cao", RETRIEVER, "paged", arm=ARM,
                       window=20, threshold=0.0)
    b = cache.run_path("r", CORPUS, "Cao", RETRIEVER, "paged", arm=ARM,
                       window=20, threshold=0.5)
    c = cache.run_path("r", CORPUS, "Cao", RETRIEVER, "paged", arm=ARM,
                       window=40, threshold=0.0)
    assert len({a.name, b.name, c.name}) == 3


@pytest.mark.parametrize("kind", ["retrieval", "rerank", "gated", "paged"])
def test_a_filename_parses_back_into_the_fields_that_made_it(kind):
    path = cache.run_path("r", CORPUS, "Cao", RETRIEVER, kind, arm=ARM,
                          window=20, threshold=0.25, top_k=20)
    got = cache.parse_stem(path.stem)
    assert got["kind"] == kind
    assert got["corpus"] == CORPUS and got["pair"] == "Cao"
    assert got["retriever"] == RETRIEVER


def test_a_pair_name_containing_a_separator_still_parses():
    """Valentine pair names carry their subset, e.g. `ChEMBL__Joinable__assays`."""
    path = cache.run_path("r", "valentine", "ChEMBL__Joinable__x", RETRIEVER,
                          "paged", arm=ARM, window=20, threshold=0.0)
    assert cache.parse_stem(path.stem)["pair"] == "ChEMBL__Joinable__x"


def test_a_foreign_filename_is_not_mistaken_for_a_run():
    assert cache.parse_stem("notes") is None
    assert cache.parse_stem("embeddings.duckdb") is None


def test_an_unknown_kind_raises_rather_than_writing_somewhere_odd():
    with pytest.raises(ValueError, match="unknown kind"):
        cache.run_path("r", CORPUS, "Cao", RETRIEVER, "magic")


# Running

def test_dry_run_refuses_to_spend_and_says_what_it_would_cost(
        pair, rankings, target_text, tmp_path):
    with pytest.raises(RuntimeError, match="dry_run is True"):
        do(pair, rankings, target_text, tmp_path, "paged", dry_run=True)


def test_dry_run_does_not_block_the_arm_that_costs_nothing(
        pair, rankings, target_text, tmp_path):
    payload, cached = do(pair, rankings, target_text, tmp_path, "retrieval",
                         dry_run=True)
    assert not cached
    assert payload["trace"]["totals"]["api_calls"] == 0


@pytest.mark.parametrize("kind", ["retrieval", "rerank", "paged"])
def test_a_run_is_written_once_and_served_from_disk_after(
        pair, rankings, target_text, tmp_path, kind):
    first, cached_first = do(pair, rankings, target_text, tmp_path, kind)
    second, cached_second = do(pair, rankings, target_text, tmp_path, kind)
    assert not cached_first and cached_second
    assert first["trace"]["columns"] == second["trace"]["columns"]


def test_the_worst_case_matches_what_the_arm_can_actually_spend():
    from schemaoracle.run import worst_case_calls

    assert worst_case_calls("retrieval", 39, 736, 20) == 0
    assert worst_case_calls("rerank", 39, 736, 20) == 39
    assert worst_case_calls("paged", 39, 736, 20) == 39 * 37


# Reading it back

def test_every_arm_round_trips_through_the_inventory(
        pair, rankings, target_text, tmp_path):
    for kind in ("retrieval", "rerank", "paged"):
        do(pair, rankings, target_text, tmp_path, kind)

    inv = inventory.scan(tmp_path, corpus=CORPUS)
    assert sorted(inv["kind"]) == ["paged", "rerank", "retrieval"]
    assert set(inv["pair"]) == {"Cao"}

    for r in inv.itertuples():
        cols = inventory.load(r)["trace"]["columns"]
        assert len(cols) == pair["source_df"].shape[1]
        assert all("committed" in c for c in cols)


def test_scanning_can_be_narrowed_to_one_arm(
        pair, rankings, target_text, tmp_path):
    for kind in ("retrieval", "rerank", "paged"):
        do(pair, rankings, target_text, tmp_path, kind)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))
    assert list(inv["kind"]) == ["paged"]


def test_a_poisoned_run_is_refused_rather_than_served_back(
        pair, rankings, target_text, tmp_path):
    """
    A run that died partway writes a file indistinguishable from a healthy one.
    Serving it back and printing `cached` is how made-up numbers reach a paper.
    """
    def boom(prompt):
        raise RuntimeError("api is down")

    run_pair(pair, kind="paged", results=tmp_path, corpus=CORPUS,
             retriever=RETRIEVER, arm=ARM, rankings=rankings,
             targets=target_text, client=ScriptedClient(boom), model="fake",
             dry_run=False)

    path = cache.run_path(tmp_path, CORPUS, pair["name"], RETRIEVER, "paged",
                          arm=ARM, window=20, threshold=0.0)
    assert path.exists()
    with pytest.raises(cache.PoisonedCache, match="ended in an API"):
        cache.load(path)
    assert cache.load(path, allow_errors=True) is not None


def test_a_run_that_is_not_there_reads_as_absent_not_as_empty(tmp_path):
    assert cache.load(tmp_path / "nope.json") is None


# Repeats

def test_the_first_run_of_a_configuration_carries_no_repeat_tag(tmp_path):
    """
    Runs written before repeats existed carry no tag. If `rep=1` renamed them,
    the whole cache would read as missing and be re-run.
    """
    for kind, kw in (("rerank", {"top_k": 20}),
                     ("paged", {"window": 20, "threshold": 0.0})):
        first = cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, kind,
                               arm=ARM, rep=1, **kw).name
        assert "__r1__" not in first
        assert first.endswith(f"__{kind}.json")


def test_a_repeat_gets_its_own_file_rather_than_the_first_one(tmp_path):
    paths = {rep: cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, "paged",
                                 arm=ARM, window=20, threshold=0.0, rep=rep)
             for rep in (1, 2, 3)}
    assert len({p.name for p in paths.values()}) == 3
    assert paths[2].name.endswith("__r2__paged.json")


def test_a_repeat_is_not_served_the_first_run_from_cache(
        pair, rankings, target_text, tmp_path):
    """
    The reason `rep` exists. Three runs of one configuration are three
    measurements of a model that does not always answer the same way; without a
    slot in the key the second and third are the first, read back.
    """
    _, cached_first = do(pair, rankings, target_text, tmp_path, "paged", rep=1)
    _, cached_again = do(pair, rankings, target_text, tmp_path, "paged", rep=1)
    _, cached_second = do(pair, rankings, target_text, tmp_path, "paged", rep=2)

    assert (cached_first, cached_again, cached_second) == (False, True, False)


def test_a_repeated_filename_parses_back_with_its_repeat(tmp_path):
    for rep in (1, 2, 7):
        for kind, kw in (("rerank", {"top_k": 20}),
                         ("paged", {"window": 20, "threshold": 0.5})):
            stem = cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, kind,
                                  arm=ARM, rep=rep, **kw).stem
            g = cache.parse_stem(stem)
            assert (g["kind"], g["rep"], g["arm"]) == (kind, rep, ARM)


def test_repeats_are_a_column_of_the_inventory(
        pair, rankings, target_text, tmp_path):
    for rep in (1, 2, 3):
        do(pair, rankings, target_text, tmp_path, "paged", rep=rep)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))
    assert sorted(inv["rep"]) == [1, 2, 3]


def test_rep_is_one_based(tmp_path):
    with pytest.raises(ValueError, match="1-based"):
        cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, "paged", arm=ARM,
                       rep=0)


# Reading an arm tag back apart

def test_the_provider_splits_back_out_of_an_arm_tag():
    """
    Inherited filenames carry the provider; ours must too, or a run written here
    and the equivalent one carried over land on different rows of the inventory
    and each looks like the other is missing.
    """
    tag = cache.arm_tag("anthropic", "claude-sonnet-5", "block_cached")
    assert cache.split_arm(tag) == ("anthropic", "claude-sonnet-5__block_cached")


def test_a_model_name_that_is_not_a_provider_is_left_alone():
    assert cache.split_arm("claude-sonnet-5__block_cached") == (
        "", "claude-sonnet-5__block_cached")


def test_an_inventory_row_splits_the_provider_off_the_arm(
        pair, rankings, target_text, tmp_path):
    arm = cache.arm_tag("anthropic", "claude-sonnet-5", "block_cached")
    do(pair, rankings, target_text, tmp_path, "paged", arm=arm)
    row = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",)).iloc[0]
    assert (row["provider"], row["arm"]) == ("anthropic",
                                             "claude-sonnet-5__block_cached")


# Coverage: what is already paid for

def test_coverage_counts_what_is_done_and_names_what_is_missing(
        pair, rankings, target_text, tmp_path):
    do(pair, rankings, target_text, tmp_path, "paged")
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))

    cov = inventory.coverage(inv, ["Cao", "Clark", "Dou"], kind="paged")
    assert len(cov) == 1
    assert cov.loc[0, "done"] == 1
    assert cov.loc[0, "todo"] == 2
    assert cov.loc[0, "missing"] == ["Clark", "Dou"]


def test_coverage_counts_a_missing_repeat_as_work_left(
        pair, rankings, target_text, tmp_path):
    do(pair, rankings, target_text, tmp_path, "paged", rep=1)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))

    cov = inventory.coverage(inv, ["Cao"], kind="paged", reps=3)
    assert (cov.loc[0, "done"], cov.loc[0, "todo"]) == (1, 2)
    # The study is covered; it is the repeats that are short, so naming the
    # study as missing would be misleading.
    assert cov.loc[0, "missing"] == ["Cao"]


def test_todo_lists_the_runs_that_would_actually_be_sent(
        pair, rankings, target_text, tmp_path):
    do(pair, rankings, target_text, tmp_path, "paged", rep=1)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))

    left = inventory.todo(inv, ["Cao", "Clark"], kind="paged", reps=2,
                          retriever=RETRIEVER, arm=ARM)
    assert left == [("Cao", 2), ("Clark", 1), ("Clark", 2)]


def test_todo_of_a_configuration_never_run_is_the_whole_corpus(
        pair, rankings, target_text, tmp_path):
    do(pair, rankings, target_text, tmp_path, "paged")
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("paged",))

    left = inventory.todo(inv, ["Cao"], kind="paged", retriever="other-model")
    assert left == [("Cao", 1)]
# Giving up on a run that cannot recover

def _always_fails(prompt):
    raise RuntimeError("Could not resolve authentication method")


def test_a_run_stops_instead_of_repeating_one_unrecoverable_error(
        pair, rankings, target_text, tmp_path):
    """
    An expired key fails every column identically. Walking on produced 1,707
    error records across 30 files and one very long wait, when five would have
    said the same thing.
    """
    from schemaoracle.paging import RunAborted

    wide = {f"c{i}": rankings["age"] for i in range(30)}
    source = pd.DataFrame({c: [1] for c in wide})
    p = {**pair, "source_df": source}

    with pytest.raises(RunAborted, match="failed and the run stopped"):
        run_pair(p, kind="paged", results=tmp_path, corpus=CORPUS,
                 retriever=RETRIEVER, arm=ARM, rankings=wide,
                 targets=target_text, client=ScriptedClient(_always_fails),
                 model="fake", dry_run=False)


def test_nothing_is_written_when_no_column_ever_answered(
        pair, rankings, target_text, tmp_path):
    """A file of nothing but errors is one the next attempt has to be told to
    ignore. There is nothing in it to resume from."""
    from schemaoracle.paging import RunAborted

    wide = {f"c{i}": rankings["age"] for i in range(30)}
    source = pd.DataFrame({c: [1] for c in wide})

    with pytest.raises(RunAborted):
        run_pair({**pair, "source_df": source}, kind="paged", results=tmp_path,
                 corpus=CORPUS, retriever=RETRIEVER, arm=ARM, rankings=wide,
                 targets=target_text, client=ScriptedClient(_always_fails),
                 model="fake", dry_run=False)
    assert list(tmp_path.glob("*.json")) == []


def test_the_columns_that_answered_before_the_failure_are_kept(
        pair, rankings, target_text, tmp_path):
    """
    Those cost money. Keeping them means the fix plus resume=True finishes the
    pair instead of starting it again.
    """
    from schemaoracle.paging import RunAborted

    ok = score_first(0.6)
    calls = {"n": 0}

    def fails_after_three(prompt):
        calls["n"] += 1
        if calls["n"] <= 3:
            return ok(prompt)
        raise RuntimeError("balance exhausted")

    wide = {f"c{i}": rankings["age"] for i in range(30)}
    source = pd.DataFrame({c: [1] for c in wide})

    with pytest.raises(RunAborted, match="Kept 3 answered column"):
        run_pair({**pair, "source_df": source}, kind="rerank", results=tmp_path,
                 corpus=CORPUS, retriever=RETRIEVER, arm=ARM, rankings=wide,
                 targets=target_text, client=ScriptedClient(fails_after_three),
                 model="fake", dry_run=False)

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    payload = cache.load(written[0], allow_errors=True)
    assert payload["trace"]["completeness"]["n_committed"] == 3
    assert len(errored_columns(payload)) == 5


def test_a_healthy_run_is_not_stopped_by_scattered_failures(
        pair, rankings, target_text, tmp_path):
    """Five in a row, not five in total. A couple of flaky requests are not a
    dead key."""
    from schemaoracle.resume import errored_columns as errs

    n = {"i": 0}
    ok = score_first(0.6)

    def flaky(prompt):
        n["i"] += 1
        if n["i"] % 4 == 0:
            raise RuntimeError("transient")
        return ok(prompt)

    wide = {f"c{i}": rankings["age"] for i in range(30)}
    source = pd.DataFrame({c: [1] for c in wide})

    payload, _ = run_pair({**pair, "source_df": source}, kind="rerank",
                          results=tmp_path, corpus=CORPUS, retriever=RETRIEVER,
                          arm=ARM, rankings=wide, targets=target_text,
                          client=ScriptedClient(flaky), model="fake",
                          dry_run=False)
    assert len(payload["trace"]["columns"]) == 30
    assert 0 < len(errs(payload)) < 30


# Finishing a run that stopped partway

def test_a_resume_runs_the_columns_the_first_attempt_never_reached(
        pair, rankings, target_text, tmp_path):
    """A resume also runs the columns an aborted walk never reached."""
    wide = {f"c{i}": rankings["age"] for i in range(8)}
    source = pd.DataFrame({c: [1] for c in wide})
    p = {**pair, "source_df": source}
    part = {k: wide[k] for k in ("c0", "c1", "c2")}

    first, _ = run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
                        retriever=RETRIEVER, arm=ARM, rankings=part,
                        targets=target_text,
                        client=ScriptedClient(score_first(0.6)), model="fake",
                        dry_run=False)
    assert len(first["trace"]["columns"]) == 3

    done, _ = run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
                       retriever=RETRIEVER, arm=ARM, rankings=wide,
                       targets=target_text,
                       client=ScriptedClient(score_first(0.6)), model="fake",
                       dry_run=False, resume=True)
    assert len(done["trace"]["columns"]) == 8
    assert {c["source_column"] for c in done["trace"]["columns"]} == set(wide)


def test_a_resume_keeps_the_answers_the_first_attempt_did_get(
        pair, rankings, target_text, tmp_path):
    """The whole reason to resume rather than re-run: those cost money."""
    wide = {f"c{i}": rankings["age"] for i in range(6)}
    source = pd.DataFrame({c: [1] for c in wide})
    p = {**pair, "source_df": source}
    part = {k: wide[k] for k in ("c0", "c1")}

    first, _ = run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
                        retriever=RETRIEVER, arm=ARM, rankings=part,
                        targets=target_text,
                        client=ScriptedClient(score_first(0.9)), model="fake",
                        dry_run=False)
    kept = {c["source_column"]: c["ranking"][0]
            for c in first["trace"]["columns"]}

    done, _ = run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
                       retriever=RETRIEVER, arm=ARM, rankings=wide,
                       targets=target_text,
                       client=ScriptedClient(score_first(0.1)), model="fake",
                       dry_run=False, resume=True)
    after = {c["source_column"]: c["ranking"][0]
             for c in done["trace"]["columns"]}
    assert len(after) == 6
    for col, first_answer in kept.items():
        assert after[col] == first_answer      # not re-scored at 0.1


def test_a_complete_run_is_still_served_from_cache_under_resume(
        pair, rankings, target_text, tmp_path):
    """resume=True must not turn every cache hit into a re-run."""
    do(pair, rankings, target_text, tmp_path, "rerank")
    _, cached = do(pair, rankings, target_text, tmp_path, "rerank", resume=True)
    assert cached


def test_an_aborted_resume_keeps_the_columns_it_did_reach(
        pair, rankings, target_text, tmp_path):
    """
    A resume that aborts has records for the columns it got to and none for the
    rest. Splicing the full request raised KeyError on the difference and threw
    away every call the resume had already paid for.
    """
    from schemaoracle.paging import RunAborted

    wide = {f"c{i}": rankings["age"] for i in range(30)}
    source = pd.DataFrame({c: [1] for c in wide})
    p = {**pair, "source_df": source}

    # A first pass over three columns, so the payload is short by 27.
    run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
             retriever=RETRIEVER, arm=ARM,
             rankings={k: wide[k] for k in ("c0", "c1", "c2")},
             targets=target_text, client=ScriptedClient(score_first(0.9)),
             model="fake", dry_run=False)

    # The resume answers four more, then the backend dies for good.
    ok = score_first(0.6)
    seen = {"n": 0}

    def dies_after_four(prompt):
        seen["n"] += 1
        if seen["n"] <= 4:
            return ok(prompt)
        raise RuntimeError("balance exhausted")

    with pytest.raises(RunAborted, match="Kept"):
        run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
                 retriever=RETRIEVER, arm=ARM, rankings=wide,
                 targets=target_text, client=ScriptedClient(dies_after_four),
                 model="fake", dry_run=False, resume=True)

    path = cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, "rerank", arm=ARM)
    saved = cache.load(path, allow_errors=True)
    answered = [c for c in saved["trace"]["columns"] if c["committed"]]
    assert len(answered) == 7          # 3 from the first pass, 4 from the resume
    assert len(saved["trace"]["columns"]) < 30   # the rest were never reached


def test_the_next_resume_picks_up_where_that_one_stopped(
        pair, rankings, target_text, tmp_path):
    """The third attempt finishes the pair from what the second kept."""
    wide = {f"c{i}": rankings["age"] for i in range(12)}
    source = pd.DataFrame({c: [1] for c in wide})
    p = {**pair, "source_df": source}

    run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
             retriever=RETRIEVER, arm=ARM,
             rankings={k: wide[k] for k in ("c0", "c1")},
             targets=target_text, client=ScriptedClient(score_first(0.9)),
             model="fake", dry_run=False)

    done, _ = run_pair(p, kind="rerank", results=tmp_path, corpus=CORPUS,
                       retriever=RETRIEVER, arm=ARM, rankings=wide,
                       targets=target_text,
                       client=ScriptedClient(score_first(0.6)), model="fake",
                       dry_run=False, resume=True)
    assert len(done["trace"]["columns"]) == 12
    assert {c["source_column"] for c in done["trace"]["columns"]} == set(wide)


def test_an_unreadable_reply_is_not_retried_for_ever(
        pair, rankings, target_text, tmp_path):
    """
    An unreadable reply is not retried; at temperature 0 it would come back the
    same.
    """
    calls = {"n": 0}

    def unreadable(prompt):
        calls["n"] += 1
        return "I am not able to help with that request."

    first, cached_first = run_pair(
        pair, kind="rerank", results=tmp_path, corpus=CORPUS,
        retriever=RETRIEVER, arm=ARM, rankings=rankings, targets=target_text,
        client=ScriptedClient(unreadable), model="fake", dry_run=False)
    assert not cached_first
    spent = calls["n"]
    assert spent > 0

    again, cached_again = run_pair(
        pair, kind="rerank", results=tmp_path, corpus=CORPUS,
        retriever=RETRIEVER, arm=ARM, rankings=rankings, targets=target_text,
        client=ScriptedClient(unreadable), model="fake", dry_run=False,
        resume=True)
    assert cached_again, "an unreadable reply was retried"
    assert calls["n"] == spent, "and it cost calls to do it"


def test_an_api_error_is_still_retried(pair, rankings, target_text, tmp_path):
    """The other half of the distinction: that column was never asked."""
    from schemaoracle.resume import retryable_columns

    def boom(prompt):
        raise RuntimeError("api is down")

    run_pair(pair, kind="rerank", results=tmp_path, corpus=CORPUS,
             retriever=RETRIEVER, arm=ARM, rankings=rankings,
             targets=target_text, client=ScriptedClient(boom), model="fake",
             dry_run=False)

    fixed, cached = run_pair(
        pair, kind="rerank", results=tmp_path, corpus=CORPUS,
        retriever=RETRIEVER, arm=ARM, rankings=rankings, targets=target_text,
        client=ScriptedClient(score_first(0.7)), model="fake", dry_run=False,
        resume=True)
    assert not cached
    assert not retryable_columns(fixed)
    assert all(c["committed"] for c in fixed["trace"]["columns"])


def test_a_payload_of_unreadable_replies_is_still_readable(
        pair, rankings, target_text, tmp_path):
    """
    Refusing it would mean a model that formats badly can never be reported at
    all, and how often it does that is a result about the model.
    """
    from schemaoracle.resume import unparseable_columns

    # The paged arm, because `rerank` falls back to the retriever's order for
    # any column it could not read, the way Magneto does, so an unparseable
    # reply never survives into a rerank trace.
    run_pair(pair, kind="paged", results=tmp_path, corpus=CORPUS,
             retriever=RETRIEVER, arm=ARM, rankings=rankings,
             targets=target_text,
             client=ScriptedClient(lambda p: "nope, sorry"), model="fake",
             dry_run=False)
    path = cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, "paged", arm=ARM)

    payload = cache.load(path)                 # no PoisonedCache
    assert payload is not None
    assert unparseable_columns(payload)
    assert not payload["trace"]["completeness"]["complete"]


def test_a_resume_reprices_the_payload_it_spliced():
    """A splice recomputes totals and completeness from the new columns."""
    from schemaoracle.resume import merge_resumed

    def col(name, stop, n_blocks, tin):
        return {"source_column": name, "stop_reason": stop,
                "accepted_block": 0 if stop == "accepted" else None,
                "blocks": [{"tokens_in": tin, "tokens_out": 1,
                            "cache_write": 0, "cache_read": 0}
                           for _ in range(n_blocks)]}

    cfg = {"trace": {"metadata": {"config": {"window": 20}}}}
    old = {"trace": {"metadata": {"config": {"window": 20}},
                     "columns": [col("a", "accepted", 1, 10),
                                 col("b", "error", 1, 10)],
                     "totals": {"api_calls": 2, "tokens_in": 20},
                     "completeness": {"n_columns": 2, "complete": False}}}
    new_run = {"trace": {"metadata": {"config": {"window": 20}},
                         "columns": [col("b", "accepted", 4, 100)]}}

    merged = merge_resumed(old, new_run, ["b"])
    assert merged["trace"]["totals"]["api_calls"] == 5        # 1 + 4
    assert merged["trace"]["totals"]["tokens_in"] == 410      # 10 + 4*100
    assert merged["trace"]["completeness"]["n_columns"] == 2
    assert merged["trace"]["completeness"]["complete"]
    assert merged["trace"]["completeness"]["n_errored"] == 0



def test_a_gateway_run_resumes_on_the_vendors_own_api():
    """
    `record_as` files an OpenRouter run under the vendor's name, so the two
    transports share a file and must splice. The wire id carries the namespace
    (`anthropic/claude-sonnet-5`) and the direct one does not, which refused
    every such resume until `canonical_of` stripped it.
    """
    from schemaoracle.resume import canonical_of, check_compatible

    def payload(model):
        return {"trace": {"metadata": {"config": {"model": model, "window": 20}},
                          "columns": []}}

    gateway, direct = payload("anthropic/claude-sonnet-5"), payload("claude-sonnet-5")
    assert canonical_of(gateway) == canonical_of(direct) == "claude-sonnet-5"
    check_compatible(gateway, direct)        # must not raise
    check_compatible(direct, gateway)

    # A different checkpoint still refuses.
    with pytest.raises(Exception, match="canonical_model"):
        check_compatible(gateway, payload("claude-sonnet-4"))


def test_a_retired_setting_is_not_part_of_the_identity():
    """
    A field the running code no longer records must not be compared, in either
    direction. `prompt_style` is on 12,428 payloads and `decisions` on 13,441,
    and the code emits neither, so comparing one would refuse every one of them.
    """
    from schemaoracle.resume import IDENTITY_FIELDS, check_compatible, identity

    assert "prompt_style" not in IDENTITY_FIELDS
    assert "decisions" not in IDENTITY_FIELDS

    def payload(cfg):
        return {"trace": {"metadata": {"config": cfg}, "columns": []}}

    recorded = payload({"window": 20, "prompt_style": "figure5",
                        "decisions": True})
    now = payload({"window": 20})

    assert identity(recorded) == identity(now)
    check_compatible(recorded, now)          # must not raise
    check_compatible(now, recorded)


def test_a_payload_recording_a_retired_setting_still_resumes():
    """
    The other direction, and the one that bites: 12,428 runs on disk record
    `prompt_style: "figure5"` and the running code no longer emits it. A
    recorded value meeting `None` must not read as drift, or every resume of
    every existing run refuses after sending its calls.
    """
    from schemaoracle.resume import check_compatible

    def payload(cfg):
        return {"trace": {"metadata": {"config": cfg}, "columns": []}}

    on_disk = payload({"window": 20, "prompt_style": "figure5"})
    now = payload({"window": 20})

    check_compatible(on_disk, now)           # must not raise
    check_compatible(now, on_disk)


def test_turning_retries_on_resumes_an_existing_run_rather_than_refusing_it():
    """
    The reason the retry tag was removed: a run made without retries is the
    same measurement for every column that parsed, so asking for retries must
    splice into it, not demand a separate file.
    """
    from schemaoracle.resume import check_compatible

    def payload(n):
        return {"trace": {"metadata": {"config": {
            "window": 20, "prompt_style": "figure5", "max_parse_retries": n}},
            "columns": []}}

    check_compatible(payload(0), payload(3))     # must not raise
    check_compatible(payload(3), payload(0))


def test_a_run_answered_mostly_by_the_retriever_is_excluded(
        pair, rankings, target_text, tmp_path):
    """
    A rerank run answered mostly by the retriever is excluded from scoring.
    """
    run_pair(pair, kind="rerank", results=tmp_path, corpus=CORPUS,
             retriever=RETRIEVER, arm=ARM, rankings=rankings,
             targets=target_text,
             client=ScriptedClient(lambda p: "no scores here"), model="fake",
             dry_run=False)

    inv = inventory.scan(tmp_path, corpus=CORPUS)
    assert len(inv) == 1
    # Whole and error-free, which is exactly why nothing caught it before.
    payload = inventory.load(inv.iloc[0])
    assert not errored_columns(payload)
    assert len(payload["trace"]["columns"]) == pair["source_df"].shape[1]

    assert len(inventory.check(inv, verbose=False)) == 0
    # Kept on request, because how often a model formats badly is a result
    # about the model and someone may want to report it.
    kept = inventory.check(inv, verbose=False, max_fallback=1.0)
    assert len(kept) == 1
    assert kept.iloc[0]["n_fallback"] == pair["source_df"].shape[1]


def test_a_healthy_run_survives_the_fallback_check_and_counts_zero(
        pair, rankings, target_text, tmp_path):
    do(pair, rankings, target_text, tmp_path, "rerank")
    inv = inventory.check(inventory.scan(tmp_path, corpus=CORPUS),
                          verbose=False)
    assert len(inv) == 1
    assert inv.iloc[0]["n_fallback"] == 0


def test_the_fallback_share_is_over_asked_columns_not_every_column(tmp_path):
    """
    The fallback share counts asked columns only, so pruning cannot dilute it.
    """
    path = cache.run_path(tmp_path, CORPUS, "Cao", RETRIEVER, "rerank",
                          arm=ARM, top_k=20)
    columns = [{"source_column": f"c{i}", "stop_reason": "pruned",
                "committed": False, "abstained": True, "ranking": []}
               for i in range(18)]
    columns += [
        {"source_column": "asked_ok", "stop_reason": "accepted",
         "committed": True, "abstained": False, "ranking": [["t", 0.9]]},
        {"source_column": "asked_fb", "stop_reason": "rerank_fallback",
         "committed": True, "abstained": False, "ranking": [["t", 0.4]]},
    ]
    path.write_text(json.dumps({"trace": {
        "columns": columns, "totals": {"api_calls": 2},
        "metadata": {"corpus": CORPUS, "pair": "Cao", "arm": ARM,
                     "retriever": RETRIEVER}}}), encoding="utf-8")

    inv = inventory.scan(tmp_path, corpus=CORPUS)
    assert len(inv) == 1
    assert len(inventory.check(inv, verbose=False)) == 0      # 1 of 2 asked
    assert len(inventory.check(inv, verbose=False, max_fallback=0.6)) == 1


def test_a_corpus_name_with_a_slash_finds_its_own_runs(pair, rankings,
                                                       target_text, tmp_path):
    """
    `run_path` sanitises the whole stem, so `valentine/opendata` is written as
    `valentine-opendata`. Comparing the raw string reported an empty inventory
    for a corpus with 400 runs in it, which reads as "nothing has been done".
    """
    run_pair(pair, kind="rerank", results=tmp_path, corpus="valentine/opendata",
             retriever=RETRIEVER, arm=ARM, rankings=rankings,
             targets=target_text, client=ScriptedClient(score_first(0.6)),
             model="fake", dry_run=False)

    inv = inventory.scan(tmp_path, corpus="valentine/opendata",
                         kinds=("rerank",))
    assert len(inv) == 1
    # And the sanitised form still works, since that is what is on disk.
    assert len(inventory.scan(tmp_path, corpus="valentine-opendata",
                              kinds=("rerank",))) == 1


# The prune tag

# Every combination, generated: `cache._PRUNE` is a closed alternation, so a
# rule added to `POLICIES` without being spelled there would parse as no prune
# at all and serve a filtered run back to an unfiltered configuration.
PRUNES = [None] + [c for n in range(1, len(pruning.POLICIES) + 1)
                   for c in combinations(pruning.POLICIES, n)]


@pytest.mark.parametrize("kind", ["rerank", "gated", "paged"])
@pytest.mark.parametrize("prune", PRUNES)
def test_a_prune_round_trips_through_the_filename(kind, prune):
    from schemaoracle.pruning import tag

    path = cache.run_path("r", CORPUS, "Cao", RETRIEVER, kind, arm=ARM,
                          window=20, threshold=0.25, top_k=20, prune=prune)
    got = cache.parse_stem(path.stem)
    assert got is not None, path.stem
    assert got["kind"] == kind and got["arm"] == ARM
    assert got["prune"] == tag(prune)


@pytest.mark.parametrize("kind", ["retrieval", "rerank", "gated", "paged"])
def test_no_prune_writes_no_tag_so_existing_runs_keep_their_names(kind):
    """
    Same precedent as `rep`, and the reason 4,215 files on disk did not have to
    be renamed when the tag was added.
    """
    plain = cache.run_path("r", CORPUS, "Cao", RETRIEVER, kind, arm=ARM,
                           window=20, threshold=0.0, top_k=20)
    tagged = cache.run_path("r", CORPUS, "Cao", RETRIEVER, kind, arm=ARM,
                            window=20, threshold=0.0, top_k=20, prune=None)
    assert plain.name == tagged.name
    assert "__p" not in plain.name.replace("__paged", "")


@pytest.mark.parametrize("variant", ["prefix_cached", "prefix-cached",
                                     "pzero", "pmutual", "pzerosrc",
                                     "tgt-match-retriever",
                                     "unp-fallback"])
def test_an_arm_variant_is_never_mistaken_for_a_prune_tag(variant):
    """
    The test that rejected the obvious design. With the tag to the right of the
    arm, `...__ollama__model__prefix-cached__rerank` parsed as
    prune="refix-cached", and a variant is exactly where new names get added.
    """
    arm = f"ollama__model__{variant}"
    stem = cache.run_path("r", CORPUS, "Cao", RETRIEVER, "rerank", arm=arm,
                          top_k=20).stem
    got = cache.parse_stem(stem)
    assert got["arm"] == arm
    assert got["prune"] == ""


def test_a_pruned_run_is_its_own_configuration(tmp_path, pair, rankings,
                                               target_text):
    """
    A prune changes which columns are asked and which candidates each prompt
    carries, so it cannot share a filename with the unpruned run or one gets
    served the other's answers.
    """
    kw = dict(results=tmp_path, corpus=CORPUS, retriever=RETRIEVER,
              rankings=rankings, targets=target_text, kind="rerank", arm=ARM,
              model="fake", dry_run=False, top_k=20)
    run_pair(pair, client=ScriptedClient(score_first(0.6)), **kw)
    run_pair(pair, client=ScriptedClient(score_first(0.6)),
                 prune="mutual", **kw)
    names = sorted(p.name for p in tmp_path.glob("*__rerank.json"))
    assert len(names) == 2, names
    assert any("__pmutual__" in n for n in names)


def test_the_prune_survives_the_inventory_reindex():
    """
    `scan` reindexes onto COLUMNS, so a field left out of that list is dropped
    and every pruned run then reads as unpruned. Invisible without this.
    """
    assert "prune" in inventory.COLUMNS
    for kind in ("rerank", "gated", "paged"):
        assert "prune" in inventory.KEYS[kind]


def test_a_pruned_and_an_unpruned_run_are_two_rows_of_the_inventory(
        tmp_path, pair, rankings, target_text):
    kw = dict(results=tmp_path, corpus=CORPUS, retriever=RETRIEVER,
              rankings=rankings, targets=target_text, kind="rerank", arm=ARM,
              model="fake", dry_run=False, top_k=20)
    run_pair(pair, client=ScriptedClient(score_first(0.6)), **kw)
    run_pair(pair, client=ScriptedClient(score_first(0.6)),
                 prune="mutual", **kw)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=("rerank",))
    assert sorted(inv["prune"]) == ["", "mutual"]
    assert len(inv.groupby(inventory.KEYS["rerank"])) == 2


@pytest.mark.parametrize("kind", ["rerank", "gated", "paged"])
def test_a_pruned_sweep_runs_and_files_itself_apart(kind, tmp_path, pair,
                                                    rankings, target_text):
    """
    The whole point of the prune living in Bench: a sweep can be run with it
    and the runs do not collide with the unpruned ones.
    """
    kw = dict(results=tmp_path, corpus=CORPUS, retriever=RETRIEVER,
              rankings=rankings, targets=target_text, kind=kind, arm=ARM,
              model="fake", dry_run=False, top_k=20, window=20)
    for prune in (None, ("mutual", "zero")):
        run_pair(pair, client=ScriptedClient(score_first(0.6)), prune=prune,
                 **kw)
    inv = inventory.scan(tmp_path, corpus=CORPUS, kinds=(kind,))
    assert sorted(inv["prune"]) == ["", "mutual-zero"]

    # The pruned run records what it dropped, and pruned columns cost nothing.
    pruned = inventory.load(inv[inv["prune"] == "mutual-zero"].iloc[0])
    meta = pruned["trace"]["metadata"]["prune"]
    assert meta["rules"] == ["mutual", "zero"]
    for col in pruned["trace"]["columns"]:
        if col["stop_reason"] == "pruned":
            assert col["api_calls"] == 0 and not col["committed"]


def test_a_pruned_run_is_never_served_an_unpruned_cache(tmp_path, pair,
                                                        rankings, target_text):
    """
    The reason the prune is in the filename at all. Without it the second call
    would find the first on disk and hand back answers to a different question.
    """
    kw = dict(results=tmp_path, corpus=CORPUS, retriever=RETRIEVER,
              rankings=rankings, targets=target_text, kind="rerank", arm=ARM,
              model="fake", dry_run=False, top_k=20)
    _, cached_a = run_pair(pair, client=ScriptedClient(score_first(0.6)), **kw)
    _, cached_b = run_pair(pair, client=ScriptedClient(score_first(0.6)),
                           prune="mutual", **kw)
    assert not cached_a and not cached_b
    # And a repeat of the pruned configuration *is* served its own cache.
    _, cached_c = run_pair(pair, client=ScriptedClient(score_first(0.6)),
                           prune="mutual", **kw)
    assert cached_c


# A stored summary that no longer matches its columns

def _stale(payload):
    """Corrupt the stored summary the way a pre-fix splice left it."""
    payload["trace"]["completeness"] = {
        **payload["trace"]["completeness"],
        "n_errored": 3, "n_committed": 0, "complete": False,
    }
    payload["trace"]["totals"] = {**payload["trace"]["totals"],
                                  "api_calls": 1}
    return payload


def test_the_run_row_is_derived_from_the_columns_not_the_stored_summary(
        pair, rankings, target_text, tmp_path):
    """The sweep row is recomputed from the columns, not the stored summary."""
    payload, _ = do(pair, rankings, target_text, tmp_path, "rerank")
    honest = run.summarise("Cao", "rerank", payload, cached=True, wall_s=0.0)
    lied = run.summarise("Cao", "rerank", _stale(payload), cached=True,
                         wall_s=0.0)

    assert honest["errored"] == 0
    assert lied["errored"] == 0                      # not the stored 3
    assert lied["committed"] == honest["committed"]  # not the stored 0
    assert lied["complete"] is True
    assert lied["calls"] == honest["calls"] > 1      # not the stored 1


def test_a_stale_summary_does_not_make_a_finished_pair_look_resumable(
        pair, rankings, target_text, tmp_path):
    """
    The columns decide what is left to do: a stale error in the summary sends
    nothing, and a real one is still caught.
    """
    payload, _ = do(pair, rankings, target_text, tmp_path, "rerank")
    assert run._to_redo(_stale(payload), rankings) == []

    payload["trace"]["columns"][0]["stop_reason"] = "error"
    assert run._to_redo(payload, rankings) == [
        payload["trace"]["columns"][0]["source_column"]]


def test_the_models_reply_is_kept_on_every_block(tmp_path, monkeypatch):
    """
    Without the raw reply a parser bug can only be repaired by asking every
    affected column again. Two have happened; both cost a full rerun.
    """
    from schemaoracle.bench import Bench
    from test_bench import QWEN

    ranking = [(f"t_{i}", 1.0 - i / 100) for i in range(40)]
    source = pd.DataFrame({"age": [1, 2], "name": ["a", "b"]})
    target_df = pd.DataFrame({f"t_{i}": [1] for i in range(40)})
    pairs = [{"name": "Cao", "source_df": source, "target_df": target_df,
              "ground_truth": [("age", "t_0")]}]
    b = Bench(corpus="gdc", pairs=pairs, results=tmp_path,
              retrievers={"qwen": QWEN},
              models=[dict(provider="anthropic", model="claude-sonnet-5",
                           client=dict(think=False))])
    monkeypatch.setattr(Bench, "orders", lambda self, rebuild=False: {
        ("qwen", p["name"]): {c: ranking for c in p["source_df"].columns}
        for p in self.pairs})
    monkeypatch.setattr(Bench, "client",
                        lambda self, m: ScriptedClient(score_first(0.9)))
    b.dry_run = False
    b.run("rerank", levels=[20])

    payload = json.loads(next(tmp_path.glob("*__rerank.json"))
                         .read_text(encoding="utf-8"))
    blocks = [blk for c in payload["trace"]["columns"] for blk in c["blocks"]]
    assert blocks and all(blk["raw_response"] for blk in blocks)
    # And the settings that change the answer but not the filename.
    assert payload["trace"]["metadata"]["config"]["client"] == {"think": False}
