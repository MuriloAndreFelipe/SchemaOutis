# SchemaOracle

Code for a study of paged reranking in schema matching.

A reranker shows a model the retriever's top *k* candidates and takes whatever
it scores highest. The retriever's recall@k is then a hard ceiling: if the
correct target sits at cosine rank 34 and *k* is 20, no model can recover it,
and the miss looks like a reranking failure. Paging turns the shortlist into a
sequence of windows (ranks 0-19, 20-39, and so on down the whole schema). The
model either accepts a window or declines it and is shown the next one.

In the terms of Papadakis et al. (ACM CSUR 53(2), 2020), the retriever does
block building and the model's accept-or-decline decision is block cleaning.

## The four arms

| arm | LLM calls per column | may decline | bounded by |
|-----|----------------------|-------------|------------|
| `retrieval` | 0 | no | it is the bound |
| `rerank` | 1 | no | recall@`top_k` |
| `gated` | 1 | yes | recall@`top_k`, the same block as `rerank` |
| `paged` | 1 to `ceil(n_targets / window)` | yes | nothing; the walk can reach the end |

`rerank` reproduces Magneto's LLM reranker: its published prompt, no way to
decline, and the retriever's scores kept when a reply cannot be parsed.

`rerank` and `paged` differ both in how much the model sees and in whether it
may refuse, so `gated` sits between them: `rerank` against `gated` measures
declining at fixed depth, and `gated` against `paged` measures depth.

All four arms read the same retrieval order, so a difference between them comes
only from what the model was shown and whether it could refuse.

## Why declining matters

On GDC, 404 of the 569 source columns have no correct target. `rerank` must
answer every one, so each is a false positive; that caps precision near 0.25
and F1 near 0.40 whatever the model. Precision is therefore reported twice:
*as committed* (declined columns leave the denominator) and *as evaluated*
(every column contributes its rank-1 target). For an arm that cannot decline
the two are equal; the gap between them is what declining buys.

| arm | calls per column | recall@1 | recall@20 | MRR | precision (committed) | F1 (committed) |
|-----|-----:|-----:|-----:|-----:|-----:|-----:|
| `retrieval` | 0 | 0.456 | 0.927 | 0.812 | | |
| `rerank` | 1.0 | 0.556 | 0.927 | 0.901 | 0.253 | 0.392 |
| `gated` | 1.0 | 0.560 | 0.923 | 0.902 | 0.394 | 0.544 |
| `paged` | 19.2 | 0.541 | 0.903 | 0.881 | 0.493 | 0.624 |

GDC, pooled over the ten studies (569 source columns, 259 gold pairs), Qwen3
embedding order, DeepSeek V4 Flash, no filter, first repeat. Results for every
model, filter and retrieval order are in `notebooks/paper_figures.ipynb`.

Results are pooled across studies, never averaged: averaging per-study scores
would weight Clark's 16 columns like Dou's 179.

## Use

```python
from schemaoracle import SchemaOracle

oracle = SchemaOracle(arm="paged", client=client, llm_model="claude-sonnet-5")
matches = oracle.get_matches(source_df, target_df)   # Valentine format

oracle.trace.completeness()   # is every column answered?
oracle.trace.totals()         # calls and tokens
oracle.retrieval_order        # the complete order the walk followed
```

`get_matches` returns the same shape as `magneto.Magneto.get_matches`, so a
Valentine harness can drive either.

Two threshold-free filters reduce the work:

```python
oracle = SchemaOracle(arm="gated", prune=("mutual", "zero"), client=client,
                      llm_model="claude-sonnet-5")
```

`mutual` skips a source column whose best target does not return the
preference (41% of the calls on GDC); `zero` drops a target no source is closer
to than its own neighbourhood is. `oracle.prune_plan` records what was dropped.

## Layout

`schemaoracle` produces runs and `evaluation` scores them; `schemaoracle`
imports neither `evaluation` nor `corpora` (`tests/test_layout.py`).

```
schemaoracle/     the matcher and the runs it writes
  matcher.py      the four arms, and get_matches
  paging.py       the traversal: windows, accept or decline, when to stop
  prompts.py      templates and reply parsing
  llm.py          Anthropic, OpenRouter and Ollama clients
  retrieval.py    the biencoder that produces the shared order
  serializers.py  column sampling and serialisation
  vectors.py      the embedding cache
  pruning.py      the mutual and zero filters
  trace.py        what a run records
  resume.py       splicing an interrupted run back together
  cache.py        run filenames, and refusing a run with API errors
  run.py          one pair of one arm
  bench.py        a notebook's configuration
  inventory.py    the runs on disk, as one table
  repair.py       reading unparseable columns as retriever fallbacks
  reseed.py       re-asking only the columns a retry could change

evaluation/
  matching.py     recall@k, MRR, precision and recall with declining as silence
  blocking.py     Pair Completeness and Reduction Ratio
  scoring.py      runs into the numbers the paper reports
  competitors.py  Magneto and Valentine's matchers as baselines
  ablation.py     the retrieval sweep
  mixture.py      the score-distribution filter (a negative result)
  prune_signals.py, prune_scoring.py
                  the search for the filters, and scoring them

corpora/          loading GDC and Valentine

notebooks/
  run_arms          rerank, gated and paged over a list of filters (spends money)
  run_competitors   the published baselines (no API calls)
  ablation          where the gold sits across retrieval configurations (GPU)
  paper_figures     every figure in the paper, from finished runs (no API calls)
```

## Install

```bash
pip install -e ".[llm,retrieval,figures,dev]"
pip install -e ".[baselines]"     # for run_competitors
```

Reading cached runs needs neither `retrieval` (torch, sentence-transformers)
nor `llm`. Magneto's bipartite reranker in magneto-python 0.2.0 fails against
valentine 1.0.0, so `run_competitors` also offers `magneto-no-bp`, Magneto
without that stage.

## Data

```
data/gdc/{target-tables,source-tables,ground-truth}/
data/Valentine-datasets/<subset>/.../<pair>_{source,target}.csv + _mapping.json
```

`corpora.load("gdc")` reads `data/`; set `SCHEMAORACLE_DATA` to use another
location. Runs are cached under `results/`. Both directories are gitignored.

## Run files

A run's filename holds everything that changes what the model was asked:

```
{corpus}__{pair}__{retriever}__retrieval.json
{corpus}__{pair}__{retriever}__k{top_k}[__p{prune}]__{arm}[__r{rep}]__rerank.json
{corpus}__{pair}__{retriever}__k{top_k}[__p{prune}]__{arm}[__r{rep}]__gated.json
{corpus}__{pair}__{retriever}__w{window}__th{threshold}[__p{prune}]__{arm}[__r{rep}]__paged.json
```

`arm` is provider, model and prompt variant. The retrieval order has no model
in its key, so one embedding pass serves every arm.

## Design choices

- **An unreadable reply is not a rejection.** The walk stops there rather than
  moving on, which could skip the gold while reporting a clean traversal.
- **Declining is an answer.** A column the model saw and declined counts as
  answered; on a corpus where most columns have no match it is usually right.
- **A run with API errors is not served from the cache.** Such a run keeps
  going and writes a file that looks complete. Loading it raises, and
  `resume=True` re-runs only the failed columns.
- **Tables never mix retrieval orders.** Orders are compared cell by cell,
  using only cells complete under both.

## Provenance

`llm.py`, `retrieval.py`, `vectors.py`, `serializers.py` and the prompts and
parsers in `prompts.py` come from the earlier SchemaSteer study unchanged, so
results remain comparable. `tests/test_prompts.py` pins every prompt template
against a snapshot of the text the cached runs were produced with.

SchemaSteer's runs were converted once into this repository's format (a rename
that recomputed nothing) and are what `results/runs/` holds. As a check, the
block confusion matrix for `claude-sonnet-5__block_cached` at threshold 0.00
(an arm whose prompt layout has since been removed, so those 50 runs are
readable but no longer reproducible)
over all ten GDC studies matches SchemaSteer's: magneto order `tp=148, fp=342,
fn=1, tn=3733`; qwen order `tp=157, fp=322, fn=0, tn=3624`. The original files
in `results/meta_blocking/` and `results/rerank_sweep/` are kept as archives
and are not read.

## Tests

```bash
python -m pytest tests -q
```

The tests need no corpus, network or GPU; the LLM backend is a fake.
