"""
SchemaOutis: four ways to answer one schema-matching question.

Every arm reads the same retrieval order and differs only in how much of it the
model may see and whether it may decline.

    retrieval   no LLM: the biencoder's cosine order
    rerank      one call over the top k (Magneto's reranker); cannot decline
    gated       the same block, with a final NONE
    paged       one call per window, walking the whole order

    from schemaoutis import SchemaOutis
    matches = SchemaOutis(arm="paged", client=client).get_matches(src, tgt)

`get_matches` returns Valentine format, as `magneto.Magneto` does.

    matcher      the arms and the entry point
    paging       the traversal: windows, accept or decline, when to stop
    prompts      templates and reply parsing
    llm          chat backends: Anthropic, OpenRouter, Ollama
    retrieval    the biencoder; serializers and vectors support it
    pruning      which columns and candidates to skip, from the order alone
    trace        what a run records
    resume       splicing an interrupted run back together
    cache, run, bench, inventory
                 running pairs, caching them, and reading them back

`schemaoutis` imports neither `corpora` nor `evaluation`
(`tests/test_layout.py`).
"""

__version__ = "0.2.0"

from .matcher import (
    ARM_SETTINGS,
    ARMS,
    SchemaOutis,
    convert_to_valentine_format,
    get_top_k_matches,
    run_arm,
    run_retrieval,
)
from .paging import DEFAULT_WINDOW, Pager, block_spans, run_paging
from .trace import BlockRecord, ColumnRecord, RunTrace

__all__ = [
    "SchemaOutis", "ARMS", "ARM_SETTINGS", "run_arm", "run_retrieval",
    "convert_to_valentine_format", "get_top_k_matches",
    "Pager", "run_paging", "block_spans", "DEFAULT_WINDOW",
    "RunTrace", "ColumnRecord", "BlockRecord",
]
