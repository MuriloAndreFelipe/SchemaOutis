"""Prompts, serialisation and reply parsing.

One published style, Figure 5 of Liu et al., Magneto, VLDB 2025:
`Name(0.95); Other(0.30)`. Three layouts:

  oneshot       the published prompt: no block header, no way to decline (rerank).
  oneshot_none  one block, plus a final NONE (gated).
  standard      adds "block n of m" and NONE as a request for the next block (paged).

The layout is recorded in the trace.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pandas as pd

from .serializers import get_samples

# Magneto's system message, verbatim from `LLMReranker._get_matches`.
MAGNETO_SYSTEM = (
    "You are an AI trained to perform schema matching by providing column "
    "similarity scores."
)

# Appended to the system turn when an unreadable reply is asked again. In the
# system turn because placed inside the few-shot prompt it had no effect.
PARSE_RETRY_SYSTEM = (
    " Each target column is written as `name type values`. Reply with the "
    "column NAME ONLY, without the type or the values, as `name(score);` pairs."
)

PROMPT_LAYOUTS = ("oneshot", "oneshot_none", "standard")

# Magneto's reranker uses these whatever the retriever sampled with.
MAGNETO_RERANK_SAMPLING_MODE = "priority_sampling"
MAGNETO_RERANK_SAMPLING_SIZE = 10

# What to do with a block whose reply could not be parsed:
#   stop      end the walk with no answer
#   fallback  end the walk and keep the retriever's window, as rerank does
UNPARSEABLE_POLICIES = ("stop", "fallback")

VERDICT_ACCEPT = "accept"
VERDICT_REJECT = "reject"
VERDICT_UNPARSEABLE = "unparseable"
VERDICT_ERROR = "error"


def serialize_for_prompt(name: str, series: pd.Series, n: int, mode: str,
                         seed: int) -> str:
    """Magneto's form: Column: <name>, Sample values: [v1,v2,...]."""
    samples = get_samples(series, n=n, mode=mode, seed=seed)
    return f"Column: {name}, Sample values: [{','.join(samples)}]"


# The templates are composed from shared parts. `tests/test_prompts.py` pins
# every rendered template against a golden fixture.

_F5_INSTRUCTIONS = (
    "From a score of 0.00 to 1.00, judge the similarity of the candidate column "
    "in the source table to each target column in the target table. Each column "
    "is represented by its name and a sample of its respective values, if "
    "available.\n\n"
)


# The paging clause: these are not all the candidates, and NONE asks for more.
_F5_NOTE_NUMBERED = (
    "The target columns below are block {block} of {n_blocks} from a larger "
    "schema, ordered by an upstream retriever. They are not guaranteed to "
    "contain a match. If none of them is a plausible match for the candidate "
    "column, reply NONE and you will be shown the next block; do not settle for "
    "the least bad option in this block.\n\n"
)

# gated: there is no next block, so NONE is final rather than a request for more.
_F5_NOTE_TERMINAL = (
    "The target columns below are the closest candidates an upstream retriever "
    "found, and they are not guaranteed to contain a match. They are the only "
    "candidates you will be shown; there is no next block. If none of them is a "
    "plausible match for the candidate column, reply NONE; do not settle for "
    "the least bad option.\n\n"
)

_F5_EX_CAND = "Column: EmpID, Sample values: [100, 101, 102]\n"
_F5_EX_TARGETS = (
    "Column: WorkerID, Sample values: [100, 101, 102]\n"
    "Column: EmpCode, Sample values: [00A, 00B, 00C]\n"
    'Column: StaffName, Sample values: ["Alice", "Bob", "Charlie"]\n'
)
_F5_EX_ANSWER = "Response: WorkerID(0.95); EmpCode(0.30); StaffName(0.05)\n\n"
_F5_EX_NONE_TARGETS = (
    "Column: RainfallMM, Sample values: [12.4, 0.0, 8.1]\n"
    'Column: SoilType, Sample values: ["clay", "loam", "sand"]\n'
)


_F5_FORMAT = (
    "Provide the name of each target column followed by its similarity score in "
    "parentheses, formatted to two decimals, and separated by semicolons. Rank "
    "the column-score pairs in descending order. "
)
_F5_FORMAT_NONE = (
    "If no target column in {block_word} block is a plausible match, respond "
    "with exactly NONE. "
)
_F5_FORMAT_TAIL = "Exclude additional information and quotations.\n\n"



def _example(header: str, targets: str) -> str:
    """One worked example: the candidate, then the targets."""
    return (header + "Candidate Column:\n" + _F5_EX_CAND
            + "Target Schemas:\n" + targets)


def _compose(*, note: str, block_word: str = "") -> str:
    """Assemble one template. An empty `block_word` means no NONE clause."""
    parts = [_F5_INSTRUCTIONS, note,
             _example("Example:\n", _F5_EX_TARGETS), _F5_EX_ANSWER]
    if block_word:
        parts += [_example("Example with no plausible match:\n",
                           _F5_EX_NONE_TARGETS),
                  "Response: NONE\n\n"]
    parts.append(_F5_FORMAT)
    if block_word:
        parts.append(_F5_FORMAT_NONE.format(block_word=block_word))
    parts += [_F5_FORMAT_TAIL, "Candidate Column:\n{cand}\n\n",
              "Target Schemas:\n{targets}\n\n", "Response:"]
    return "".join(parts)


# layout -> (the paging clause, the word the NONE clause uses). An empty word
# means the layout offers no way to decline.
LAYOUTS = {
    "oneshot":      ("", ""),
    "oneshot_none": (_F5_NOTE_TERMINAL, "this"),
    "standard":     (_F5_NOTE_NUMBERED, "this"),
}

TEMPLATES = {layout: _compose(note=note, block_word=word)
             for layout, (note, word) in LAYOUTS.items()}

# The published prompt, by the name the paper uses.
MAGNETO_PROMPT = TEMPLATES["oneshot"]


def build_prompt(cand, targets, block=0, n_blocks=1, layout="standard"):
    """One prompt. `block` is 0-based; the text says `block + 1`."""
    if layout not in TEMPLATES:
        raise ValueError(
            f"no template for layout={layout!r}; layouts {PROMPT_LAYOUTS}")
    return TEMPLATES[layout].format(
        cand=cand, targets="\n".join(targets),
        block=block + 1, n_blocks=n_blocks,
    )



# `Name(0.95)`: the score is the last parenthesised number before a semicolon
# or the end of the line, so a name may itself hold parentheses, as in
# OpenData's `Biodiversity(marker)(0.95)`. Scores outside [0, 1] are rejected
# by the caller.
_PAIR_RE = re.compile(
    r"([^;\n]+?)\s*\(\s*([0-9]*\.?[0-9]+)\s*\)(?=\s*(?:;|\n|$))")


def _norm(name: str) -> str:
    # All punctuation, not only separators: the prompt shows
    # `miller2_Biodiversity(marker)` as `miller2 biodiversity marker`, and the
    # model answers with what it was shown.
    return re.sub(r"[\W_]+", "", str(name).lower())


def _clean_name(name: str) -> str:
    name = name.strip().strip('"\'`').strip()
    # The model sometimes echoes the serialisation prefix.
    if name.lower().startswith("column:"):
        name = name.split(":", 1)[1].strip()
    return name


def _echoed(key: str, valid: Mapping[str, str],
            texts: Mapping[str, str] | None) -> str | None:
    """
    The target whose prompt line a reply name echoes, as in `alcohol drinks per
    day unknown` for `alcohol_drinks_per_day`. The echo must start with the full
    name and be a prefix of that target's own line, so a longer name that only
    shares a prefix is not taken for it.
    """
    if not texts:
        return None
    hits = [real for n, real in valid.items()
            if n and key.startswith(n)
            and _norm(texts.get(real, "")).startswith(key)]
    return max(hits, key=lambda r: len(_norm(r)), default=None)


# Below this a dropped-prefix match is too weak to trust, whatever the guard:
# a three-character tail names nothing on its own.
_MIN_TAIL = 4


def _unprefixed(key: str, valid: Mapping[str, str]) -> str | None:
    """
    The target whose name ends with `key`, for a reply that dropped the table
    prefix every candidate shares: Valentine blocks are all `miller2_...`, and
    the model answers `country region percent` for
    `miller2_CountryRegionPercent`. Only an unambiguous tail counts, so a bare
    `percent` against four `...Percent` targets stays unmatched.
    """
    if len(key) < _MIN_TAIL:
        return None
    hits = [real for n, real in valid.items() if n != key and n.endswith(key)]
    return hits[0] if len(hits) == 1 else None


def _collect(candidates, valid: Mapping[str, str], count_unmatched: bool,
             texts: Mapping[str, str] | None = None):
    """
    Keep `(name, score)` guesses that name a real target. `valid` maps a
    normalised name to the real one; `texts` maps the real name to the line
    the prompt showed for it. Names matching no target are counted.
    """
    pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    unmatched = 0
    for raw_name, raw_score in candidates:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not 0.0 <= score <= 1.0:
            continue
        key = _norm(_clean_name(str(raw_name)))
        real = (valid.get(key) or _echoed(key, valid, texts)
                or _unprefixed(key, valid))
        if real is None:
            unmatched += count_unmatched
            continue
        if real in seen:
            continue
        seen.add(real)
        pairs.append((real, score))
    return pairs, unmatched


def parse_scores(text: str, valid: Mapping[str, str],
                 texts: Mapping[str, str] | None = None,
                 ) -> tuple[list[tuple[str, float]], int]:
    """Parse a Figure-5 reply into ``[(target, score), ...]``."""
    return _collect(_PAIR_RE.findall(text or ""), valid, count_unmatched=True,
                    texts=texts)


# handles reasoning models whose <think> block arrives inside the reply.

_THINK_TAGS = ("think", "thinking", "reasoning", "reflection", "scratchpad",
               "analysis")
_THINK_BLOCK_RE = re.compile(
    r"<(" + "|".join(_THINK_TAGS) + r")\b[^>]*>.*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# An opening tag with no close: generation stopped inside the reasoning.
_THINK_OPEN_RE = re.compile(
    r"<(?:" + "|".join(_THINK_TAGS) + r")\b[^>]*>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+\-]*")

# Name: 0.95, Name = 0.95, Name - 0.95, Name -> 0.95, Name | 0.95, Name (0.95).
# Names are checked against the target list afterwards.
_LOOSE_PAIR_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_. ()\-]{0,60}?)\s*"
    r"(?:[:=|]|->|→|-|\()\s*"
    r"(0?\.\d+|[01](?:\.\d+)?)\s*\)?"
)
# List decoration before a name: "1. ", "2) ", "- ", "* ".
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")


def strip_reasoning(text: str) -> tuple[str, bool]:
    """
    Remove reasoning blocks and code fences. Returns (answer, truncated), where
    `truncated` means a reasoning tag was never closed and no answer followed.
    """
    body = _THINK_BLOCK_RE.sub(" ", text or "")

    truncated = False
    if match := _THINK_OPEN_RE.search(body):
        truncated = True
        body = body[: match.start()]

    return _FENCE_RE.sub(" ", body).strip(), truncated


def _after_last_response_marker(text: str) -> str:
    """
    Keep what follows the last `Response:`. The worked examples contain the
    marker too, so the last one is the model's own answer.
    """
    idx = text.lower().rfind("response:")
    if idx == -1:
        return text
    tail = text[idx + len("response:"):]
    # A trailing bare marker means the answer came before it.
    return tail if tail.strip() else text


def parse_loose(text: str, valid: Mapping[str, str],
                texts: Mapping[str, str] | None = None,
                ) -> tuple[list[tuple[str, float]], int]:
    """
    Last resort: any known target name followed by a number in [0, 1], with
    any separator. Unknown names are not counted as hallucinations here.
    """
    candidates = [(_LIST_MARKER_RE.sub("", n).strip('"\'`* '), s)
                  for n, s in _LOOSE_PAIR_RE.findall(text or "")]
    return _collect(candidates, valid, count_unmatched=False, texts=texts)


# Outcomes of `parse_response`, as recorded in the trace.
PARSE_OK = "parsed"
PARSE_TRUNCATED = "truncated_in_reasoning"
PARSE_FAILED = "unparseable"
# Empty reply because the model spent the turn in its thinking channel. The
# caller detects this from the response object, not the text.
PARSE_ONLY_THINKING = "answered_in_thinking"

# The few-shot prompt invites a model to keep going after its answer and write
# new examples. These cut it off where a new problem would start.
RERANK_STOP_SEQUENCES = (
    "\nCandidate Column:",
    "\nTarget Schemas:",
    "\nExample:",
    "\nExample with",
)


def parse_response(
    text: str,
    valid: Mapping[str, str],
    tolerant: bool = True,
    texts: Mapping[str, str] | None = None,
) -> tuple[list[tuple[str, float]], int, str]:
    """
    Parse a reranking reply. Returns (pairs, n_hallucinated, note).

    Tries the published format first; with `tolerant`, then the loose parser.
    The note records which one succeeded. `texts`, the line shown for each
    target, lets a name echoed with its type and values count.
    """
    body, truncated = strip_reasoning(text)
    if not body:
        return [], 0, PARSE_TRUNCATED if truncated else PARSE_FAILED

    body = _after_last_response_marker(body)

    attempts = [("figure5", parse_scores)]
    if tolerant:
        attempts.append(("loose", parse_loose))

    for i, (name, parser) in enumerate(attempts):
        pairs, unmatched = parser(body, valid, texts)
        if pairs:
            return pairs, unmatched, PARSE_OK if i == 0 else f"recovered_as_{name}"

    return [], 0, PARSE_TRUNCATED if truncated else PARSE_FAILED


# A reply that is only a "none"-like word, with any decoration.
_BARE_NONE_RE = re.compile(
    r"^\W*(?:response\s*:)?\W*(none|no\s+match|n/?a)\b[\s.!]*$", re.IGNORECASE)
_NONE_WORD_RE = re.compile(r"\bnone\b|\bno\s+match\b", re.IGNORECASE)


def says_none(text: str) -> bool:
    """
    Whether a reply explicitly says nothing matches, as opposed to being
    unreadable. "NONE", "Response: NONE" and "No match." count; a reply that
    also scores candidates does not.
    """
    # Reasoning is not an answer, so it is removed first.
    body, _ = strip_reasoning(text)
    if not body:
        return False
    if _BARE_NONE_RE.match(body):
        return True
    # Only reached when parsing found no scores.
    return bool(_NONE_WORD_RE.search(body)) and len(body) <= 200


def _text_of(response: Any) -> str:
    """The text blocks of an Anthropic-shaped response, concatenated."""
    out: list[str] = []
    for block in getattr(response, "content", None) or []:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                out.append(block.get("text", ""))
        elif getattr(block, "type", None) == "text":
            out.append(getattr(block, "text", ""))
    return "".join(out)
