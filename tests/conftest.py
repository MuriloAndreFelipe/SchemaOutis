"""Shared fixtures: a fake backend and a synthetic corpus."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeUsage:
    def __init__(self, tin=10, tout=5):
        self.input_tokens = tin
        self.output_tokens = tout
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class FakeResponse:
    def __init__(self, text, thinking=""):
        self.content = [{"type": "text", "text": text}]
        self.usage = FakeUsage()
        # A reasoning backend reports this apart from the content, and an empty
        # content with a full thinking field is the failure that reads as
        # unparseable unless something looks here.
        self.thinking = thinking


class ScriptedClient:
    """
    Replies with `reply(prompt)`. To get an acceptance, name a target that is
    in the block; any other name reads as a hallucination.
    """

    def __init__(self, reply):
        self._reply = reply
        self.messages = self
        self.calls = []

    def create(self, **kw):
        content = kw["messages"][0]["content"]
        prompt = (content if isinstance(content, str)
                  else "".join(b["text"] for b in content))
        self.calls.append(kw)
        reply = self._reply(prompt)
        # A reply may be given as (content, thinking) to script a backend that
        # answered in its reasoning channel.
        if isinstance(reply, tuple):
            return FakeResponse(*reply)
        return FakeResponse(reply)


def accept_at(target):
    """Decline every window until the one containing `target`."""
    def reply(prompt):
        return f"{target}(0.90)" if f"{target}," in prompt else "NONE"
    return reply


@pytest.fixture
def targets():
    return [f"t_{i}" for i in range(60)]


@pytest.fixture
def ranking(targets):
    return [(n, 1.0 - i / 100) for i, n in enumerate(targets)]


@pytest.fixture
def target_text(targets):
    return {n: f"Column: {n}, Sample values: [1, 2]" for n in targets}


@pytest.fixture
def source():
    return pd.DataFrame({"age": [1, 2, 3], "nomatch": [4, 5, 6]})


def score_first(score: float):
    """
    Score the first target in the window shown, as a real model would. A fixed
    name would fall outside later windows and stop the walk.
    """
    import re

    def reply(prompt):
        names = re.findall(r"Column: (t_\d+),", prompt)
        return f"{names[0]}({score:.2f})" if names else "NONE"
    return reply
