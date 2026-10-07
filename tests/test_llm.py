"""
Building the LLM clients: which endpoint each talks to and what it sends. No
requests are made.
"""

from __future__ import annotations

import os

import pytest

from schemaoracle.llm import (
    OLLAMA_API_KEY_ENV,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_DEFAULT_BASE_URL,
    DropsRejectedParams,
    build_client,
    load_env,
    is_cloud_model,
)


@pytest.mark.parametrize("name, cloud", [
    ("deepseek-v4-flash:cloud", True),
    ("gpt-oss:120b-cloud", True),
    ("qwen3.5:9b", False),
    ("gemma4:e4b", False),
])
def test_a_hosted_model_is_recognised_by_its_name(name, cloud):
    assert is_cloud_model(name) is cloud


def test_the_local_daemon_is_never_sent_a_key(monkeypatch):
    """
    A hosted model reached through `ollama signin` goes over localhost, and the
    daemon never asked for a credential. Sending one anyway puts it on a wire
    that did not need it.
    """
    monkeypatch.setenv(OLLAMA_API_KEY_ENV, "sk-should-not-be-sent")
    client = build_client("ollama")
    assert client.base_url == OLLAMA_DEFAULT_BASE_URL
    assert "authorization" not in client._http.headers


def test_the_hosted_endpoint_gets_the_key(monkeypatch):
    monkeypatch.setenv(OLLAMA_API_KEY_ENV, "sk-test")
    client = build_client("ollama", base_url=OLLAMA_CLOUD_BASE_URL)
    assert client.base_url == OLLAMA_CLOUD_BASE_URL
    assert client._http.headers["authorization"] == "Bearer sk-test"


def test_an_explicit_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv(OLLAMA_API_KEY_ENV, "sk-env")
    client = build_client("ollama", base_url=OLLAMA_CLOUD_BASE_URL,
                          api_key="sk-explicit")
    assert client._http.headers["authorization"] == "Bearer sk-explicit"


def test_no_key_anywhere_is_not_an_error(monkeypatch):
    """Discovery and the local path both work without one."""
    monkeypatch.delenv(OLLAMA_API_KEY_ENV, raising=False)
    client = build_client("ollama", base_url=OLLAMA_CLOUD_BASE_URL)
    assert client.api_key is None
    assert "authorization" not in client._http.headers


def test_a_v1_base_url_is_accepted_and_trimmed():
    """
    The OpenAI-compatible docs hand out the /v1 URL, and it is an easy thing to
    paste. Native /api/chat is the only endpoint that takes num_ctx.
    """
    client = build_client("ollama", base_url="http://localhost:11434/v1")
    assert client.base_url == "http://localhost:11434"


def test_thinking_can_be_turned_off_per_client():
    """
    A reasoning model spends its token budget in the thinking channel and
    returns empty content. On this prompt that is the difference between an
    answer and nothing at all.
    """
    assert build_client("ollama", think=False).think is False
    assert build_client("ollama").think is None


# Credentials

def test_a_missing_anthropic_key_fails_before_the_first_call(monkeypatch):
    """
    Without this check the SDK builds a client anyway and every request fails,
    one authentication error per column.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no Anthropic API key"):
        build_client("anthropic")


def test_an_explicit_anthropic_key_is_enough(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert build_client("anthropic", api_key="sk-test") is not None


def test_load_env_puts_a_dotenv_key_where_the_sdk_looks(monkeypatch, tmp_path):
    """The gap that caused it: the key was in .env and nothing read .env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('ANTHROPIC_API_KEY="sk-from-file"\n# a comment\n\n',
                   encoding="utf-8")

    assert load_env(env) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-file"
    assert build_client("anthropic") is not None


def test_load_env_does_not_overwrite_a_key_already_exported(monkeypatch, tmp_path):
    """A shell export is the more deliberate of the two."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-from-file\n", encoding="utf-8")

    assert load_env(env) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-shell"
    assert load_env(env, override=True) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-file"


def test_load_env_is_quiet_when_there_is_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_env() == []


# Parameters a newer checkpoint no longer takes

def _bad_request(message):
    import anthropic
    import httpx

    return anthropic.BadRequestError(
        message,
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )


class FakeMessages:
    """Rejects the named parameter the way the real API does."""

    def __init__(self, reject=(), text="OK"):
        self.reject = set(reject)
        self.text = text
        self.seen = []

    def create(self, **kwargs):
        self.seen.append(dict(kwargs))
        for name in self.reject:
            if name in kwargs:
                raise _bad_request(f"`{name}` is deprecated for this model.")
        return self.text

    def count_tokens(self, **kwargs):
        return 0


class FakeAnthropic:
    def __init__(self, reject=()):
        self.messages = FakeMessages(reject)


def test_a_deprecated_parameter_is_dropped_and_the_call_retried():
    """
    Sonnet 5 answers `temperature` with a 400. Every call in a sweep carries
    one, so all of them fail for a reason that has nothing to do with the data.
    """
    inner = FakeAnthropic(reject=["temperature"])
    client = DropsRejectedParams(inner)

    assert client.messages.create(model="m", temperature=0.0, max_tokens=8) == "OK"
    assert client.dropped == {"temperature"}
    assert len(inner.messages.seen) == 2               # rejected, then retried
    assert "temperature" not in inner.messages.seen[1]
    assert inner.messages.seen[1]["max_tokens"] == 8   # nothing else lost


def test_the_drop_is_remembered_for_the_rest_of_the_run():
    """Otherwise every column pays for a wasted round trip."""
    inner = FakeAnthropic(reject=["temperature"])
    client = DropsRejectedParams(inner)

    for _ in range(3):
        client.messages.create(model="m", temperature=0.0)
    assert len(inner.messages.seen) == 4               # 2 for the first, 1 each after


def test_a_rejection_naming_something_we_did_not_send_is_raised():
    """Retrying cannot help, and swallowing it would hide a real problem."""
    import anthropic

    class Odd(FakeAnthropic):
        def __init__(self):
            super().__init__()
            self.messages.create = lambda **kw: (_ for _ in ()).throw(
                _bad_request("`top_k` is deprecated for this model."))

    with pytest.raises(anthropic.BadRequestError):
        DropsRejectedParams(Odd()).messages.create(model="m", temperature=0.0)


def test_an_unrelated_bad_request_is_raised_unchanged():
    import anthropic

    class Odd(FakeAnthropic):
        def __init__(self):
            super().__init__()
            self.messages.create = lambda **kw: (_ for _ in ()).throw(
                _bad_request("credit balance is too low"))

    with pytest.raises(anthropic.BadRequestError, match="credit balance"):
        DropsRejectedParams(Odd()).messages.create(model="m", temperature=0.0)


def test_a_backend_rejecting_everything_is_not_retried_forever():
    inner = FakeAnthropic(reject=["a", "b", "c", "d", "e", "f"])
    client = DropsRejectedParams(inner)
    with pytest.raises(RuntimeError, match="not a parameter problem"):
        client.messages.create(**{k: 1 for k in "abcdef"})


def test_the_wrapper_passes_everything_else_through():
    inner = FakeAnthropic()
    inner.base_url = "https://api.anthropic.com"
    assert DropsRejectedParams(inner).base_url == "https://api.anthropic.com"
