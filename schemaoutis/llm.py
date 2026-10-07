"""LLM backends behind one interface.

`Pager` only calls `client.messages.create(...)` and reads an Anthropic-shaped
response, so any of these can drive it:

  anthropic   the Anthropic API (default)
  openrouter  the same checkpoints through OpenRouter, OpenAI wire format
  ollama      a local or hosted model through Ollama's native /api/chat
  echo        no model; for exercising the plumbing

A gateway that speaks the Anthropic wire format needs no adapter: use
`anthropic` with `base_url=`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT_S = 600.0

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Environment variables consulted for each provider's key, in order.
API_KEY_ENV = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"),
}


# One checkpoint has two names, `claude-sonnet-5` on the Anthropic API and
# `anthropic/claude-sonnet-5` on OpenRouter. Anything keyed on the model uses
# the canonical form so runs over either transport line up.

# Prefix -> OpenRouter vendor namespace.
_VENDOR_BY_PREFIX = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("gemini-", "google"),
)

# Only these prefixes are stripped: an Ollama tag such as
# `hf.co/bartowski/Qwen3-8B-GGUF:Q4` also contains a slash.
_VENDOR_NAMESPACES = frozenset(
    {vendor for _, vendor in _VENDOR_BY_PREFIX}
    | {"anthropic", "openai", "google", "meta-llama", "mistralai", "qwen", "x-ai"}
)


def canonical_model(model: str) -> str:
    """The provider-independent name: `anthropic/claude-sonnet-5` -> `claude-sonnet-5`."""
    name = str(model or "")
    vendor, sep, rest = name.partition("/")
    if sep and vendor.lower() in _VENDOR_NAMESPACES:
        return rest
    return name


def provider_model(model: str, provider: str) -> str:
    """
    The wire id for `model` on `provider`. Raises if the OpenRouter namespace
    cannot be inferred, rather than guessing a vendor.
    """
    name = str(model or "")

    if provider == "openrouter":
        if "/" in name:
            return name
        for prefix, vendor in _VENDOR_BY_PREFIX:
            if name.startswith(prefix):
                return f"{vendor}/{name}"
        raise ValueError(
            f"cannot infer the OpenRouter namespace for model {name!r}. Pass "
            "the full OpenRouter id, e.g. 'anthropic/claude-sonnet-5' (see "
            "https://openrouter.ai/models)."
        )

    if provider == "anthropic":
        return canonical_model(name)

    # ollama, echo: the name is the backend's own and a slash is part of it.
    return name


def load_env(path: str | None = None, override: bool = False) -> list[str]:
    """
    Read a .env file into os.environ and return the names set. The SDKs only
    read os.environ. Searches upwards from the working directory; variables
    already set win unless `override`.
    """
    import os
    from pathlib import Path

    if path is None:
        for parent in [Path.cwd(), *Path.cwd().parents]:
            if (parent / ".env").exists():
                path = parent / ".env"
                break
        else:
            return []

    loaded = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if not value or (name in os.environ and not override):
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded


def resolve_api_key(provider: str, api_key: str | None = None) -> str | None:
    """Explicit key, else the provider's environment variable, else None."""
    if api_key:
        return api_key
    import os

    for name in API_KEY_ENV.get(provider, ()):
        value = os.environ.get(name)
        if value:
            return value
    return None


# Ollama defaults num_ctx to 4096 and silently truncates longer prompts. It can
# only be raised through the native /api/chat, not the OpenAI-compatible /v1.
OLLAMA_DEFAULT_NUM_CTX = 16384


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class AdaptedResponse:
    content: list[Any] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    # A reasoning model's deliberation, kept out of `content` so the parser
    # never reads it as the answer.
    thinking: str = ""
    # Backend timing, when reported (Ollama).
    timings: dict[str, Any] = field(default_factory=dict)


def _attr(block: Any, name: str, default: Any = None) -> Any:
    """Blocks arrive as dicts, these dataclasses, or SDK objects."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`{name, description, input_schema}` -> OpenAI function-tool objects."""
    out = []
    for tool in tools or []:
        schema = tool.get("input_schema") or {"type": "object", "properties": {}}
        # Some servers reject an object schema with no properties.
        if "properties" not in schema:
            schema = {**schema, "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return out


def anthropic_system_to_text(system: Any) -> str:
    """A system prompt given as a string or as a list of text blocks."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        text = _attr(block, "text")
        if text:
            parts.append(str(text))
    return "\n\n".join(parts)


def ollama_messages_to_native(
    messages: list[dict[str, Any]],
    tool_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Translate the conversation for Ollama's native chat endpoint. Ollama's tool
    calls carry no id, so tool results are matched by position and `tool_name`
    rather than `tool_call_id`.
    """
    names = tool_names or {}
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        blocks = list(content or [])

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                btype = _attr(block, "type")
                if btype == "text":
                    text_parts.append(str(_attr(block, "text", "")))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "function": {
                                "name": str(_attr(block, "name", "")),
                                # An object here, not a JSON string as in OpenAI.
                                "arguments": _attr(block, "input", {}) or {},
                            }
                        }
                    )
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts),
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        text_parts = []
        for block in blocks:
            btype = _attr(block, "type")
            if btype == "tool_result":
                body = _attr(block, "content", "")
                if not isinstance(body, str):
                    body = json.dumps(body, default=str)
                entry = {"role": "tool", "content": body}
                name = names.get(str(_attr(block, "tool_use_id", "")))
                if name:
                    entry["tool_name"] = name
                out.append(entry)
            elif btype == "text":
                text_parts.append(str(_attr(block, "text", "")))
        if text_parts:
            out.append({"role": "user", "content": "\n".join(text_parts)})

    return out


def ollama_response_to_anthropic(
    payload: dict[str, Any], model: str, id_prefix: str = "oll"
) -> tuple[AdaptedResponse, dict[str, str]]:
    """
    Translate a native Ollama response. Returns (response, id -> tool name);
    tool-call ids are synthesised because Ollama sends none.
    """
    message = payload.get("message") or {}
    content: list[Any] = []
    names: dict[str, str] = {}

    text = message.get("content")
    if isinstance(text, str) and text.strip():
        content.append(TextBlock(text=text))

    # Reasoning models report their deliberation here when the modelfile routes
    # it to a separate field.
    thinking = message.get("thinking")
    thinking = str(thinking) if isinstance(thinking, str) and thinking.strip() else ""

    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args or "{}")
            except (TypeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        call_id = f"{id_prefix}_{i}"
        name = str(fn.get("name") or "")
        names[call_id] = name
        content.append(ToolUseBlock(id=call_id, name=name, input=args))

    if any(_attr(b, "type") == "tool_use" for b in content):
        stop_reason = "tool_use"
    else:
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "load": "end_turn",
        }.get(payload.get("done_reason") or "stop", "end_turn")

    usage = Usage(
        input_tokens=int(payload.get("prompt_eval_count") or 0),
        output_tokens=int(payload.get("eval_count") or 0),
    )

    # `prompt_eval_count` is the tokens Ollama actually evaluated, so a count
    # well below the prompt's length means the KV cache reused a prefix.
    # Durations arrive in nanoseconds.
    def _ms(field: str) -> float:
        return round(float(payload.get(field) or 0) / 1e6, 2)

    timings = {
        "prompt_eval_count": int(payload.get("prompt_eval_count") or 0),
        "prompt_eval_ms": _ms("prompt_eval_duration"),
        "eval_ms": _ms("eval_duration"),
        "load_ms": _ms("load_duration"),
        "total_ms": _ms("total_duration"),
    }
    return (
        AdaptedResponse(
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            model=payload.get("model") or model,
            thinking=thinking,
            timings=timings,
        ),
        names,
    )


# Ollama's hosted endpoint. A `:cloud` model is usually reached through the
# local daemon after `ollama signin`; pointing base_url here needs an API key.
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"

OLLAMA_API_KEY_ENV = "OLLAMA_API_KEY"


def is_cloud_model(name: str) -> bool:
    """Whether a model name asks Ollama for a hosted model."""
    return str(name).endswith(("-cloud", ":cloud"))


def list_ollama_models(base_url: str = OLLAMA_DEFAULT_BASE_URL,
                       timeout_s: float = 10.0,
                       api_key: str | None = None) -> list[str]:
    """Names of the models this Ollama endpoint offers, or [] if unreachable."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_s,
                         headers=headers)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001 - discovery is best-effort
        return []


class OllamaClient:
    """
    Ollama's native /api/chat behind the Anthropic client surface. Native
    rather than /v1 because only native accepts `num_ctx` and `seed`.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
        seed: int | None = None,
        keep_alive: str = "10m",
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        api_key: str | None = None,
    ):
        import os

        import httpx

        # Accept a /v1 base_url too, as given in the OpenAI-compatible docs.
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self.base_url = base

        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.seed = seed
        self.keep_alive = keep_alive
        self.options = options or {}
        # Ollama 0.9+. `think=False` stops a reasoning model spending its whole
        # budget deliberating and returning no answer. None leaves it to the
        # model's template.
        self.think = think

        # Only the hosted endpoint gets the key.
        self.api_key = api_key or os.environ.get(OLLAMA_API_KEY_ENV) or None
        headers = ({"Authorization": f"Bearer {self.api_key}"}
                   if self.api_key and self.base_url != OLLAMA_DEFAULT_BASE_URL
                   else {})

        self._http = httpx.Client(timeout=timeout_s, headers=headers)
        self._tool_names: dict[str, str] = {}
        self._call_counter = 0
        self.messages = _OllamaShim(self)

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def available_models(self) -> list[str]:
        return list_ollama_models(self.base_url,
                                  timeout_s=min(self.timeout_s, 10.0),
                                  api_key=self.api_key)

    def _build_body(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        system = anthropic_system_to_text(kwargs.get("system"))
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(
            ollama_messages_to_native(kwargs.get("messages") or [], self._tool_names)
        )

        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        # Anthropic's `stop_sequences` is Ollama's `stop` option.
        if kwargs.get("stop_sequences"):
            options["stop"] = list(kwargs["stop_sequences"])
        if self.seed is not None:
            options["seed"] = self.seed
        if kwargs.get("max_tokens"):
            options["num_predict"] = int(kwargs["max_tokens"])
        options.update(self.options)

        body: dict[str, Any] = {
            "model": kwargs.get("model"),
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if self.think is not None:
            body["think"] = self.think
        tools = kwargs.get("tools")
        if tools:
            body["tools"] = anthropic_tools_to_openai(tools)
        return body

    def _create(self, **kwargs: Any) -> AdaptedResponse:
        import httpx

        body = self._build_body(kwargs)
        try:
            resp = self._http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"cannot reach Ollama at {self.base_url}. Start it with "
                "`ollama serve`."
            ) from exc

        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"Ollama refused the request (HTTP {resp.status_code}) at "
                f"{self.base_url}. A hosted model needs `ollama signin`, or "
                f"{OLLAMA_API_KEY_ENV} when base_url is {OLLAMA_CLOUD_BASE_URL}."
            )
        if resp.status_code == 404:
            model = kwargs.get("model")
            offered = self.available_models()
            fix = ("Sign in with `ollama signin`, then `ollama pull "
                   f"{model}`, so the daemon can proxy it."
                   if is_cloud_model(model)
                   else f"Pull it with `ollama pull {model}`.")
            raise RuntimeError(
                f"Ollama has no model {model!r} at {self.base_url}. "
                f"Available: {offered or '(none)'}. {fix}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        self._call_counter += 1
        adapted, names = ollama_response_to_anthropic(
            resp.json(), kwargs.get("model", ""), id_prefix=f"oll{self._call_counter}"
        )
        self._tool_names.update(names)
        return adapted


class _OllamaShim:
    def __init__(self, owner: OllamaClient):
        self._owner = owner

    def create(self, **kwargs: Any) -> AdaptedResponse:
        return self._owner._create(**kwargs)

    def count_tokens(self, **kwargs: Any):
        raise NotImplementedError("Ollama has no token-counting endpoint")


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]], system: Any = None
) -> list[dict[str, Any]]:
    """
    Translate the conversation to OpenAI chat-completions shape. Tool results
    become `{"role": "tool", ...}` entries directly after the assistant turn,
    which is where Anthropic already places them.
    """
    out: list[dict[str, Any]] = []
    system_text = anthropic_system_to_text(system)
    if system_text:
        out.append({"role": "system", "content": system_text})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        blocks = list(content or [])

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                btype = _attr(block, "type")
                if btype == "text":
                    text_parts.append(str(_attr(block, "text", "")))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(_attr(block, "id", "")),
                            "type": "function",
                            "function": {
                                "name": str(_attr(block, "name", "")),
                                # A JSON string here, not an object as in Ollama.
                                "arguments": json.dumps(
                                    _attr(block, "input", {}) or {}, default=str
                                ),
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant"}
            text = "\n".join(p for p in text_parts if p)
            # None, not "": some gateways reject an empty string here.
            entry["content"] = text or None
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        text_parts = []
        for block in blocks:
            btype = _attr(block, "type")
            if btype == "tool_result":
                body = _attr(block, "content", "")
                if not isinstance(body, str):
                    body = json.dumps(body, default=str)
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(_attr(block, "tool_use_id", "")),
                        "content": body,
                    }
                )
            elif btype == "text":
                text_parts.append(str(_attr(block, "text", "")))
        if text_parts:
            out.append({"role": "user", "content": "\n".join(text_parts)})

    return out


def openai_response_to_anthropic(
    payload: dict[str, Any], model: str, id_prefix: str = "or"
) -> AdaptedResponse:
    """Translate an OpenAI chat-completions response into the Anthropic shape."""
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content: list[Any] = []
    text = message.get("content")
    if isinstance(text, str) and text.strip():
        content.append(TextBlock(text=text))

    reasoning = message.get("reasoning")
    thinking = str(reasoning) if isinstance(reasoning, str) and reasoning.strip() else ""

    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args or "{}")
            except (TypeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        content.append(
            ToolUseBlock(
                id=str(call.get("id") or f"{id_prefix}_{i}"),
                name=str(fn.get("name") or ""),
                input=args,
            )
        )

    if any(_attr(b, "type") == "tool_use" for b in content):
        stop_reason = "tool_use"
    else:
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal",
        }.get(choice.get("finish_reason") or "stop", "end_turn")

    raw_usage = payload.get("usage") or {}
    details = raw_usage.get("prompt_tokens_details") or {}
    usage = Usage(
        input_tokens=int(raw_usage.get("prompt_tokens") or 0),
        output_tokens=int(raw_usage.get("completion_tokens") or 0),
        cache_read_input_tokens=int(details.get("cached_tokens") or 0),
    )
    return AdaptedResponse(
        content=content,
        stop_reason=stop_reason,
        usage=usage,
        model=payload.get("model") or model,
        thinking=thinking,
    )


class OpenRouterClient:
    """
    OpenRouter behind the Anthropic client surface.

    With `pin_upstream` (the default) each request is pinned to the model's own
    vendor with fallbacks off, so a run is not split between, say, Anthropic
    and Bedrock without an error.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        pin_upstream: bool = True,
        app_url: str | None = None,
        app_title: str | None = "schemasteer-agentic",
        extra_body: dict[str, Any] | None = None,
    ):
        import httpx

        key = resolve_api_key("openrouter", api_key)
        if not key:
            raise ValueError(
                "no OpenRouter API key. Set OPENROUTER_API_KEY in the "
                "environment or .env, or pass api_key=."
            )
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.pin_upstream = pin_upstream
        self.extra_body = extra_body or {}

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_title:
            headers["X-Title"] = app_title

        self._http = httpx.Client(timeout=timeout_s, headers=headers)
        # Which upstream served each call, as OpenRouter reports it.
        self.upstreams: list[str] = []
        self._call_counter = 0
        self.messages = _OpenRouterShim(self)

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def _routing(self, model: str) -> dict[str, Any] | None:
        """Pin the request to the model's own vendor, with no fallbacks."""
        if not self.pin_upstream or "/" not in model:
            return None
        vendor = model.split("/", 1)[0]
        # OpenRouter's provider slugs are capitalised.
        return {"order": [vendor.capitalize()], "allow_fallbacks": False}

    def _build_body(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        model = provider_model(kwargs.get("model", ""), "openrouter")
        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages_to_openai(
                kwargs.get("messages") or [], kwargs.get("system")
            ),
            "stream": False,
        }
        if kwargs.get("max_tokens"):
            body["max_tokens"] = int(kwargs["max_tokens"])
        if "temperature" in kwargs:
            body["temperature"] = kwargs["temperature"]
        if kwargs.get("stop_sequences"):
            body["stop"] = list(kwargs["stop_sequences"])

        tools = kwargs.get("tools")
        if tools:
            body["tools"] = anthropic_tools_to_openai(tools)
        choice = kwargs.get("tool_choice")
        if choice:
            if choice.get("type") == "tool" and choice.get("name"):
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice["name"]},
                }
            elif choice.get("type") in ("any", "required"):
                body["tool_choice"] = "required"

        routing = self._routing(model)
        if routing:
            body["provider"] = routing
        body.update(self.extra_body)
        return body

    def _create(self, **kwargs: Any) -> AdaptedResponse:
        import httpx

        body = self._build_body(kwargs)
        try:
            resp = self._http.post(f"{self.base_url}/chat/completions", json=body)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"cannot reach OpenRouter at {self.base_url}"
            ) from exc

        if resp.status_code == 401:
            raise RuntimeError(
                "OpenRouter rejected the API key (401). Check OPENROUTER_API_KEY."
            )
        if resp.status_code == 402:
            raise RuntimeError(
                f"OpenRouter has insufficient credit for this request (402): "
                f"{resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        payload = resp.json()
        # OpenRouter can answer 200 with an error body, for instance when the
        # pinned upstream is down.
        if payload.get("error") and not payload.get("choices"):
            err = payload["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"OpenRouter error: {message}")

        served_by = payload.get("provider")
        if served_by:
            self.upstreams.append(str(served_by))

        self._call_counter += 1
        return openai_response_to_anthropic(
            payload, body["model"], id_prefix=f"or{self._call_counter}"
        )

    def upstream_report(self) -> dict[str, Any]:
        """Which upstreams served this client's calls."""
        counts: dict[str, int] = {}
        for name in self.upstreams:
            counts[name] = counts.get(name, 0) + 1
        return {
            "pin_upstream": self.pin_upstream,
            "n_calls_attributed": len(self.upstreams),
            "upstreams": dict(sorted(counts.items())),
            "single_upstream": len(counts) <= 1,
        }


class _OpenRouterShim:
    def __init__(self, owner: OpenRouterClient):
        self._owner = owner

    def create(self, **kwargs: Any) -> AdaptedResponse:
        return self._owner._create(**kwargs)

    def count_tokens(self, **kwargs: Any):
        raise NotImplementedError("OpenRouter exposes no token-counting endpoint")


class EchoClient:
    """
    No model: replies with the candidates named in the prompt, as a tool call.
    Written for SchemaSteer's tool-calling loop; `Pager` reads text replies and
    will find this one unparseable.
    """

    def __init__(self, n_candidates: int = 5):
        self.n_candidates = n_candidates
        self.messages = _EchoShim(self)


class _EchoShim:
    def __init__(self, owner: EchoClient):
        self._owner = owner

    def create(self, **kwargs: Any) -> AdaptedResponse:
        import re

        text = ""
        for msg in kwargs.get("messages") or []:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                text = msg["content"]
                break

        names = re.findall(r'target\."([^"]+)"', text)
        seen: list[str] = []
        for name in names:
            if name not in seen:
                seen.append(name)
        seen = seen[: self._owner.n_candidates] or ["unknown"]

        n = len(seen)
        ranking = [
            {"target_column": name, "score": round((n - i) / n, 4)}
            for i, name in enumerate(seen)
        ]
        return AdaptedResponse(
            content=[
                ToolUseBlock(
                    id=f"echo_{int(time.time() * 1000) % 100000}",
                    name="submit_ranking",
                    input={
                        "ranking": ranking,
                        "rationale": "echo client: biencoder order, no probing",
                    },
                )
            ],
            stop_reason="tool_use",
            usage=Usage(),
            model="echo",
        )

    def count_tokens(self, **kwargs: Any):
        raise NotImplementedError("echo client does not count tokens")


PROVIDERS = ("anthropic", "openrouter", "ollama", "echo")


# The API names the parameter it rejects, e.g. "`temperature` is deprecated
# for this model."
_REJECTED_PARAM = re.compile(
    r"[`'\"]?(\w+)[`'\"]?\s+is\s+(?:deprecated|not supported|unsupported)",
    re.IGNORECASE,
)


class DropsRejectedParams:
    """
    A client that stops sending a parameter the backend refuses.

    Sonnet 5 answers `temperature` with a 400. The first rejection is caught,
    the named parameter is dropped for the rest of the run and the call is
    retried; `dropped` is recorded in the trace.
    """

    # More rejections than this is not a parameter problem.
    MAX_DROPS = 4

    def __init__(self, inner: Any):
        self.inner = inner
        self.dropped: set[str] = set()
        self.messages = _DropShim(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _create(self, **kwargs: Any) -> Any:
        import anthropic

        for _ in range(self.MAX_DROPS + 1):
            attempt = {k: v for k, v in kwargs.items() if k not in self.dropped}
            try:
                return self.inner.messages.create(**attempt)
            except anthropic.BadRequestError as exc:
                m = _REJECTED_PARAM.search(str(exc))
                name = m.group(1) if m else None
                if name is None or name not in attempt:
                    raise
                self.dropped.add(name)
        raise RuntimeError(
            f"the backend rejected more than {self.MAX_DROPS} parameters "
            f"({sorted(self.dropped)}); this is not a parameter problem"
        )


class _DropShim:
    def __init__(self, owner: DropsRejectedParams):
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)

    def count_tokens(self, **kwargs: Any):
        return self._owner.inner.messages.count_tokens(**kwargs)


def build_client(
    provider: str = "anthropic",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    extra_body: dict[str, Any] | None = None,
    num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
    seed: int | None = None,
    think: bool | None = None,
    pin_upstream: bool = True,
) -> Any:
    """The client for a provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"llm_provider must be one of {PROVIDERS}, got {provider!r}")

    if provider == "echo":
        return EchoClient()

    if provider == "openrouter":
        return OpenRouterClient(
            api_key=api_key,
            base_url=base_url or OPENROUTER_DEFAULT_BASE_URL,
            timeout_s=timeout_s,
            pin_upstream=pin_upstream,
            extra_body=extra_body,
        )

    if provider == "ollama":
        return OllamaClient(
            base_url=base_url or OLLAMA_DEFAULT_BASE_URL,
            timeout_s=timeout_s,
            num_ctx=num_ctx,
            seed=seed,
            options=extra_body,
            think=think,
            api_key=api_key,
        )

    import anthropic

    # Checked here: without a key the SDK still builds a client, and every
    # request then fails.
    key = resolve_api_key("anthropic", api_key)
    if not key:
        names = ", ".join(API_KEY_ENV["anthropic"])
        raise ValueError(
            f"no Anthropic API key. Set {names}, call "
            "`schemaoutis.llm.load_env()` if it is in a .env file, or pass "
            "api_key=."
        )

    kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        # A gateway that speaks the Anthropic wire format.
        kwargs["base_url"] = base_url
    return DropsRejectedParams(anthropic.Anthropic(**kwargs))
