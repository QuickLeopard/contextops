"""LLM clients for bench: Ollama, LM Studio, OpenRouter.

All three speak OpenAI-compatible HTTP. We just change `base_url`.
"""

from __future__ import annotations

import os
import time

import urllib.request
import urllib.error
import json as jsonlib

from contextops.pricing import estimate_cost
from contextops_bench.client_protocol import BenchClient
from contextops_bench.types import BenchResult, CompletionResponse  # noqa: F401 (re-exported)


# Models on OpenRouter that honor Anthropic-style explicit prompt cache_control.
# For these, the bench sends the stable prefix in a `system` field with
# `cache_control: {type: "ephemeral"}` so the provider can cache it explicitly.
# Other models on OpenRouter fall back to OpenAI-style auto prefix caching
# (which is position-independent — the cache works regardless of section order).
ANTHROPIC_PREFIXES = ("anthropic/",)


def _is_anthropic_model(model: str) -> bool:
    return any(model.startswith(p) for p in ANTHROPIC_PREFIXES)


class BaseHTTPClient:
    """Minimal OpenAI-compatible client. No external deps."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "sk-no-key-required"
        self.timeout = timeout

    # Subclasses override to True if they support `system=` kwarg with cache_control.
    # For now only OpenRouterClient does, and only for Anthropic-prefixed models.
    supports_split_messages: bool = False

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = jsonlib.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return jsonlib.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return jsonlib.loads(resp.read().decode("utf-8"))


class OllamaClient(BaseHTTPClient):
    """Ollama — http://localhost:11434/v1 (OpenAI-compatible)."""

    PROVIDER = "ollama"
    supports_split_messages: bool = False  # Ollama ignores cache_control

    def __init__(self, base_url: str = "http://localhost:11434/v1", **kwargs):
        super().__init__(base_url, **kwargs)

    def list_models(self) -> list[str]:
        try:
            data = self._get("/models")
            return [m["id"] for m in data.get("data", [])]
        except (urllib.error.URLError, ValueError, KeyError):
            # URLError = server down/unreachable; ValueError = bad JSON;
            # KeyError = unexpected payload shape. Don't mask real bugs.
            return []

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64,
                 system: str | None = None) -> CompletionResponse:
        t0 = time.time()
        # If system is provided, prepend it as a system message
        if system:
            messages = [{"role": "system", "content": system}] + list(messages)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = self._post("/chat/completions", payload)
        raw["_latency_ms"] = (time.time() - t0) * 1000

        choice = raw.get("choices", [{}])[0]
        usage = raw.get("usage", {}) or {}
        return CompletionResponse(
            text=choice.get("message", {}).get("content", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0) or usage.get("cache_read_input_tokens", 0),
            cost_usd=0.0,  # local = free
            model=raw.get("model", model),
            raw=raw,
        )


class LMStudioClient(OllamaClient):
    """LM Studio — same OpenAI-compatible shape, default http://localhost:1234/v1."""

    PROVIDER = "lmstudio"

    def __init__(self, base_url: str = "http://localhost:1234/v1", **kwargs):
        super().__init__(base_url, **kwargs)


class AnthropicDirectClient(BaseHTTPClient):
    """Anthropic native API at https://api.anthropic.com — bypasses OpenRouter.

    This is the only path that reliably surfaces Anthropic's `cache_read_input_tokens`
    in the response. OpenRouter's OpenAI-compatible adapter drops the `cache_control`
    marker during OpenAI→Anthropic translation, so cache stays at 0 even with correct
    setup. Use this for definitive cache measurements.

    Auth: `ANTHROPIC_API_KEY` env var (or pass `api_key=...`).

    Native Anthropic request format:
      POST /v1/messages
      Headers: x-api-key, anthropic-version: 2023-06-01, content-type
      Body: {
        "model": "claude-haiku-4-5-20251001",
        "system": [
          {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [...],
        "max_tokens": ...
      }

    Response usage block has:
      - input_tokens
      - output_tokens
      - cache_creation_input_tokens  (first call writes the prefix to cache)
      - cache_read_input_tokens      (subsequent calls read from cache)
    """

    PROVIDER = "direct_anthropic"
    supports_split_messages: bool = True  # Uses Anthropic's native system field

    # Map our short names to Anthropic's dated model IDs
    MODEL_MAP = {
        "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
        "anthropic/claude-3-haiku": "claude-3-haiku-20240307",
        "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
        "anthropic/claude-opus-4.6": "claude-opus-4-6",
    }

    def __init__(self, api_key: str | None = None, **kwargs):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "AnthropicDirectClient requires ANTHROPIC_API_KEY env var or api_key arg. "
                "Get a free key at https://console.anthropic.com (no card needed for trial credits)."
            )
        super().__init__("https://api.anthropic.com", api_key, **kwargs)

    def list_models(self) -> list[str]:
        return list(self.MODEL_MAP.values())

    def _resolve_model(self, model: str) -> str:
        if model in self.MODEL_MAP:
            return self.MODEL_MAP[model]
        if model.startswith("claude-"):
            return model
        # Try to strip "anthropic/" prefix
        if model.startswith("anthropic/"):
            return model[len("anthropic/"):]
        return model

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64,
                 system: str | None = None) -> CompletionResponse:
        t0 = time.time()
        native_model = self._resolve_model(model)

        # Anthropic native format: system is a top-level field, not in messages.
        # cache_control goes on the system content block, not on each message.
        payload: dict = {
            "model": native_model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Anthropic-specific headers
        url = f"{self.base_url}/v1/messages"
        data = jsonlib.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = jsonlib.loads(resp.read().decode("utf-8"))
        raw["_latency_ms"] = (time.time() - t0) * 1000

        content_blocks = raw.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = raw.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        # Anthropic native: cache_creation_input_tokens (first call) and
        # cache_read_input_tokens (subsequent calls). Both are 0 if cache
        # isn't activating.
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

        cost = estimate_cost(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cached_tokens=cached_tokens, cache_creation_tokens=cache_creation_tokens,
            model=native_model,
        )

        return CompletionResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            model=raw.get("model", native_model),
            raw=raw,
        )


class OpenRouterClient(BaseHTTPClient):
    """OpenRouter — https://openrouter.ai/api/v1."""

    PROVIDER = "openrouter"
    supports_split_messages: bool = True  # Anthropic models get explicit cache_control

    # Per-block cache_control intro split heuristic (see `_shape_messages`).
    # OpenRouter requires the cache_control marker on the SECOND-or-later block,
    # so we split the system text into a small intro + the cacheable body.
    _INTRO_SPLIT_MAX = 200
    _INTRO_SPLIT_DIVISOR = 4

    def __init__(self, api_key: str | None = None, **kwargs):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenRouter requires OPENROUTER_API_KEY env var or api_key arg"
            )
        super().__init__("https://openrouter.ai/api/v1", api_key, **kwargs)
        # Read env once at construction, not on every complete() call.
        self.cache_mode = os.environ.get("OPENROUTER_CACHE_MODE", "per_block")
        self.provider_pin = os.environ.get("OPENROUTER_PROVIDER_PIN")
        self.debug_provider = bool(os.environ.get("OPENROUTER_DEBUG_PROVIDER"))

    def _shape_messages(
        self, *, model: str, messages: list[dict], system: str | None,
    ) -> list[dict]:
        """Prepend `system` to messages, adding cache_control for Anthropic models.

        - Anthropic + `top_level` mode: plain system message (cache_control goes
          on the payload, handled by the caller).
        - Anthropic + `per_block` mode (default): split system into two blocks
          with `cache_control` on the SECOND block (OpenRouter requires the
          marker not be on the first block).
        - Non-Anthropic: plain system message prepend.
        - No system: messages unchanged.
        """
        if not system:
            return messages
        if _is_anthropic_model(model):
            if self.cache_mode == "top_level":
                return [{"role": "system", "content": system}] + list(messages)
            split_at = min(self._INTRO_SPLIT_MAX, len(system) // self._INTRO_SPLIT_DIVISOR)
            return [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system[:split_at]},
                        {
                            "type": "text",
                            "text": system[split_at:],
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ] + list(messages)
        return [{"role": "system", "content": system}] + list(messages)

    def _apply_provider_pinning(
        self, payload: dict, *, model: str, pin_provider: str | list[str] | None,
    ) -> None:
        """Mutate `payload` in place to pin the upstream provider.

        With `top_level` cache mode the top-level cache_control field already
        forces Anthropic-direct routing. With `per_block` we explicitly pin to
        Anthropic to avoid the default Bedrock route (whose cache is separate).
        """
        if pin_provider is None:
            pin_provider = self.provider_pin
        if _is_anthropic_model(model) and pin_provider is None and self.cache_mode != "top_level":
            pin_provider = ["anthropic"]
        if pin_provider:
            if isinstance(pin_provider, str):
                pin_provider = [pin_provider]
            payload["provider"] = {"order": pin_provider, "allow_fallbacks": False}

    def _maybe_debug(self, *, raw: dict, cached_tokens: int, prompt_tokens: int) -> None:
        """Print a one-line cache/provider trace when OPENROUTER_DEBUG_PROVIDER is set."""
        if not self.debug_provider:
            return
        provider = raw.get("provider", "?")
        print(
            f"  [debug] provider={provider:<20s} cached={cached_tokens:>5}/{prompt_tokens:<5}",
            flush=True,
        )

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64,
                 system: str | None = None,
                 pin_provider: str | list[str] | None = None) -> CompletionResponse:
        t0 = time.time()
        messages = self._shape_messages(model=model, messages=messages, system=system)

        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # OpenRouter field — requests detailed usage breakdown including
            # prompt_tokens_details.cached_tokens. Some accounts / endpoints
            # only return this when explicitly asked.
            "usage": {"include": True},
        }

        # Top-level cache_control (Anthropic automatic caching). When set,
        # OpenRouter auto-pins to Anthropic direct and excludes Bedrock/Vertex.
        if system and _is_anthropic_model(model) and self.cache_mode == "top_level":
            payload["cache_control"] = {"type": "ephemeral"}

        self._apply_provider_pinning(payload, model=model, pin_provider=pin_provider)

        raw = self._post("/chat/completions", payload)
        raw["_latency_ms"] = (time.time() - t0) * 1000

        choice = raw.get("choices", [{}])[0]
        usage = raw.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        # OpenRouter may surface cache info in any of these depending on the model
        cached_tokens = (
            usage.get("cached_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        )

        # Estimate cost via the shared pricing module (resolves the provider
        # prefix on the OpenRouter model id, e.g. "anthropic/claude-haiku-4.5").
        cache_creation_tokens = (
            (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens", 0)
        )
        cost = estimate_cost(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cached_tokens=cached_tokens, cache_creation_tokens=cache_creation_tokens,
            model=model,
        )

        self._maybe_debug(raw=raw, cached_tokens=cached_tokens, prompt_tokens=prompt_tokens)

        return CompletionResponse(
            text=choice.get("message", {}).get("content", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            model=raw.get("model", model),
            raw=raw,
        )


class EchoClient:
    """Offline stub that simulates realistic provider-cache behavior.

    Cache hit estimation rule (matches real Anthropic / OpenAI behavior):
      - The prompt prefix is "stable" if its first N tokens match the previous call.
      - In a serial loop, sequential prompts share the same `system` + `tools` prefix,
        so cache hit should grow as more calls accumulate.
      - Edge cases (empty, huge) behave realistically — empty prompt = 100% cache hit
        (whole thing fits in any prefix), huge prompt = ~0% cache hit (rarely repeated).

    Use this for CI smoke tests where you can't hit a real provider.
    """

    PROVIDER = "echo"
    supports_split_messages: bool = False  # offline stub; ignores `system`

    def __init__(self, base_cache_rate: float = 0.30):
        self.calls = 0
        self._seen_prefixes: dict[str, int] = {}
        self.base_cache_rate = base_cache_rate

    def reset(self) -> None:
        """Clear cache state. Called between bench phases for fair A/B comparison."""
        self.calls = 0
        self._seen_prefixes = {}

    def list_models(self) -> list[str]:
        return ["echo-model"]

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64,
                 system: str | None = None) -> CompletionResponse:
        # `system` is accepted to satisfy the BenchClient Protocol (LSP) but
        # ignored: the offline simulation keys cache hits off the message content.
        self.calls += 1
        last = messages[-1]["content"] if messages else ""
        prompt_tokens = max(1, len(last) // 4)

        # Split the prompt on blank lines (which is how ContextOps renders sections)
        # and use the FIRST section (system+tools) as the stable-prefix fingerprint.
        # This mirrors what real providers cache: the static prefix.
        sections = last.split("\n\n")
        stable_prefix = sections[0] if sections else last[:64]

        seen = self._seen_prefixes.get(stable_prefix, 0)
        self._seen_prefixes[stable_prefix] = seen + 1

        # Cache hit rate grows with repeats, caps at base_cache_rate.
        if seen == 0:
            cache_rate = 0.0
        else:
            cache_rate = min(self.base_cache_rate, self.base_cache_rate * (seen / (seen + 1)))

        cached_tokens = int(prompt_tokens * cache_rate)
        return CompletionResponse(
            text="echo",
            prompt_tokens=prompt_tokens,
            completion_tokens=4,
            cached_tokens=cached_tokens,
            cost_usd=0.0,
            model=model,
            raw={},
        )


# Provider → client class registry. Single source of truth for what providers
# exist; `get_client` and the CLI `choices` both derive from this.
CLIENTS: dict[str, type] = {
    cls.PROVIDER: cls for cls in
    (OllamaClient, LMStudioClient, OpenRouterClient, AnthropicDirectClient, EchoClient)
}

# Convenience aliases (shorter/older names) → canonical provider key.
_PROVIDER_ALIASES: dict[str, str] = {
    "anthropic": "direct_anthropic",
}


def get_client(provider: str, **kwargs) -> BenchClient:
    """Factory: one of the keys in `CLIENTS` (see `sorted(CLIENTS)`).

    Accepts the aliases in `_PROVIDER_ALIASES` (e.g. "anthropic" → "direct_anthropic").
    """
    p = provider.lower()
    p = _PROVIDER_ALIASES.get(p, p)
    if p not in CLIENTS:
        raise ValueError(
            f"Unknown provider: {provider}. Use one of: {sorted(CLIENTS)}."
        )
    return CLIENTS[p](**kwargs)