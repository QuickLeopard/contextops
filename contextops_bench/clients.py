"""LLM clients for bench: Ollama, LM Studio, OpenRouter.

All three speak OpenAI-compatible HTTP. We just change `base_url`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import urllib.request
import urllib.error
import json as jsonlib


# Models on OpenRouter that honor Anthropic-style explicit prompt cache_control.
# For these, the bench sends the stable prefix in a `system` field with
# `cache_control: {type: "ephemeral"}` so the provider can cache it explicitly.
# Other models on OpenRouter fall back to OpenAI-style auto prefix caching
# (which is position-independent — the cache works regardless of section order).
ANTHROPIC_PREFIXES = ("anthropic/",)


def _is_anthropic_model(model: str) -> bool:
    return any(model.startswith(p) for p in ANTHROPIC_PREFIXES)


@dataclass
class BenchResult:
    """One benchmark observation."""

    prompt_id: int
    model: str
    provider: str
    use_optimized: bool
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    section_order: list[str] = field(default_factory=list)


@dataclass
class CompletionResponse:
    """Normalised response from any provider."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    model: str
    raw: dict


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
        except Exception:
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

    PRICING = {
        # $/M tokens (input, output). Update from anthropic.com/pricing.
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-opus-4-6": (15.00, 75.00),
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
        cached_tokens = (
            usage.get("cache_read_input_tokens", 0)
            or 0
        )
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

        # Estimate cost from PRICING table. Cache reads are 10% of input cost
        # and cache writes are 25% more than input cost.
        pricing = self.PRICING.get(native_model, (1.0, 5.0))
        input_cost = pricing[0]
        output_cost = pricing[1]
        # Non-cached input tokens
        non_cached_input = max(0, prompt_tokens - cached_tokens)
        cost = (
            (non_cached_input / 1_000_000) * input_cost
            + (cached_tokens / 1_000_000) * input_cost * 0.1  # cache_read = 10% of input
            + (cache_creation_tokens / 1_000_000) * input_cost * 1.25  # cache_write = 125% of input
            + (completion_tokens / 1_000_000) * output_cost
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


class ZenDirectClient(BaseHTTPClient):
    """OpenCode Zen gateway — uses Anthropic's native format at /v1/messages.

    Zen is a curated model gateway by the OpenCode team. For Anthropic models,
    it exposes Anthropic's native API format at /v1/messages (not OpenAI-compatible).
    This means:
      - cache_control: {type: "ephemeral"} markers pass through to Anthropic
      - cache_read_input_tokens and cache_creation_input_tokens come back
      - No Bedrock routing (Zen always goes Anthropic direct for Claude models)
      - Same pricing as direct Anthropic (Zen is pass-through at cost)

    Auth: `ZEN_API_KEY` env var (or pass `api_key=...`).
    Endpoint: `https://opencode.ai/zen/v1/messages` for Anthropic models.

    Model ID format is the bare name, e.g. `claude-sonnet-4-6` (NOT prefixed
    with `anthropic/`). Zen handles the routing internally.
    """

    PROVIDER = "direct_zen"
    supports_split_messages: bool = True  # Uses Anthropic's native system field

    MODEL_MAP = {
        # Zen's Anthropic model names use dashes, not dots
        "claude-haiku-4-5": "claude-haiku-4-5",
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-sonnet-5": "claude-sonnet-5",
        "claude-opus-4-6": "claude-opus-4-6",
    }

    PRICING = {
        # Same as direct Anthropic — Zen is pass-through at cost
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-sonnet-5": (2.00, 10.00),
        "claude-opus-4-6": (5.00, 25.00),
    }

    def __init__(self, api_key: str | None = None, **kwargs):
        # Try multiple env var names — Zen is new and the convention isn't settled.
        # Print which one we found so users can debug auth issues.
        api_key = (
            api_key
            or os.environ.get("ZEN_API_KEY")
            or os.environ.get("OPENCODE_API_KEY")
            or os.environ.get("OPENCODE_ZEN_API_KEY")
            or os.environ.get("OPENCODE_ZEN_KEY")
            or ""
        )
        if not api_key:
            raise ValueError(
                "ZenDirectClient requires one of these env vars: ZEN_API_KEY, "
                "OPENCODE_API_KEY, OPENCODE_ZEN_API_KEY, OPENCODE_ZEN_KEY. "
                "Get a key at https://opencode.ai/zen"
            )
        # Debug: show first 8 chars of the key (so user can confirm it's set)
        # and which env var won. Helps debug 403/401 issues.
        if os.environ.get("DEBUG_KEY"):
            print(
                f"[ZenDirectClient] using key starting with: {api_key[:8]}... "
                f"(len={len(api_key)})",
                flush=True,
            )
        super().__init__("https://opencode.ai/zen/v1", api_key, **kwargs)

    def _resolve_model(self, model: str) -> str:
        if model in self.MODEL_MAP:
            return self.MODEL_MAP[model]
        # Strip common prefixes
        for prefix in ("opencode/", "anthropic/", "zen/"):
            if model.startswith(prefix):
                return model[len(prefix):]
        return model

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64,
                 system: str | None = None) -> CompletionResponse:
        t0 = time.time()
        native_model = self._resolve_model(model)

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

        # Zen uses Anthropic's NATIVE auth: x-api-key (lowercase). NOT Bearer.
        # Zen also requires a non-default User-Agent — Cloudflare WAF blocks
        # the default `Python-urllib/3.x` with error 1010. Use `opencode-cli`
        # (the canonical OpenCode UA). Override with ZEN_USER_AGENT env var.
        url = f"{self.base_url}/messages"
        data = jsonlib.dumps(payload).encode("utf-8")
        user_agent = os.environ.get("ZEN_USER_AGENT", "opencode-cli/0.5.0")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "User-Agent": user_agent,
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
        cached_tokens = usage.get("cache_read_input_tokens", 0)
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

        pricing = self.PRICING.get(native_model, (1.0, 5.0))
        input_cost = pricing[0]
        non_cached_input = max(0, prompt_tokens - cached_tokens)
        cost = (
            (non_cached_input / 1_000_000) * input_cost
            + (cached_tokens / 1_000_000) * input_cost * 0.1
            + (cache_creation_tokens / 1_000_000) * input_cost * 1.25
            + (completion_tokens / 1_000_000) * pricing[1]
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

    PRICING = {
        # Rough $/M tokens (input, output). Update quarterly from openrouter.ai/models.
        "openai/gpt-4o-mini": (0.15, 0.60),
        "openai/gpt-4o": (2.50, 10.00),
        "anthropic/claude-haiku-4.5": (1.00, 5.00),
        "anthropic/claude-3-haiku": (0.25, 1.25),
        "anthropic/claude-sonnet-4.6": (3.00, 15.00),
        "meta-llama/llama-3.1-70b-instruct": (0.59, 0.79),
        "meta-llama/llama-3.1-8b-instruct": (0.06, 0.06),
        "qwen/qwen-2.5-72b-instruct": (0.40, 0.40),
        "google/gemini-2.0-flash-exp": (0.10, 0.40),
        # Gemini 2.5 has implicit caching with NO cache_control needed.
        "google/gemini-2.5-flash": (0.30, 2.50),
        "google/gemini-2.5-flash-lite": (0.10, 0.40),
        "google/gemini-2.5-pro": (1.25, 10.00),
    }

    # Cache pricing per model (cache_read, cache_write) as multipliers of input price.
    # Anthropic: 0.10 read, 1.25 write (write costs MORE than input).
    # Gemini:    0.10 read, 0.28 write (write costs LESS than input — opposite).
    CACHE_MULTIPLIERS = {
        "anthropic/claude-haiku-4.5": (0.10, 1.25),
        "anthropic/claude-3-haiku": (0.10, 1.25),
        "anthropic/claude-sonnet-4.6": (0.10, 1.25),
        "google/gemini-2.5-flash": (0.10, 0.28),
        "google/gemini-2.5-flash-lite": (0.10, 0.28),
        "google/gemini-2.5-pro": (0.10, 0.30),
    }

    def __init__(self, api_key: str | None = None, **kwargs):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenRouter requires OPENROUTER_API_KEY env var or api_key arg"
            )
        super().__init__("https://openrouter.ai/api/v1", api_key, **kwargs)

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64,
                 system: str | None = None,
                 pin_provider: str | list[str] | None = None) -> CompletionResponse:
        t0 = time.time()
        # Cache mode:
        #   "per_block" (default for our use case) — cache_control on a content block.
        #                    For stable system + variable user (our A/B), this is correct:
        #                    the system block is cacheable, the user block changes every
        #                    call. Per OpenRouter docs, the cache_control must be on the
        #                    SECOND-or-later block (not the first), otherwise the marker
        #                    is at position 0 and there's no prefix to cache.
        #   "top_level"        — top-level cache_control, recommended by OpenRouter for
        #                    multi-turn conversations. Caches "all content up to the last
        #                    cacheable block" — but the last block is the user message,
        #                    which changes every call, so the cache key changes every call
        #                    and we never hit. Bad fit for our single-turn A/B.
        # Set OPENROUTER_CACHE_MODE=top_level to switch.
        cache_mode = os.environ.get("OPENROUTER_CACHE_MODE", "per_block")

        # For Anthropic models with system content: structure the request so the
        # cache_control marker is on a meaningful position.
        if system and _is_anthropic_model(model):
            if cache_mode == "top_level":
                # Top-level automatic caching: cache_control at the top of the
                # request body. OpenRouter handles the breakpoint placement.
                # Note: this also forces Anthropic-direct routing (no Bedrock/Vertex).
                messages = [{"role": "system", "content": system}] + list(messages)
            else:
                # Per-block explicit: split system into two blocks, put
                # cache_control on the SECOND block. First block is a small
                # "intro" (no cache_control), second is the bulk of the
                # content (with cache_control). This matches the OpenRouter
                # docs example exactly.
                split_at = min(200, len(system) // 4)
                intro = system[:split_at]
                body = system[split_at:]
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": intro},
                            {
                                "type": "text",
                                "text": body,
                                "cache_control": {"type": "ephemeral"},
                            },
                        ],
                    }
                ] + list(messages)
        elif system:
            messages = [{"role": "system", "content": system}] + list(messages)

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

        # OpenRouter sticky routing is keyed by the first system + first
        # user message. Since each bench prompt has a different user message,
        # OpenRouter would treat each call as a new conversation and route
        # to a different endpoint, fragmenting the cache. We override that
        # with an explicit session_id so all calls in a bench run stick
        # to the same endpoint and share the cache.
        # Set OPENROUTER_SESSION_ID env var to a custom value, otherwise we
        # auto-derive one from the model + bench run.
        session_id = os.environ.get("OPENROUTER_SESSION_ID")
        if not session_id and _is_anthropic_model(model):
            # Auto-derive a stable session ID for the bench run. This gets
            # the sticky-routing benefit without forcing the user to set
            # an env var.
            import hashlib
            session_id = f"contextops-bench-{hashlib.md5(model.encode()).hexdigest()[:12]}"
        if session_id:
            payload["session_id"] = session_id

        # Top-level cache_control (Anthropic automatic caching). When set,
        # OpenRouter auto-pins to Anthropic direct and excludes Bedrock/Vertex.
        if system and _is_anthropic_model(model) and cache_mode == "top_level":
            payload["cache_control"] = {"type": "ephemeral"}

        # Provider pinning. With top_level mode, the cache_control field
        # already forces Anthropic direct. With per_block mode, we explicitly
        # pin to avoid the default Bedrock route.
        if pin_provider is None:
            pin_provider = os.environ.get("OPENROUTER_PROVIDER_PIN")
        if _is_anthropic_model(model) and pin_provider is None and cache_mode != "top_level":
            # Only pin explicitly when top_level isn't doing it for us
            pin_provider = ["anthropic"]
        if pin_provider:
            if isinstance(pin_provider, str):
                pin_provider = [pin_provider]
            payload["provider"] = {
                "order": pin_provider,
                "allow_fallbacks": False,
            }

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
            or 0
        )

        # Estimate cost from PRICING table. Apply cache_read discount and
        # cache_write surcharge using per-model multipliers. Defaults to
        # Anthropic-like (0.10 read, 1.25 write) if model not in CACHE_MULTIPLIERS.
        cache_creation_tokens = (
            (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens", 0)
        )
        pricing = self.PRICING.get(model, (1.0, 1.0))
        input_cost = pricing[0]
        read_mult, write_mult = self.CACHE_MULTIPLIERS.get(model, (0.10, 1.25))
        non_cached_input = max(0, prompt_tokens - cached_tokens)
        cost = (
            (non_cached_input / 1_000_000) * input_cost
            + (cached_tokens / 1_000_000) * input_cost * read_mult
            + (cache_creation_tokens / 1_000_000) * input_cost * write_mult
            + (completion_tokens / 1_000_000) * pricing[1]
        )

        # Optional per-call debug log of upstream provider + cache state.
        # Set OPENROUTER_DEBUG_PROVIDER=1 to enable. Helps verify that
        # OpenRouter is routing cache-bearing calls to the same provider
        # (otherwise the cache gets invalidated between calls).
        if os.environ.get("OPENROUTER_DEBUG_PROVIDER"):
            provider = raw.get("provider", "?")
            print(
                f"  [debug] provider={provider:<20s} cached={cached_tokens:>5}/{prompt_tokens:<5}",
                flush=True,
            )

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
                 temperature: float = 0.0, max_tokens: int = 64) -> CompletionResponse:
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


def get_client(provider: str, **kwargs) -> BaseHTTPClient | EchoClient:
    """Factory: 'ollama' | 'lmstudio' | 'openrouter' | 'direct_anthropic' | 'direct_zen' | 'echo'."""
    p = provider.lower()
    if p == "ollama":
        return OllamaClient(**kwargs)
    if p == "lmstudio":
        return LMStudioClient(**kwargs)
    if p == "openrouter":
        return OpenRouterClient(**kwargs)
    if p == "direct_anthropic" or p == "anthropic":
        return AnthropicDirectClient(**kwargs)
    if p == "direct_zen" or p == "zen":
        return ZenDirectClient(**kwargs)
    if p == "echo":
        return EchoClient()
    raise ValueError(
        f"Unknown provider: {provider}. "
        f"Use ollama/lmstudio/openrouter/direct_anthropic/direct_zen/echo."
    )