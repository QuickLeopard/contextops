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


@dataclass
class BenchResult:
    """One benchmark observation."""

    prompt_id: int
    model: str
    provider: str
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

    def __init__(self, base_url: str = "http://localhost:11434/v1", **kwargs):
        super().__init__(base_url, **kwargs)

    def list_models(self) -> list[str]:
        try:
            data = self._get("/models")
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64) -> CompletionResponse:
        t0 = time.time()
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


class OpenRouterClient(BaseHTTPClient):
    """OpenRouter — https://openrouter.ai/api/v1."""

    PROVIDER = "openrouter"

    PRICING = {
        # Rough $/M tokens (input, output). Update quarterly.
        # Source: openrouter.ai/models
        "openai/gpt-4o-mini": (0.15, 0.60),
        "openai/gpt-4o": (2.50, 10.00),
        "anthropic/claude-3.5-haiku": (0.80, 4.00),
        "anthropic/claude-3.5-sonnet": (3.00, 15.00),
        "meta-llama/llama-3.1-70b-instruct": (0.59, 0.79),
        "meta-llama/llama-3.1-8b-instruct": (0.06, 0.06),
        "qwen/qwen-2.5-72b-instruct": (0.40, 0.40),
        "google/gemini-2.0-flash-exp": (0.10, 0.40),
    }

    def __init__(self, api_key: str | None = None, **kwargs):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenRouter requires OPENROUTER_API_KEY env var or api_key arg"
            )
        super().__init__("https://openrouter.ai/api/v1", api_key, **kwargs)

    def complete(self, *, model: str, messages: list[dict],
                 temperature: float = 0.0, max_tokens: int = 64) -> CompletionResponse:
        t0 = time.time()
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
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = (
            usage.get("cached_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
            or 0
        )

        # Estimate cost from PRICING table.
        pricing = self.PRICING.get(model, (1.0, 1.0))
        cost = (
            (prompt_tokens / 1_000_000) * pricing[0]
            + (completion_tokens / 1_000_000) * pricing[1]
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
    """Factory: 'ollama' | 'lmstudio' | 'openrouter' | 'echo'."""
    p = provider.lower()
    if p == "ollama":
        return OllamaClient(**kwargs)
    if p == "lmstudio":
        return LMStudioClient(**kwargs)
    if p == "openrouter":
        return OpenRouterClient(**kwargs)
    if p == "echo":
        return EchoClient()
    raise ValueError(f"Unknown provider: {provider}. Use ollama/lmstudio/openrouter/echo.")