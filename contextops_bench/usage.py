"""Shared provider-usage normalizer.

Providers surface token-usage fields under inconsistent names. This module is
the single place that knows how to read them, so that:

  - `cached_tokens` means the same thing regardless of provider (cache reads),
  - `cache_creation_tokens` is no longer silently dropped (it was previously
    parsed for cost math but never stored on the response).

Every client calls `extract_usage(raw, provider)` instead of parsing inline.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextops_bench.types import CompletionResponse


@dataclass(frozen=True)
class UsageMetrics:
    """Normalized token-usage metrics from any provider response."""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int              # cache READS (billed at ~10% of input)
    cache_creation_tokens: int      # cache WRITES (billed at ~1.25× input)

    def cost(self, *, model: str) -> float:
        """Estimate USD cost via the shared pricing module."""
        from contextops.pricing import estimate_cost
        return estimate_cost(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached_tokens=self.cached_tokens,
            cache_creation_tokens=self.cache_creation_tokens,
            model=model,
        )


def _as_int(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def extract_usage(raw: dict, provider: str) -> UsageMetrics:
    """Extract normalized UsageMetrics from a provider response.

    `provider` is the client's PROVIDER string (e.g. "openrouter",
    "direct_anthropic", "ollama"). The parsing strategy is chosen by family:
    Anthropic-native uses `input_tokens`/`output_tokens`/`cache_read_input_tokens`;
    OpenAI-compatible (Ollama/LMStudio/OpenRouter) uses `prompt_tokens`/
    `completion_tokens` plus a fallback chain for cached fields.
    """
    usage = (raw.get("usage") or {}) if isinstance(raw, dict) else {}
    ptd = usage.get("prompt_tokens_details") or {}

    if provider == "direct_anthropic":
        # Anthropic native API: top-level input/output, cache fields are explicit.
        return UsageMetrics(
            prompt_tokens=_as_int(usage.get("input_tokens")),
            completion_tokens=_as_int(usage.get("output_tokens")),
            cached_tokens=_as_int(usage.get("cache_read_input_tokens")),
            cache_creation_tokens=_as_int(usage.get("cache_creation_input_tokens")),
        )

    # OpenAI-compatible family (ollama, lmstudio, openrouter).
    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens"))
    # Cached reads: try every field name any provider has been seen to use.
    cached_tokens = (
        _as_int(usage.get("cached_tokens"))
        or _as_int(usage.get("cache_read_input_tokens"))
        or _as_int(ptd.get("cached_tokens"))
    )
    # Cache writes: OpenRouter surfaces this under prompt_tokens_details.
    cache_creation_tokens = _as_int(ptd.get("cache_write_tokens"))
    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


def build_response(
    *, raw: dict, metrics: UsageMetrics, model: str, text: str,
) -> CompletionResponse:
    """Build a CompletionResponse carrying the full usage breakdown.

    Includes `cache_creation_tokens` (previously discarded after cost math).
    """
    resp = CompletionResponse(
        text=text,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        cached_tokens=metrics.cached_tokens,
        cost_usd=metrics.cost(model=model),
        model=raw.get("model", model),
        raw=raw,
    )
    # Stash cache-creation on raw so callers that read resp.raw can see it,
    # and as an attribute for typed access (CompletionResponse is a frozen-ish
    # dataclass; we attach via raw to avoid widening it for now).
    raw["_cache_creation_tokens"] = metrics.cache_creation_tokens
    return resp
