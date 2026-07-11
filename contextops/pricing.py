"""Canonical model pricing + cache-aware cost estimation. Single source of truth.

Used by both the library (`optimizer`) and the bench harness so that cost
estimates are consistent and the pricing table lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """$/M tokens for a model."""

    input_per_m: float
    output_per_m: float
    cache_read_factor: float = 0.10   # cached tokens billed at 10% of input
    cache_write_factor: float = 1.25  # 5-min TTL write surcharge


# $/M tokens. Refresh quarterly. Keyed by canonical model id.
# Prefixed aliases (e.g. "anthropic/claude-haiku-4.5") are resolved by
# `estimate_cost` stripping the provider prefix before lookup.
PRICING: dict[str, Price] = {
    "gpt-4o":                  Price(2.50, 10.00),
    "gpt-4o-mini":             Price(0.15, 0.60),
    "gpt-5":                   Price(5.00, 15.00),
    "claude-opus-4.6":         Price(15.00, 75.00),
    "claude-sonnet-4.6":       Price(3.00, 15.00),
    "claude-haiku-4.5":        Price(1.00, 5.00),   # was 0.80 in optimizer — reconciled
    "claude-haiku-4-5":        Price(1.00, 5.00),   # alias: Anthropic native id
    "claude-sonnet-4-6":       Price(3.00, 15.00),  # alias: Anthropic native id
    "claude-opus-4-6":         Price(15.00, 75.00), # alias: Anthropic native id
    "claude-3-haiku":          Price(0.25, 1.25),
    "claude-3-haiku-20240307": Price(0.25, 1.25),
    "qwen3-30b":               Price(0.20, 0.20),
    "gigachat":                Price(0.10, 0.10),
    "yandexgpt":               Price(0.10, 0.10),
    # OpenRouter-only entries (provider-prefixed):
    "meta-llama/llama-3.1-70b-instruct": Price(0.59, 0.79),
    "meta-llama/llama-3.1-8b-instruct":  Price(0.06, 0.06),
    "qwen/qwen-2.5-72b-instruct":        Price(0.40, 0.40),
    "google/gemini-2.0-flash-exp":       Price(0.10, 0.40),
}


def _normalize(model: str) -> str:
    """Strip a single provider prefix (`anthropic/`, `openai/`, ...) for lookup."""
    return model.split("/", 1)[-1] if "/" in model else model


def estimate_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    cache_creation_tokens: int,
    model: str,
) -> float:
    """Estimate USD cost from token usage + model pricing.

    Tries the model id as-is first (so `anthropic/claude-haiku-4.5` matches a
    prefixed key), then the prefix-stripped form (so it also matches
    `claude-haiku-4.5`), then falls back to a default Price.
    """
    price = PRICING.get(model) or PRICING.get(_normalize(model)) or Price(1.0, 1.0)
    non_cached_input = max(0, prompt_tokens - cached_tokens)
    return (
        (non_cached_input / 1_000_000) * price.input_per_m
        + (cached_tokens / 1_000_000) * price.input_per_m * price.cache_read_factor
        + (cache_creation_tokens / 1_000_000) * price.input_per_m * price.cache_write_factor
        + (completion_tokens / 1_000_000) * price.output_per_m
    )
