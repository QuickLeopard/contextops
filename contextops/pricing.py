"""Approximate per-model token pricing.

Single source of truth for the $/1M input-token estimates used by
`contextops.optimizer` (and available to the CLI, reports, or other future
consumers). These are ballpark figures for cost-savings *estimates*, not
authoritative billing data — providers change pricing frequently.
Refresh quarterly.
"""

from __future__ import annotations

# $/1M tokens (input).
PRICING: dict[str, float] = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-5": 5.00,
    "claude-opus-4.6": 15.00,
    "claude-sonnet-4.6": 3.00,
    "claude-haiku-4.5": 0.80,
    "qwen3-30b": 0.20,
    "gigachat": 0.10,
    "yandexgpt": 0.10,
}

# Fallback $/1M price for unknown models — conservative middle-of-the-road
# estimate so savings numbers aren't wildly wrong for unlisted models.
DEFAULT_PRICE_PER_M = 1.0

# Cache reads are estimated at ~10% of a full-price input token across
# providers (exact discounts vary — see contextops_bench/clients.py for
# provider-specific cache_read/cache_write multipliers used in real billing).
CACHE_READ_DISCOUNT = 0.1


def price_per_million(model: str) -> float:
    """$/1M input tokens for `model`, falling back to a conservative default."""
    return PRICING.get(model, DEFAULT_PRICE_PER_M)
