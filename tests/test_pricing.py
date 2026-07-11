"""Tests for the shared pricing module."""

from contextops.pricing import PRICING, Price, estimate_cost


def test_estimate_cost_basic():
    # 2000 prompt tokens, 100 completion, gpt-4o-mini (0.15 in / 0.60 out), no cache.
    cost = estimate_cost(
        prompt_tokens=2000, completion_tokens=100,
        cached_tokens=0, cache_creation_tokens=0, model="gpt-4o-mini",
    )
    expected = (2000 / 1_000_000) * 0.15 + (100 / 1_000_000) * 0.60
    assert abs(cost - expected) < 1e-12


def test_estimate_cost_cached_tokens_discounted():
    # cached tokens billed at 10% of input price.
    cost = estimate_cost(
        prompt_tokens=2000, completion_tokens=0,
        cached_tokens=2000, cache_creation_tokens=0, model="gpt-4o-mini",
    )
    expected = (2000 / 1_000_000) * 0.15 * 0.10
    assert abs(cost - expected) < 1e-12


def test_estimate_cost_cache_write_surcharge():
    # cache_creation billed at 1.25x input price.
    cost = estimate_cost(
        prompt_tokens=0, completion_tokens=0,
        cached_tokens=0, cache_creation_tokens=1000, model="gpt-4o-mini",
    )
    expected = (1000 / 1_000_000) * 0.15 * 1.25
    assert abs(cost - expected) < 1e-12


def test_dotted_and_dashed_aliases_match():
    # The old drift: optimizer had claude-haiku-4.5=0.80, bench had 1.00.
    # Both aliases must now resolve to the same reconciled price.
    cost_dotted = estimate_cost(
        prompt_tokens=1_000_000, completion_tokens=0,
        cached_tokens=0, cache_creation_tokens=0, model="claude-haiku-4.5",
    )
    cost_dashed = estimate_cost(
        prompt_tokens=1_000_000, completion_tokens=0,
        cached_tokens=0, cache_creation_tokens=0, model="claude-haiku-4-5",
    )
    assert cost_dotted == cost_dashed == 1.00


def test_prefixed_model_resolves():
    # OpenRouter-style id strips the provider prefix before lookup.
    cost_prefixed = estimate_cost(
        prompt_tokens=1_000_000, completion_tokens=0,
        cached_tokens=0, cache_creation_tokens=0, model="anthropic/claude-haiku-4.5",
    )
    cost_bare = estimate_cost(
        prompt_tokens=1_000_000, completion_tokens=0,
        cached_tokens=0, cache_creation_tokens=0, model="claude-haiku-4.5",
    )
    assert cost_prefixed == cost_bare


def test_unknown_model_falls_back():
    cost = estimate_cost(
        prompt_tokens=1_000_000, completion_tokens=1_000_000,
        cached_tokens=0, cache_creation_tokens=0, model="some-future-model-9x9",
    )
    # default Price(1.0, 1.0): 1M in + 1M out = $2.00
    assert abs(cost - 2.00) < 1e-12


def test_pricing_table_has_reconciled_haiku():
    # Regression guard: the old drift ($0.80 in optimizer) must stay reconciled.
    assert PRICING["claude-haiku-4.5"].input_per_m == 1.00
    assert isinstance(PRICING["claude-haiku-4.5"], Price)
