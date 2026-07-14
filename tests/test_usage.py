"""Tests for the shared provider-usage normalizer."""

from contextops_bench.usage import UsageMetrics, extract_usage


def test_extract_usage_anthropic_native():
    raw = {"usage": {
        "input_tokens": 1500,
        "output_tokens": 50,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 500,
    }}
    m = extract_usage(raw, "direct_anthropic")
    assert m.prompt_tokens == 1500
    assert m.completion_tokens == 50
    assert m.cached_tokens == 1000
    assert m.cache_creation_tokens == 500


def test_extract_usage_openrouter_all_fields():
    raw = {"usage": {
        "prompt_tokens": 2000,
        "completion_tokens": 100,
        "prompt_tokens_details": {
            "cached_tokens": 800,
            "cache_write_tokens": 200,
        },
    }}
    m = extract_usage(raw, "openrouter")
    assert m.prompt_tokens == 2000
    assert m.completion_tokens == 100
    assert m.cached_tokens == 800
    assert m.cache_creation_tokens == 200


def test_extract_usage_openrouter_cache_read_fallback():
    """OpenRouter sometimes surfaces cache reads under cache_read_input_tokens."""
    raw = {"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 10,
        "cache_read_input_tokens": 400,
    }}
    m = extract_usage(raw, "openrouter")
    assert m.cached_tokens == 400


def test_extract_usage_ollama_no_cache():
    """Ollama (local) typically reports no cache fields → zeros."""
    raw = {"usage": {"prompt_tokens": 500, "completion_tokens": 20}}
    m = extract_usage(raw, "ollama")
    assert m.prompt_tokens == 500
    assert m.completion_tokens == 20
    assert m.cached_tokens == 0
    assert m.cache_creation_tokens == 0


def test_extract_usage_missing_usage_block():
    """No usage block at all → all zeros, no crash."""
    m = extract_usage({}, "openrouter")
    assert m.prompt_tokens == 0
    assert m.completion_tokens == 0


def test_extract_usage_handles_non_int_values():
    """Non-numeric values should coerce to 0, not crash."""
    raw = {"usage": {"prompt_tokens": "oops", "completion_tokens": None}}
    m = extract_usage(raw, "ollama")
    assert m.prompt_tokens == 0
    assert m.completion_tokens == 0


def test_usage_metrics_cost():
    """UsageMetrics.cost() delegates to the shared pricing module."""
    m = UsageMetrics(
        prompt_tokens=1_000_000, completion_tokens=0,
        cached_tokens=0, cache_creation_tokens=0,
    )
    # gpt-4o-mini is $0.15/M input → 1M tokens = $0.15
    assert abs(m.cost(model="gpt-4o-mini") - 0.15) < 1e-9


def test_cache_creation_tokens_not_discarded():
    """The whole point of #5: cache_creation must survive into the metrics."""
    raw = {"usage": {
        "input_tokens": 1000, "output_tokens": 10,
        "cache_creation_input_tokens": 300,
    }}
    m = extract_usage(raw, "direct_anthropic")
    assert m.cache_creation_tokens == 300  # previously discarded after cost math
