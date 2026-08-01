"""Tests for the optimizer. Run with `pytest`."""

from typing import get_args

from contextops.models import Prompt, Section
from contextops.optimizer import (
    DEFAULT_CONFIG,
    OptimizerConfig,
    count_tokens,
    estimate_cache_hit,
    optimize,
    reorder,
)


def test_count_tokens_basic():
    assert count_tokens("hello world", "gpt-4o") > 0
    assert count_tokens("", "gpt-4o") == 0


def test_reorder_stable_first():
    p = Prompt(
        query="what?",
        documents="doc",
        system="sys",
        tools="tools",
        history=[{"role": "user", "content": "hi"}],
    )
    optimized = reorder(p)
    sections = [s[0] for s in optimized.sections()]
    # system/tools should come before documents/history/query
    assert sections.index("system") < sections.index("query")
    assert sections.index("tools") < sections.index("documents")


def test_reorder_quality_mode_keeps_input_order():
    # Build Prompt with goal='quality' so reorder() preserves the declaration order
    # emitted by sections() — i.e. system, tools, role, context, documents, history, query.
    p = Prompt(
        query="q",
        documents="d",
        system="s",
        goal="quality",
    )
    optimized = reorder(p)
    sections = [s[0] for s in optimized.sections()]
    # declaration order from Prompt fields, regardless of init kwargs order
    assert sections == ["system", "documents", "query"]


def test_optimize_returns_result():
    p = Prompt(
        query="hello",
        system="you are helpful",
        tools="[]",
        model="gpt-4o",
    )
    r = optimize(p)
    assert r.optimized_tokens > 0
    assert r.estimated_cache_hit_rate > 0


def test_cache_hit_higher_when_reordered():
    p_bad = Prompt(
        query="q",
        documents="d",
        system="s",
        tools="t",
        history=[{"role": "user", "content": "h"}],
    )
    p_good = reorder(p_bad)
    hit_bad = estimate_cache_hit(p_bad, reordered=False)
    hit_good = estimate_cache_hit(p_good, reordered=True)
    assert hit_good > hit_bad


def test_empty_prompt_safe():
    p = Prompt()
    r = optimize(p)
    assert r.optimized_tokens == 0
    # Empty prompt = baseline hit rate (no sections to optimize).
    assert r.estimated_cache_hit_rate == 0.05


def test_cache_hit_is_position_aware():
    """Regression test: estimate_cache_hit must penalize a broken render
    order instead of summing a flat bonus per section type regardless of
    position. Uses `_bench_render_order` (the mechanism the bench harness
    uses to simulate worst-case ordering) to construct a prompt whose
    section TYPES are identical but whose actual render order is reversed.
    """
    p = Prompt(system="s", tools="t", documents="d", query="q")
    canonical_hit = estimate_cache_hit(p, reordered=True)

    reversed_p = p.model_copy(deep=True)
    reversed_p._bench_render_order = ["query", "documents", "tools", "system"]
    reversed_hit = estimate_cache_hit(reversed_p, reordered=True)

    assert reversed_hit < canonical_hit


def test_section_literal_order_matches_stability_order():
    """Regression test: Section literal ordering should match canonical
    stability order so type-level iteration is not surprising.
    """
    from contextops.optimizer import _STABILITY_ORDER

    literal_order = list(get_args(Section))
    assert literal_order == sorted(
        literal_order, key=lambda s: _STABILITY_ORDER[s]
    )


def test_default_config_matches_module_constants():
    assert DEFAULT_CONFIG.baseline_hit_rate == 0.05
    assert DEFAULT_CONFIG.optimized_hit_rate == 0.78
    assert DEFAULT_CONFIG.cache_read_discount == 0.1
    assert DEFAULT_CONFIG.stability_order["system"] < DEFAULT_CONFIG.stability_order["query"]


def test_optimizer_config_injection_changes_hit_rate():
    p = Prompt(system="s", tools="t", documents="d", query="q")
    custom = OptimizerConfig(baseline_hit_rate=0.1, optimized_hit_rate=0.95)
    hit = estimate_cache_hit(p, reordered=True, config=custom)
    assert hit >= custom.baseline_hit_rate
    assert hit <= custom.optimized_hit_rate


def test_optimizer_config_custom_stability_order_changes_reorder():
    p = Prompt(system="s", documents="d", query="q")
    custom = OptimizerConfig(
        stability_order={"query": 0, "documents": 1, "system": 2}
    )
    r = reorder(p, config=custom)
    # `Prompt.sections()` is always emitted in field-declaration order, so the
    # actual optimized render order is tracked on `_bench_render_order`.
    assert r._bench_render_order == ["query", "documents", "system"]


def test_optimize_uses_injected_config_for_pricing_and_discount():
    p = Prompt(system="s" * 1000, query="q", model="gpt-4o")
    custom = OptimizerConfig(
        price_per_million=lambda _m: 10.0,
        cache_read_discount=0.5,
    )
    r = optimize(p, config=custom)
    # Savings should be non-negative and driven by the injected price/discount.
    assert r.estimated_cost_savings_usd >= 0