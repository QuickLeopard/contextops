"""Tests for the optimizer. Run with `pytest`."""

from contextops.models import Prompt
from contextops.optimizer import count_tokens, estimate_cache_hit, optimize, reorder


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


def test_prompt_sections_respects_render_order():
    """`sections()` yields in declaration order by default, and in `render_order` when set."""
    p = Prompt(system="S", query="Q")
    # Default: declaration order.
    assert [s[0] for s in p.sections()] == ["system", "query"]
    # When render_order is set, sections() yields in that order.
    p.render_order = ["query", "system"]
    assert [s[0] for s in p.sections()] == ["query", "system"]
    # render_order referencing absent sections is ignored.
    p.render_order = ["query", "documents", "system"]
    assert [s[0] for s in p.sections()] == ["query", "system"]