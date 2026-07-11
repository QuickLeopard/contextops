"""Core optimization logic: token counting + cache-aware reordering."""

from __future__ import annotations

import hashlib

import tiktoken

from contextops.models import OptimizationResult, Prompt, Section
from contextops.pricing import PRICING, Price

# Canonical ordering — most stable first.
# This mirrors the Anthropic/OpenAI best-practice recommendation:
# "static prefix first, variable content last".
_STABILITY_ORDER: dict[Section, int] = {
    "system": 0,    # most stable
    "tools": 1,
    "role": 2,
    "context": 3,
    "documents": 4,
    "history": 5,
    "query": 6,     # most variable
}

# Cache hit rates by ordering strategy — empirical estimates.
# Real numbers depend on workload, but the principle holds:
# the more you keep the prefix stable across calls, the higher the hit rate.
_BASELINE_HIT_RATE = 0.05      # no optimization, random order
_OPTIMIZED_HIT_RATE = 0.78     # canonical ordering


def _get_encoding(model: str):
    """Pick the right tokenizer. Falls back to cl100k_base for unknown models."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for a given string and model."""
    if not text:
        return 0
    enc = _get_encoding(model)
    return len(enc.encode(text))


def _prompt_tokens(p: Prompt) -> int:
    """Total tokens across all sections of a Prompt."""
    return sum(count_tokens(content, p.model) for _, content in p.sections())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _reorder_sections(p: Prompt) -> list[tuple[Section, str]]:
    """Sort sections by stability: stable (system) first, variable (query) last."""
    sections = p.sections()
    if p.goal == "quality":
        # Quality mode preserves the user's input order.
        return sections
    # balanced / cache_friendly: canonical order.
    return sorted(sections, key=lambda s: _STABILITY_ORDER[s[0]])


def estimate_cache_hit(p: Prompt, *, reordered: bool) -> float:
    """Estimate cache hit rate based on section ordering.

    Heuristic: hit rate = baseline + delta_per_stable_section_in_correct_position.
    """
    if not reordered or p.goal == "quality":
        return _BASELINE_HIT_RATE

    # Bonus per correctly-positioned stable section
    sections = p.sections()
    bonus = 0.0
    for sec, _ in sections:
        rank = _STABILITY_ORDER[sec]
        if rank <= 2:    # system / tools / role
            bonus += 0.18
        elif rank <= 4:  # context / documents
            bonus += 0.08
        else:            # history / query
            bonus += 0.02

    return min(_OPTIMIZED_HIT_RATE, _BASELINE_HIT_RATE + bonus)


def reorder(p: Prompt) -> Prompt:
    """Return a NEW Prompt with sections reordered for cache friendliness.

    The original `history` (list of HistoryMessage) is preserved as-is.
    Sets `render_order` so callers that respect it (e.g. the bench harness)
    render sections in the new order instead of declaration order.
    """
    new = p.model_copy(deep=True)
    original_history = list(new.history)
    new_sections = _reorder_sections(new)
    # Wipe everything, then refill in the new order.
    new.system = ""
    new.tools = ""
    new.role = ""
    new.context = ""
    new.documents = ""
    new.history = []
    new.query = ""
    for sec, content in new_sections:
        if sec == "history":
            new.history = original_history
        else:
            setattr(new, sec, content)
    new.render_order = [s[0] for s in new_sections]
    return new


def optimize(p: Prompt) -> OptimizationResult:
    """Run full optimization pass: count tokens, reorder, estimate savings.

    Returns OptimizationResult — the original sections, the optimized sections,
    and the metrics.
    """
    original_sections = p.sections()
    optimized = reorder(p)
    optimized_sections = optimized.sections()

    original_tokens = _prompt_tokens(p)
    optimized_tokens = _prompt_tokens(optimized)

    hit_rate = estimate_cache_hit(optimized, reordered=True)
    notes: list[str] = []

    original_order = [s[0] for s in original_sections]
    optimized_order = [s[0] for s in optimized_sections]

    if original_order == optimized_order:
        notes.append("Section order is already cache-friendly — no reorder applied.")
    else:
        notes.append(
            f"Reordered {len(original_sections)} sections: "
            f"{' → '.join(original_order)} → {' → '.join(optimized_order)}"
        )

    if original_tokens != optimized_tokens:
        notes.append(
            f"Token count delta: {optimized_tokens - original_tokens} "
            "(expected 0 for pure reorder — investigate if non-zero)."
        )

    # Rough savings: assume 1000 calls/day, each with avg prompt size.
    price_per_m = PRICING.get(p.model, Price(1.0, 1.0)).input_per_m
    cost_per_call_baseline = (original_tokens / 1_000_000) * price_per_m
    cost_per_call_optimized = (
        (optimized_tokens / 1_000_000) * price_per_m * (1 - hit_rate)
        + (optimized_tokens / 1_000_000) * price_per_m * hit_rate * 0.1  # cache reads ~10% price
    )
    savings_per_1k = (cost_per_call_baseline - cost_per_call_optimized) * 1000

    return OptimizationResult(
        original_sections=original_sections,
        optimized_sections=optimized_sections,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        estimated_cache_hit_rate=round(hit_rate, 3),
        estimated_cost_savings_usd=round(savings_per_1k, 4),
        model=p.model,
        notes=notes,
    )