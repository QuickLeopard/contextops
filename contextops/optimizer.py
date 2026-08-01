"""Core optimization logic: token counting + cache-aware reordering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import tiktoken

from contextops.models import OptimizationResult, Prompt, Section
from contextops.pricing import CACHE_READ_DISCOUNT, price_per_million

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


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration knobs for cache-aware optimization.

    Encapsulates the constants that control stability ordering, cache-hit
    estimation, and cost modeling. Callers can inject a custom instance to
    experiment with different pricing / hit-rate assumptions without mutating
    global module state.
    """

    stability_order: dict[Section, int] = field(
        default_factory=lambda: dict(_STABILITY_ORDER)
    )
    baseline_hit_rate: float = _BASELINE_HIT_RATE
    optimized_hit_rate: float = _OPTIMIZED_HIT_RATE
    section_bonuses: tuple[tuple[int, float], ...] = ((2, 0.18), (4, 0.08), (6, 0.02))
    default_section_bonus: float = 0.02
    cache_read_discount: float = CACHE_READ_DISCOUNT
    price_per_million: Callable[[str], float] = price_per_million

    def bonus_for_rank(self, rank: int) -> float:
        """Return the cache-hit bonus accrued for a section at `rank`."""
        for max_rank, bonus in self.section_bonuses:
            if rank <= max_rank:
                return bonus
        return self.default_section_bonus


DEFAULT_CONFIG = OptimizerConfig()


@lru_cache(maxsize=32)
def _get_encoding(model: str):
    """Pick the right tokenizer. Falls back to cl100k_base for unknown models.

    Memoized — `tiktoken.encoding_for_model`/`get_encoding` load and parse a
    BPE rank file from disk on every call otherwise, which is wasteful when
    `count_tokens` is invoked per-section across many prompts/eval rows.
    """
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


def _sections_tokens(
    sections: list[tuple[Section, str]], model: str, cache: dict[str, int]
) -> int:
    """Total tokens across `sections`, memoized by content in `cache`.

    A pure reorder never mutates section content — only its position — so
    the original and reordered section lists share identical strings. This
    lets `optimize()` avoid re-running the tiktoken encoder twice on the
    same content while still summing each list independently (so a future
    change that *does* mutate content would still be reflected correctly).
    """
    total = 0
    for _, content in sections:
        if content not in cache:
            cache[content] = count_tokens(content, model)
        total += cache[content]
    return total


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _reorder_sections(
    p: Prompt, config: OptimizerConfig = DEFAULT_CONFIG
) -> list[tuple[Section, str]]:
    """Sort sections by stability: stable (system) first, variable (query) last."""
    sections = p.sections()
    if p.goal == "quality":
        # Quality mode preserves the user's input order.
        return sections
    # balanced / cache_friendly: canonical order.
    return sorted(sections, key=lambda s: config.stability_order[s[0]])


def estimate_cache_hit(
    p: Prompt, *, reordered: bool, config: OptimizerConfig | None = None
) -> float:
    """Estimate cache hit rate based on section ordering.

    Heuristic: hit rate = baseline + bonus for each section inside the
    longest *stable-ordered* prefix — i.e. sections whose stability rank is
    non-decreasing from the start of the render order. A section that lands
    out of its canonical position breaks the cached prefix; sections after
    that point earn no bonus, mirroring how a broken prefix invalidates the
    provider's KV cache for everything that follows it.

    Uses `_bench_render_order` (set by `reorder()` / the bench harness) as
    the actual render order when present, since that reflects what the LLM
    provider sees; falls back to `p.sections()` declaration order otherwise.
    """
    cfg = config or DEFAULT_CONFIG
    if not reordered or p.goal == "quality":
        return cfg.baseline_hit_rate

    section_map = dict(p.sections())
    order = getattr(p, "_bench_render_order", None) or [s[0] for s in p.sections()]

    bonus = 0.0
    max_rank_so_far = -1
    for sec in order:
        if sec not in section_map:
            continue
        rank = cfg.stability_order[sec]
        if rank < max_rank_so_far:
            # Order broke here: the cached prefix ends — stop accruing bonus.
            break
        max_rank_so_far = rank
        bonus += cfg.bonus_for_rank(rank)

    return min(cfg.optimized_hit_rate, cfg.baseline_hit_rate + bonus)


def reorder(p: Prompt, config: OptimizerConfig | None = None) -> Prompt:
    """Return a NEW Prompt with sections reordered for cache friendliness.

    The original `history` (list of HistoryMessage) is preserved as-is.
    Sets `_bench_render_order` so callers that respect it (e.g. the bench
    harness) render sections in the new order instead of declaration order.
    """
    cfg = config or DEFAULT_CONFIG
    new = p.model_copy(deep=True)
    original_history = list(new.history)
    new_sections = _reorder_sections(new, cfg)
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
    new._bench_render_order = [s[0] for s in new_sections]  # type: ignore[attr-defined]
    return new


def optimize(
    p: Prompt, config: OptimizerConfig | None = None
) -> OptimizationResult:
    """Run full optimization pass: count tokens, reorder, estimate savings.

    Returns OptimizationResult — the original sections, the optimized sections,
    and the metrics.
    """
    cfg = config or DEFAULT_CONFIG
    original_sections = p.sections()
    optimized = reorder(p, cfg)
    optimized_sections = optimized.sections()

    token_cache: dict[str, int] = {}
    original_tokens = _sections_tokens(original_sections, p.model, token_cache)
    optimized_tokens = _sections_tokens(optimized_sections, p.model, token_cache)

    hit_rate = estimate_cache_hit(optimized, reordered=True, config=cfg)
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
    price_per_m = cfg.price_per_million(p.model)
    cost_per_call_baseline = (original_tokens / 1_000_000) * price_per_m
    cost_per_call_optimized = (
        (optimized_tokens / 1_000_000) * price_per_m * (1 - hit_rate)
        + (optimized_tokens / 1_000_000) * price_per_m * hit_rate * cfg.cache_read_discount
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