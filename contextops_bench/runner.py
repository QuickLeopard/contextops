"""Benchmark runner — drives prompts through ContextOps + a real LLM provider."""

from __future__ import annotations

import csv
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from contextops.models import Prompt
from contextops.optimizer import reorder, count_tokens
from contextops.render import render_prompt as _render_public, split_prompt as _split_public
from contextops_bench.clients import BenchResult, CompletionResponse


# Sections treated as "stable" content (sent in the system message for
# Anthropic cache_control). Anything not in this set is treated as variable
# content (sent in the user message). Re-exported from the public module.
STABLE_SECTIONS: frozenset[str] = frozenset({"system", "tools", "role"})


def _render_prompt(p: Prompt) -> tuple[str, list[str]]:
    """Delegate to the public `contextops.render.render_prompt`."""
    return _render_public(p)


def _split_prompt(p: Prompt) -> tuple[str, str, list[str]]:
    """Delegate to the public `contextops.render.split_prompt`."""
    split = _split_public(p)
    return split.system, split.user, split.section_order


def _reverse_prompt(p: Prompt) -> Prompt:
    """Return a new Prompt rendered in WORST-CASE order for cache.

    Worst case = query/history first, system/tools last. This is what a naive
    prompt template often produces (e.g. putting the user's question at the top
    of the message). Sets `render_order` so `_render_prompt` emits this order.
    """
    new = p.model_copy(deep=True)
    sections = new.sections()
    if not sections:
        return new
    current_order = [s[0] for s in sections]
    new.render_order = list(reversed(current_order))
    return new


def run_one(
    prompt: Prompt,
    *,
    prompt_id: int,
    client,
    use_optimized: bool = True,
    reverse_baseline: bool = True,
) -> BenchResult:
    """Run a single prompt through the chosen client.

    `use_optimized=True` runs `reorder(p)` to put stable sections first.
    `use_optimized=False` with `reverse_baseline=True` runs `_reverse_prompt(p)`
    to simulate the WORST ordering (query first, system last) — this is what
    naive prompt construction often produces and is the real-world counterfactual.
    """
    if use_optimized:
        target = reorder(prompt)
    elif reverse_baseline:
        target = _reverse_prompt(prompt)
    else:
        target = prompt
    prompt_str, section_order = _render_prompt(target)
    system_str, user_str, _ = _split_prompt(target)

    provider = client.PROVIDER
    # OPTIMIZED arm: send stable sections (system, tools, role) as a separate
    # system message with cache_control. Variable sections go in a user message.
    # This is what a well-designed prompt template does in production.
    #
    # BASELINE arm: send EVERYTHING in a single user message with no system
    # field. This is the "naive" prompt template that doesn't separate stable
    # from variable content — what most teams accidentally ship. The provider
    # has no cache_control marker to anchor on, so cache hits depend entirely
    # on whether the prefix matches a previous call.
    if use_optimized and getattr(client, "supports_split_messages", False) and system_str:
        messages = [{"role": "user", "content": user_str or "(empty)"}]
        system_arg = system_str
    else:
        messages = [{"role": "user", "content": prompt_str or "(empty)"}]
        system_arg = None

    try:
        resp: CompletionResponse = client.complete(
            model=target.model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
            system=system_arg,
        )
        return BenchResult(
            prompt_id=prompt_id,
            model=target.model,
            provider=provider,
            use_optimized=use_optimized,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cached_tokens=resp.cached_tokens,
            cost_usd=resp.cost_usd,
            latency_ms=resp.raw.get("_latency_ms", 0.0),  # optional override
            section_order=section_order,
        )
    except Exception as e:
        return BenchResult(
            prompt_id=prompt_id,
            model=target.model,
            provider=provider,
            use_optimized=use_optimized,
            prompt_tokens=count_tokens(prompt_str, target.model),
            completion_tokens=0,
            cost_usd=0.0,
            latency_ms=0.0,
            error=str(e)[:200],
            section_order=section_order,
        )


def _reset_client_state(client) -> None:
    """Reset a client's in-memory cache state in place.

    For stateful clients like EchoClient: call `reset()` to wipe cache.
    For stateless HTTP clients (Ollama, OpenRouter): no-op (they have no
    in-memory state; provider-side cache is global to the API key regardless).
    """
    if hasattr(client, "reset"):
        client.reset()


def run_batch(
    prompts: Iterable[Prompt],
    *,
    client,
    parallel: int = 1,
    label: str = "bench",
    on_progress: Callable[[int, int], None] | None = None,
    cache_warm: bool = True,
) -> list[BenchResult]:
    """Run a batch of prompts. Each prompt is run TWICE — once optimized, once
    baseline — so the A/B comparison is on the same content, just rendered in
    two orderings. Total result count is 2 * len(prompts).

    `cache_warm=True` (default): runs ALL optimized first to warm the cache, then
    ALL baseline with a FRESH client (no in-memory state leakage between phases).
    This simulates a real deployment: new layout deployed, used for a while,
    then someone tries the old layout for comparison. Result: optimized wins
    because its cache is warm and stable prefix is in the same position.

    `cache_warm=False`: alternate optimized/baseline, single client.
    Less realistic but isolates the effect of reorder on identical prompt strings
    with a shared cache.
    """
    items = list(prompts)
    n = len(items)

    # Each prompt produces 2 results: (prompt_id, item, use_optimized)
    jobs: list[tuple[int, Prompt, bool]] = []
    for i, p in enumerate(items):
        jobs.append((i, p, True))   # optimized version
        jobs.append((i, p, False))  # baseline version

    if cache_warm:
        # Run all optimized first (cache warms on stable prefix), then all baseline
        # with the client's in-memory state reset. The reset just clears in-memory
        # state for stateful clients like EchoClient; for stateless HTTP clients the
        # server-side provider cache is global to the API key and persists either way.
        client_opt = client
        client_base = client
        _reset_client_state(client_base)
        optimized_jobs = [(idx, p, True, client_opt) for (idx, p, use_opt) in jobs if use_opt]
        baseline_jobs = [(idx, p, False, client_base) for (idx, p, use_opt) in jobs if not use_opt]
        jobs_with_client = optimized_jobs + baseline_jobs
    else:
        # Alternate: optimized, baseline, optimized, baseline ...
        jobs_with_client = [
            (idx, p, use_opt, client) for (idx, p, use_opt) in jobs
        ]

    results: list[BenchResult | None] = [None] * len(jobs_with_client)
    completed = 0

    def _work(slot: int, prompt_id: int, p: Prompt, use_opt: bool, c) -> tuple[int, BenchResult]:
        return slot, run_one(p, prompt_id=prompt_id, client=c, use_optimized=use_opt)

    if parallel <= 1:
        for slot, (prompt_id, p, use_opt, c) in enumerate(jobs_with_client):
            i, r = _work(slot, prompt_id, p, use_opt, c)
            results[i] = r
            completed += 1
            if on_progress:
                on_progress(completed, n)
    else:
        # For parallel, alternate so each prompt's two jobs may run concurrently
        # on different workers — better throughput, but cache_warm semantics are
        # weaker (no clean warm-then-cold split).
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(_work, slot, prompt_id, p, use_opt, c)
                for slot, (prompt_id, p, use_opt, c) in enumerate(jobs_with_client)
            ]
            for fut in as_completed(futures):
                i, r = fut.result()
                results[i] = r
                completed += 1
                if on_progress:
                    on_progress(completed, n)

    return [r for r in results if r is not None]


def save_csv(results: list[BenchResult], path: str | Path) -> None:
    """Persist all observations to CSV.

    Raises ValueError on empty input — callers should never pass empty (it would
    produce a headerless file indistinguishable from "no rows written").
    """
    if not results:
        raise ValueError("no results to save")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        fieldnames = list(asdict(results[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            d["section_order"] = "|".join(d["section_order"])
            writer.writerow(d)


def summarize(
    results: list[BenchResult],
    *,
    exclude_ids: set[int] | None = None,
) -> dict:
    """Compute summary stats. Group by `use_optimized`.

    Each prompt runs twice (once optimized, once baseline) so the A/B is on the
    same content rendered in two orderings. `exclude_ids` is a set of prompt_ids
    to drop from headline stats (e.g. edge cases that are deliberately degenerate
    — empty prompts, 100k-token blobs). Excluded prompt_ids are dropped from BOTH
    arms so the A/B stays paired.
    """
    n = len(results)
    if n == 0:
        return {"optimized": {}, "baseline": {}, "delta": {}}

    # Filter out excluded ids (edge cases) before computing headline stats.
    excluded = exclude_ids or set()
    optimized_all = [r for r in results if r.use_optimized]
    baseline_all = [r for r in results if not r.use_optimized]

    optimized = [r for r in optimized_all if r.prompt_id not in excluded]
    baseline = [r for r in baseline_all if r.prompt_id not in excluded]
    excluded_optimized = [r for r in optimized_all if r.prompt_id in excluded]
    excluded_baseline = [r for r in baseline_all if r.prompt_id in excluded]

    summary = {
        "optimized": _stats(optimized),
        "baseline": _stats(baseline),
        "excluded": {
            "optimized": _stats(excluded_optimized),
            "baseline": _stats(excluded_baseline),
            "count": len(excluded),
        },
        "delta": {},
    }

    if summary["optimized"] and summary["baseline"]:
        summary["delta"] = {
            "cache_hit_rate_delta": round(
                summary["optimized"]["cache_hit_rate_mean"]
                - summary["baseline"]["cache_hit_rate_mean"],
                3,
            ),
            "cost_per_call_delta_usd": round(
                summary["optimized"]["cost_usd_per_call"]
                - summary["baseline"]["cost_usd_per_call"],
                6,
            ),
            "prompt_tokens_delta_mean": round(
                summary["optimized"]["prompt_tokens_mean"]
                - summary["baseline"]["prompt_tokens_mean"],
                1,
            ),
        }

    return summary


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Returns 0.0 for empty input.

    Uses the nearest-rank method: rank = ceil(p/100 * n), 1-indexed. For small n
    this can return the max (e.g. p95 of 10 values ranks ceil(9.5)=10 → the max),
    which is the correct nearest-rank behaviour rather than a bug.
    """
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(s)))  # 1-indexed
    return s[min(rank, len(s)) - 1]


def _stats(rows: list[BenchResult]) -> dict:
    """Aggregate stats. Always returns the same keys (NaN-ish defaults for empty)."""
    if not rows:
        return {
            "n": 0, "errors": 0,
            "prompt_tokens_mean": 0.0, "prompt_tokens_p50": 0.0,
            "completion_tokens_mean": 0.0, "cached_tokens_mean": 0.0,
            "cost_usd_total": 0.0, "cost_usd_per_call": 0.0,
            "latency_ms_p50": 0.0, "latency_ms_p95": 0.0,
            "cache_hit_rate_mean": 0.0, "cache_hit_rate_p50": 0.0,
        }
    prompt_tokens = [r.prompt_tokens for r in rows]
    completion_tokens = [r.completion_tokens for r in rows]
    cached_tokens = [r.cached_tokens for r in rows]
    costs = [r.cost_usd for r in rows]
    latencies = [r.latency_ms for r in rows if r.latency_ms > 0]
    errors = [r for r in rows if r.error]
    cache_hits = [
        r.cached_tokens / r.prompt_tokens
        for r in rows if r.prompt_tokens > 0
    ]
    return {
        "n": len(rows),
        "errors": len(errors),
        "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 1),
        "prompt_tokens_p50": round(statistics.median(prompt_tokens), 1),
        "completion_tokens_mean": round(statistics.mean(completion_tokens), 1),
        "cached_tokens_mean": round(statistics.mean(cached_tokens), 1),
        "cost_usd_total": round(sum(costs), 6),
        "cost_usd_per_call": round(statistics.mean(costs), 6),
        "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_ms_p95": round(_percentile(latencies, 95), 1),
        "cache_hit_rate_mean": round(statistics.mean(cache_hits), 3) if cache_hits else 0.0,
        "cache_hit_rate_p50": round(statistics.median(cache_hits), 3) if cache_hits else 0.0,
    }


def render_summary(summary: dict, label: str) -> str:
    """Render summary as a fixed-width table."""
    lines = [f"=== {label} ==="]
    for side in ("optimized", "baseline"):
        s = summary.get(side, {})
        if not s or s.get("n", 0) == 0:
            continue
        lines.append(f"\n[{side.upper()}] (n={s.get('n', 0)}, errors={s.get('errors', 0)})")
        lines.append(f"  prompt tokens:     mean={s['prompt_tokens_mean']:>8}  p50={s['prompt_tokens_p50']:>8}")
        lines.append(f"  completion tokens: mean={s['completion_tokens_mean']:>8}")
        lines.append(f"  cached tokens:     mean={s['cached_tokens_mean']:>8}")
        lines.append(f"  cache hit rate:    mean={s['cache_hit_rate_mean']:.1%}  p50={s['cache_hit_rate_p50']:.1%}")
        lines.append(f"  cost / call:       ${s['cost_usd_per_call']:.6f}   total=${s['cost_usd_total']:.4f}")
        lines.append(f"  latency:           p50={s['latency_ms_p50']:.0f}ms  p95={s['latency_ms_p95']:.0f}ms")

    excluded = summary.get("excluded", {})
    if excluded.get("count", 0) > 0:
        lines.append(f"\n[EXCLUDED] {excluded['count']} edge-case prompt(s) excluded from headline stats")
        lines.append(f"  optimized:  n={excluded['optimized'].get('n', 0)}")
        lines.append(f"  baseline:   n={excluded['baseline'].get('n', 0)}")

    d = summary.get("delta", {})
    if d:
        lines.append("\n[DELTA] (optimized − baseline)")
        lines.append(f"  cache hit rate:    {d['cache_hit_rate_delta']:+.1%}")
        lines.append(f"  cost / call:       {d['cost_per_call_delta_usd']:+.6f}")
        lines.append(f"  prompt tokens:     {d['prompt_tokens_delta_mean']:+.1f}")
    return "\n".join(lines)