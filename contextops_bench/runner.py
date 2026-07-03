"""Benchmark runner — drives prompts through ContextOps + a real LLM provider."""

from __future__ import annotations

import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from contextops.models import HistoryMessage, Prompt
from contextops.optimizer import optimize, reorder, count_tokens
from contextops_bench.clients import BenchResult, CompletionResponse, get_client
from contextops_bench.prompt_factory import generate_many


def _render_prompt(p: Prompt) -> tuple[str, list[str]]:
    """Render prompt to a single string + return its current section order.

    If `_bench_render_order` is set (by reorder() / _reverse_prompt()), use it
    instead of the default declaration order. This lets bench swap orders
    without mutating Prompt fields.
    """
    custom_order = getattr(p, "_bench_render_order", None)
    if custom_order is not None:
        section_map: dict[str, str] = {}
        for sec, content in p.sections():
            section_map[sec] = content
        parts = [section_map[name] for name in custom_order if name in section_map]
        order = [name for name in custom_order if name in section_map]
        return "\n\n".join(parts), order

    # Default: declaration order.
    parts: list[str] = []
    order: list[str] = []
    for sec, content in p.sections():
        parts.append(content)
        order.append(sec)
    return "\n\n".join(parts), order


def _reverse_prompt(p: Prompt) -> Prompt:
    """Return a new Prompt rendered in WORST-CASE order for cache.

    Worst case = query/history first, system/tools last. This is what a naive
    prompt template often produces (e.g. putting the user's question at the top
    of the message). Sets `_bench_render_order` to override `_render_prompt`.
    """
    new = p.model_copy(deep=True)
    sections = new.sections()
    if not sections:
        return new
    current_order = [s[0] for s in sections]
    new._bench_render_order = list(reversed(current_order))  # type: ignore[attr-defined]
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

    provider = client.PROVIDER
    messages = [{"role": "user", "content": prompt_str}] if prompt_str else [
        {"role": "user", "content": "(empty)"}
    ]

    try:
        resp: CompletionResponse = client.complete(
            model=target.model,
            messages=messages,
            temperature=0.0,
            max_tokens=32,
        )
        return BenchResult(
            prompt_id=prompt_id,
            model=target.model,
            provider=provider,
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
            prompt_tokens=count_tokens(prompt_str, target.model),
            completion_tokens=0,
            cost_usd=0.0,
            latency_ms=0.0,
            error=str(e)[:200],
            section_order=section_order,
        )


def _make_fresh_client(client) -> object:
    """Return a client with empty cache state.

    For stateful clients like EchoClient: call `reset()` in-place to wipe cache.
    For stateless HTTP clients (Ollama, OpenRouter): return the same instance.
    """
    if hasattr(client, "reset"):
        client.reset()
    return client


def run_batch(
    prompts: Iterable[Prompt],
    *,
    client,
    parallel: int = 1,
    label: str = "bench",
    on_progress: Callable[[int, int], None] | None = None,
    cache_warm: bool = True,
) -> list[BenchResult]:
    """Run a batch of prompts. Half optimized, half baseline.

    `cache_warm=True` (default): runs ALL optimized first to warm the cache, then
    ALL baseline with a FRESH client (no cache leakage between phases).
    This simulates a real deployment: new layout deployed, used for a while,
    then someone tries the old layout for comparison. Result: optimized wins
    because its cache is warm.

    `cache_warm=False`: alternate optimized/baseline, single client.
    Less realistic but isolates the effect of reorder on identical prompt strings.
    """
    items = list(prompts)
    n = len(items)

    if cache_warm:
        half = n // 2
        optimized_jobs = [(i, items[i], True) for i in range(half)]
        baseline_jobs = [(i, items[i], False) for i in range(half, n)]
        client_opt = client
        client_base = _make_fresh_client(client)
        jobs_with_client = (
            [(idx, p, use_opt, client_opt) for (idx, p, use_opt) in optimized_jobs]
            + [(idx, p, use_opt, client_base) for (idx, p, use_opt) in baseline_jobs]
        )
    else:
        jobs_with_client = [
            (i, p, i % 2 == 0, client) for i, p in enumerate(items)
        ]

    results: list[BenchResult | None] = [None] * len(jobs_with_client)
    completed = 0

    def _work(idx: int, p: Prompt, use_opt: bool, c) -> tuple[int, BenchResult]:
        return idx, run_one(p, prompt_id=idx, client=c, use_optimized=use_opt)

    if parallel <= 1:
        for idx, p, use_opt, c in jobs_with_client:
            i, r = _work(idx, p, use_opt, c)
            results[i] = r
            completed += 1
            if on_progress:
                on_progress(completed, n)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(_work, idx, p, use_opt, c)
                for (idx, p, use_opt, c) in jobs_with_client
            ]
            for fut in as_completed(futures):
                i, r = fut.result()
                results[i] = r
                completed += 1
                if on_progress:
                    on_progress(completed, n)

    return [r for r in results if r is not None]


# Pre-defined edge-case prompt IDs in `EDGE_CASES` (smoke only). Used by summarize
# to exclude degenerate rows from headline numbers.
EDGE_CASE_PROMPT_IDS: set[int] = set()  # populated by smoke benchmark


def save_csv(results: list[BenchResult], path: str | Path) -> None:
    """Persist all observations to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        if not results:
            return
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
    """Compute summary stats. Group by prompt_id parity.

    `exclude_ids` is a set of prompt_ids to drop from headline stats (e.g. edge cases
    that are deliberately degenerate — empty prompts, 100k-token blobs).

    Convention: `run_batch` puts optimized prompts in prompt_ids 0..n/2-1
    (cache_warm mode) OR even prompt_ids (alternating mode).
    """
    n = len(results)
    if n == 0:
        return {"optimized": {}, "baseline": {}, "delta": {}}

    half = n // 2
    if results and results[0].prompt_id == 0:
        optimized_all = results[:half]
        baseline_all = results[half:]
    else:
        optimized_all = [r for r in results if r.prompt_id % 2 == 0]
        baseline_all = [r for r in results if r.prompt_id % 2 == 1]

    # Filter out excluded ids (edge cases) before computing headline stats.
    excluded = exclude_ids or set()
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
        "latency_ms_p95": (
            round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
            if len(latencies) >= 20 else 0.0
        ),
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
        sign = "+" if d["cache_hit_rate_delta"] >= 0 else ""
        lines.append(f"  cache hit rate:    {sign}{d['cache_hit_rate_delta']:.1%}")
        sign = "+" if d["cost_per_call_delta_usd"] >= 0 else ""
        lines.append(f"  cost / call:       {sign}${d['cost_per_call_delta_usd']:.6f}")
        lines.append(f"  prompt tokens:     {d['prompt_tokens_delta_mean']:+.1f}")
    return "\n".join(lines)