"""Eval entry points for v0.2.

`compare()` (v0.1) — structural comparison only.
`evaluate()` — LLM-as-judge quality scoring on a single prompt.
`evaluate_ab()` — full A/B test: run two prompts over a dataset, judge both, compare.
"""

from __future__ import annotations

from typing import Callable, Optional

from contextops.clients import default_judge
from contextops.dataset import DatasetItem
from contextops.judge import JudgeClient, score_many
from contextops.models import OptimizationResult, Prompt
from contextops.optimizer import optimize, reorder
from contextops.report import a_b_compare, aggregate, render_table


def compare(baseline: Prompt, optimized: Optional[Prompt] = None) -> dict:
    """v0.1 structural comparison. Kept for backwards compatibility."""
    baseline_result = optimize(baseline)
    if optimized is None:
        optimized = reorder(baseline)
    optimized_result = optimize(optimized)

    return {
        "baseline": _summary(baseline_result),
        "optimized": _summary(optimized_result),
        "delta": {
            "tokens": optimized_result.optimized_tokens - baseline_result.original_tokens,
            "cache_hit_rate": round(
                optimized_result.estimated_cache_hit_rate
                - baseline_result.estimated_cache_hit_rate,
                3,
            ),
            "cost_savings_per_1k_usd": round(
                optimized_result.estimated_cost_savings_usd
                - baseline_result.estimated_cost_savings_usd,
                4,
            ),
        },
    }


def _summary(r: OptimizationResult) -> dict:
    return {
        "section_order": [s[0] for s in r.optimized_sections],
        "tokens": r.optimized_tokens,
        "cache_hit_rate": r.estimated_cache_hit_rate,
        "cost_savings_per_1k_usd": r.estimated_cost_savings_usd,
        "notes": r.notes,
    }


def _render_prompt(p: Prompt) -> str:
    """Render a structured Prompt back into a single string for the LLM.

    `p.sections()` already renders `history` into a flat string, so every
    section (including history) is just appended as-is here.
    """
    return "\n\n".join(content for _, content in p.sections())


def evaluate(
    prompt: Prompt,
    *,
    run_fn: Callable[[str], str],
    dataset: list[DatasetItem],
    metrics: Optional[list[str]] = None,
    judge: Optional[JudgeClient] = None,
    judge_model: str = "gpt-4o-mini",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_render: Optional[Callable[[Prompt, DatasetItem], str]] = None,
    progress_offset: int = 0,
    progress_total: Optional[int] = None,
    max_workers: int = 1,
) -> dict:
    """Score a single prompt's responses on a dataset.

    `run_fn(prompt_str) -> response_str` is the user's LLM call.
    `on_render(prompt, item) -> str` is an OPTIONAL hook for full control over
    how each dataset row is injected. Defaults to:
      "{prompt_str}\n\nContext: {item.context}\nQuery: {item.query}"

    `progress_offset`/`progress_total` let a caller (e.g. `evaluate_ab`) fold
    this call's progress into one larger, monotonically increasing count —
    the "respond" and "judge" phases below never reset the count, they just
    keep incrementing from `progress_offset`.

    `max_workers > 1` fans judge calls out across a thread pool (see
    `judge.score_many`). The "respond" phase (`run_fn`) always stays serial
    since it's user-supplied and may not be thread-safe.

    Returns the full report dict.
    """
    metrics = metrics or ["relevance", "completeness"]
    judge = judge or default_judge()

    n_items = len(dataset)
    n_metrics = len(metrics)
    local_total = n_items * (1 + n_metrics)  # respond units + judge units
    total = progress_total if progress_total is not None else local_total

    prompt_str = _render_prompt(prompt)
    responses: list[str] = []
    for i, item in enumerate(dataset):
        if on_render is not None:
            full_prompt = on_render(prompt, item)
        else:
            full_prompt = _default_render(prompt_str, item)
        responses.append(run_fn(full_prompt))
        if on_progress:
            on_progress(progress_offset + i + 1, total, "respond")

    contexts = [item.context for item in dataset]
    queries = [item.query for item in dataset]
    expecteds = [item.expected for item in dataset]

    def _on_judge_progress(i: int, n: int, m: str) -> None:
        if on_progress:
            # `i` is score_many's own monotonic unit count (1..n_items*n_metrics);
            # continue counting from where the "respond" phase left off.
            on_progress(progress_offset + n_items + i, total, f"judge:{m}")

    scores = score_many(
        responses,
        metrics=metrics,
        judge=judge,
        model=judge_model,
        contexts=contexts,
        queries=queries,
        expecteds=expecteds,
        on_progress=_on_judge_progress,
        max_workers=max_workers,
    )
    return {
        "prompt_sections": [s[0] for s in prompt.sections()],
        "metrics_requested": metrics,
        "dataset_size": len(dataset),
        "aggregate": aggregate(scores),
        "scores": scores,
    }


def _default_render(prompt_str: str, item: DatasetItem) -> str:
    """Default: append Context + Query blocks after the rendered prompt."""
    parts: list[str] = []
    if prompt_str:
        parts.append(prompt_str)
    if item.context:
        parts.append(f"Context:\n{item.context}")
    if item.query:
        parts.append(f"Query:\n{item.query}")
    return "\n\n".join(parts)


def evaluate_ab(
    baseline: Prompt,
    optimized: Prompt,
    *,
    run_fn: Callable[[str], str],
    dataset: list[DatasetItem],
    metrics: Optional[list[str]] = None,
    judge: Optional[JudgeClient] = None,
    judge_model: str = "gpt-4o-mini",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    on_render: Optional[Callable[[Prompt, DatasetItem], str]] = None,
    max_workers: int = 1,
) -> dict:
    """Run two prompts over the same dataset, judge both, return A/B report.

    Adds structural deltas (tokens, cache hit rate) on top of quality deltas.
    `on_render(prompt, item) -> str` is forwarded to both evaluations so users
    can fully control how dataset rows are injected into the prompt string.
    `max_workers > 1` fans judge calls out across a thread pool for both
    sides (see `judge.score_many`).
    """
    metrics = metrics or ["relevance", "completeness", "faithfulness"]
    judge = judge or default_judge()

    n_items = len(dataset)
    n_metrics = len(metrics)
    side_total = n_items * (1 + n_metrics)  # one evaluate() call's total units
    total = side_total * 2  # baseline + optimized

    baseline_report = evaluate(
        baseline,
        run_fn=run_fn,
        dataset=dataset,
        metrics=metrics,
        judge=judge,
        judge_model=judge_model,
        on_progress=lambda i, n, p: on_progress(i, n, f"baseline:{p}") if on_progress else None,
        on_render=on_render,
        progress_offset=0,
        progress_total=total,
        max_workers=max_workers,
    )
    optimized_report = evaluate(
        optimized,
        run_fn=run_fn,
        dataset=dataset,
        metrics=metrics,
        judge=judge,
        judge_model=judge_model,
        on_progress=lambda i, n, p: on_progress(i, n, f"optimized:{p}") if on_progress else None,
        on_render=on_render,
        progress_offset=side_total,
        progress_total=total,
        max_workers=max_workers,
    )

    structural = compare(baseline, optimized)
    quality = a_b_compare(baseline_report["scores"], optimized_report["scores"])

    return {
        "baseline": baseline_report,
        "optimized": optimized_report,
        "structural": structural["delta"],
        "quality": quality,
        "summary_table": render_table(
            {m: {**baseline_report["aggregate"].get(m, {}), "mean": quality[m]["baseline_mean"]}
             for m in quality}
        ),
    }