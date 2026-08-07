"""Benchmark the RAG Curator (contextops.curator) against a real LLM provider.

Measures whether filtering noisy retrieval chunks with `curate()` before
sending them to an LLM actually helps: fewer tokens/cost AND better answer
quality (judge metrics + `context_precision`), compared to raw/uncurated
chunk stuffing (send everything the retriever returned, unfiltered).

This is a DIFFERENT axis from `contextops_bench.runner`, which A/B's SECTION
ORDERING (optimized vs reversed) for identical content to measure cache hit
rate. Here the CONTENT itself differs between arms (curated vs raw chunks)
for the same query — there is no cache angle at all, so this module reuses
only the low-level LLM client plumbing (`client.complete()`), not
`runner.py`'s cache-oriented split-message logic.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from contextops.curator import (
    CurationResult,
    CuratorConfig,
    DocumentChunk,
    context_precision,
    curate,
)
from contextops.judge import JudgeClient, score_many
from contextops.models import Prompt
from contextops.report import a_b_compare
from contextops_bench.stats import bootstrap_ci, effect_size_pct

# A quality-gate run needs at least this many paired (raw, curated) items
# to be considered adequately powered — same threshold as the cache-hit-rate
# gate in contextops_bench/quality.py, for consistency across bench modes.
QUALITY_MIN_N = 20

# (query, expected_answer, relevant_chunk_text) — simple, checkable facts.
FACTS: list[tuple[str, str, str]] = [
    ("What is the capital of France?", "Paris",
     "Paris is the capital and most populous city of France."),
    ("What is the capital of Japan?", "Tokyo",
     "Tokyo is the capital city of Japan."),
    ("What is the boiling point of water at sea level?", "100 degrees Celsius",
     "Water boils at 100 degrees Celsius (212 Fahrenheit) at sea level."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare",
     "Romeo and Juliet is a tragedy written by William Shakespeare."),
    ("What is the largest planet in the solar system?", "Jupiter",
     "Jupiter is the largest planet in the solar system."),
    ("What is the chemical symbol for gold?", "Au",
     "The chemical symbol for gold is Au, from the Latin word aurum."),
    ("How many continents are there on Earth?", "Seven",
     "There are seven continents on Earth."),
    ("What language is spoken in Brazil?", "Portuguese",
     "Portuguese is the official language of Brazil."),
    ("What is the tallest mountain on Earth?", "Mount Everest",
     "Mount Everest is the tallest mountain on Earth above sea level."),
    ("What is the currency of Japan?", "Yen",
     "The yen is the official currency of Japan."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci",
     "The Mona Lisa was painted by the Italian artist Leonardo da Vinci."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean",
     "The Pacific Ocean is the largest and deepest ocean on Earth."),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide",
     "Plants absorb carbon dioxide from the atmosphere during photosynthesis."),
    ("What is the freezing point of water in Celsius?", "0 degrees Celsius",
     "Water freezes at 0 degrees Celsius at standard atmospheric pressure."),
    ("What is the smallest prime number?", "2",
     "The number 2 is the smallest prime number and the only even prime."),
]

# Unrelated filler chunks — simulate retrieval noise (topically distant text
# a similarity search might still surface as a low-ranked, low-relevance hit).
NOISE_CHUNKS: list[str] = [
    "The stock market closed higher today amid renewed investor optimism.",
    "A new species of beetle was discovered in the Amazon rainforest.",
    "The recipe calls for two cups of flour and a teaspoon of salt.",
    "Traffic on the highway was delayed due to ongoing construction work.",
    "The museum's new exhibit features Renaissance-era sculptures.",
    "Local officials announced a new recycling program starting next month.",
    "The football match ended in a scoreless draw after extra time.",
    "Researchers published a study on deep-sea coral reef ecosystems.",
    "The outdoor concert was rescheduled due to inclement weather.",
    "A vintage car show will be held downtown this coming weekend.",
    "The company reported a modest increase in quarterly revenue.",
    "Volunteers cleaned up litter along the riverbank over the weekend.",
    "The airline announced new direct routes to several European cities.",
    "A community garden project was launched in the east neighborhood.",
    "The novel's long-awaited sequel is expected to release next spring.",
]


@dataclass
class CurationBenchItem:
    """One synthetic RAG query plus a mixed set of relevant/noise/duplicate chunks."""

    query: str
    expected: str
    chunks: list[DocumentChunk]


def generate_curation_dataset(
    n: int,
    *,
    noise_ratio: float = 0.7,
    chunks_per_item: int = 6,
    dup_rate: float = 0.15,
    seed: int = 42,
) -> list[CurationBenchItem]:
    """Deterministic synthetic dataset for benching the curator.

    Each item mixes a relevant chunk (high `similarity`, drawn from `FACTS`)
    with noise chunks (low `similarity`, drawn from `NOISE_CHUNKS`) at
    `noise_ratio`, and occasionally injects a near-duplicate of the relevant
    chunk (at `dup_rate`) to exercise `curate()`'s dedup path. Fully
    reproducible given the same `seed`.
    """
    rng = random.Random(seed)
    items: list[CurationBenchItem] = []
    for _ in range(n):
        query, expected, relevant_text = rng.choice(FACTS)
        n_noise = max(0, round(chunks_per_item * noise_ratio))
        n_relevant = max(1, chunks_per_item - n_noise)

        chunks: list[DocumentChunk] = []
        for _ in range(n_relevant):
            chunks.append(DocumentChunk(text=relevant_text, similarity=rng.uniform(0.80, 0.98)))
        for _ in range(n_noise):
            noise_text = rng.choice(NOISE_CHUNKS)
            chunks.append(DocumentChunk(text=noise_text, similarity=rng.uniform(0.10, 0.50)))

        if rng.random() < dup_rate:
            # Near-duplicate of the relevant chunk — exercises curate()'s
            # Jaccard-overlap dedup, not just the similarity threshold.
            chunks.append(DocumentChunk(text=relevant_text, similarity=rng.uniform(0.75, 0.95)))

        rng.shuffle(chunks)
        items.append(CurationBenchItem(query=query, expected=expected, chunks=chunks))
    return items


def _render(prompt: Prompt) -> str:
    """Flatten a Prompt's sections (documents + query, in canonical order) into one string."""
    return "\n\n".join(content for _, content in prompt.sections())


def _complete_text(client: Any, *, model: str, prompt_str: str, max_tokens: int = 64) -> tuple[str, dict]:
    """Call `client.complete()` with a single user message. Returns (text, stats)."""
    resp = client.complete(
        model=model,
        messages=[{"role": "user", "content": prompt_str or "(empty)"}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = resp.raw if isinstance(resp.raw, dict) else {}
    return resp.text, {
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "cost_usd": resp.cost_usd,
        "latency_ms": raw.get("_latency_ms", 0.0),
    }


class _ClientAsJudge:
    """Adapts any bench client (`BaseHTTPClient`/`EchoClient`/anything with a
    `.complete()` returning `CompletionResponse`) to the `JudgeClient`
    protocol (`complete(*, model, messages, temperature=0.0) -> str`).

    Lets the curator bench self-judge using the SAME real API key/client
    already configured for the LLM-under-test — no separate judge
    credentials required. Known limitation: self-judging bias (a model
    grading its own or a sibling model's answers) — see the `[JUDGE]` line
    `render_curator_summary()` prints when this adapter is used.
    """

    def __init__(self, client: Any, *, max_tokens: int = 150):
        self._client = client
        self._max_tokens = max_tokens

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str:
        # `contextops.judge._build_messages()` always emits a leading
        # {"role": "system", ...} message. Anthropic-native clients (Zen,
        # direct_anthropic, direct_google — anything with
        # `supports_split_messages=True`, mirroring `runner.py`'s handling)
        # reject a "system" role inside `messages` with HTTP 400; they
        # require system content via a separate `system=` kwarg instead.
        kwargs: dict = {}
        if getattr(self._client, "supports_split_messages", False) and messages and messages[0]["role"] == "system":
            kwargs["system"] = messages[0]["content"]
            messages = messages[1:]
        resp = self._client.complete(
            model=model, messages=messages, temperature=temperature, max_tokens=self._max_tokens, **kwargs,
        )
        return resp.text


def evaluate_curator_quality_gate(
    raw_scores: list[dict],
    curated_scores: list[dict],
    *,
    min_n: int = QUALITY_MIN_N,
) -> dict:
    """Statistical significance gate on the curator bench's judge-metric
    deltas — mirrors `contextops_bench.quality.evaluate_quality_gate`'s
    bootstrap-CI approach, applied per metric.

    `raw_scores`/`curated_scores` are `score_many()` outputs: each entry has
    `"metric"`, `"score"`, and `"index"` (the dataset item index). Entries
    are paired by `(metric, index)` to build per-item diff arrays, since
    `score_many()` always returns results ordered by
    `(response_index, metric_index)` regardless of `max_workers`.

    Returns `{"n", "min_n", "low_n", "verified", "significant_metrics",
    "reasons", "per_metric": {metric: {"ci_low", "ci_high", "significant",
    "effect_size_pct"}}}`.
    """
    by_metric_raw: dict[str, dict[int, float]] = {}
    by_metric_curated: dict[str, dict[int, float]] = {}
    for s in raw_scores:
        by_metric_raw.setdefault(s["metric"], {})[s["index"]] = s["score"]
    for s in curated_scores:
        by_metric_curated.setdefault(s["metric"], {})[s["index"]] = s["score"]

    metrics = sorted(set(by_metric_raw) | set(by_metric_curated))
    per_metric: dict[str, dict] = {}
    significant_metrics: list[str] = []
    n = 0

    for metric in metrics:
        raw_by_idx = by_metric_raw.get(metric, {})
        curated_by_idx = by_metric_curated.get(metric, {})
        shared_idx = sorted(set(raw_by_idx) & set(curated_by_idx))
        raw_list = [raw_by_idx[i] for i in shared_idx]
        curated_list = [curated_by_idx[i] for i in shared_idx]
        n = max(n, len(shared_idx))

        if not shared_idx:
            per_metric[metric] = {
                "ci_low": None, "ci_high": None, "significant": False, "effect_size_pct": 0.0,
            }
            continue

        diffs = [c - r for r, c in zip(raw_list, curated_list)]
        ci_low, ci_high = bootstrap_ci(diffs, seed=0)
        significant = not (ci_low <= 0 <= ci_high)
        if significant:
            significant_metrics.append(metric)
        per_metric[metric] = {
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": significant,
            "effect_size_pct": effect_size_pct(curated_list, raw_list),
        }

    low_n = n < min_n
    reasons: list[str] = []
    if low_n:
        reasons.append(f"n={n} < min_n={min_n}")
    if n and not significant_metrics:
        reasons.append("no metric's 95% CI excludes zero — no statistically significant quality delta")

    return {
        "n": n,
        "min_n": min_n,
        "low_n": low_n,
        "verified": (not low_n) and bool(significant_metrics),
        "significant_metrics": significant_metrics,
        "reasons": reasons,
        "per_metric": per_metric,
    }


def _stats(records: list[dict], precisions: list[float]) -> dict:
    """Aggregate one arm's per-call records + context_precision scores."""
    if not records:
        return {}
    prompt_tokens = [r["prompt_tokens"] for r in records]
    completion_tokens = [r["completion_tokens"] for r in records]
    costs = [r["cost_usd"] for r in records]
    latencies = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
    return {
        "n": len(records),
        "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 1),
        "completion_tokens_mean": round(statistics.mean(completion_tokens), 1),
        "cost_usd_per_call": round(statistics.mean(costs), 6),
        "cost_usd_total": round(sum(costs), 6),
        "latency_ms_p50": round(statistics.median(latencies), 1) if latencies else 0.0,
        "context_precision_mean": round(statistics.mean(precisions), 3) if precisions else 0.0,
    }


def run_curator_bench(
    items: list[CurationBenchItem],
    *,
    client: Any,
    model: str,
    judge: JudgeClient,
    metrics: list[str],
    config: CuratorConfig | None = None,
    judge_model: str = "gpt-4o-mini",
    max_workers: int = 1,
    max_tokens: int = 200,
    judge_label: str = "echo",
) -> dict:
    """Run every item through both arms (raw vs curated chunks), score, aggregate.

    "raw" = every retrieved chunk joined into `documents`, unfiltered.
    "curated" = only `curate(item.chunks, config).kept` joined into `documents`.
    Both arms share the same `query`/`model` and are sent to the same `client`.

    `max_tokens` controls the LLM-under-test's answer length (default 200 —
    short answers starve both the judge and `context_precision`'s n-gram
    heuristic of signal). `judge_label` is a free-form string describing the
    judge (`"echo"`, `"self"`, `"litellm:<model>"`, ...) stored in the summary
    for transparency — see `render_curator_summary`'s `[JUDGE]` line.

    Returns a summary dict: `{"provider", "model", "n", "raw", "curated",
    "quality", "quality_gate", "curation", "delta", "judge"}` — see
    `render_curator_summary` for the human-readable rendering and
    `scripts/generate_dashboard.py` for how this shape feeds the public
    dashboard.
    """
    config = config or CuratorConfig()

    raw_records: list[dict] = []
    curated_records: list[dict] = []
    raw_responses: list[str] = []
    curated_responses: list[str] = []
    raw_precisions: list[float] = []
    curated_precisions: list[float] = []
    drop_rates: list[float] = []
    dedup_drop_count = 0

    for item in items:
        curation_result: CurationResult = curate(item.chunks, config)

        raw_prompt = Prompt(
            documents="\n\n".join(c.text for c in item.chunks),
            query=item.query,
            model=model,
        )
        curated_prompt = Prompt(
            documents="\n\n".join(c.text for c in curation_result.kept),
            query=item.query,
            model=model,
        )

        raw_text, raw_stats = _complete_text(
            client, model=model, prompt_str=_render(raw_prompt), max_tokens=max_tokens
        )
        curated_text, curated_stats = _complete_text(
            client, model=model, prompt_str=_render(curated_prompt), max_tokens=max_tokens
        )

        raw_records.append(raw_stats)
        curated_records.append(curated_stats)
        raw_responses.append(raw_text)
        curated_responses.append(curated_text)

        # Raw arm: nothing was filtered, so "kept" = every chunk sent.
        raw_precisions.append(context_precision(item.chunks, raw_text)["precision"])
        curated_precisions.append(context_precision(curation_result.kept, curated_text)["precision"])

        n_chunks = len(item.chunks)
        drop_rates.append(len(curation_result.dropped) / n_chunks if n_chunks else 0.0)
        dedup_drop_count += sum(1 for _, reason in curation_result.dropped if "duplicate" in reason)

    queries = [it.query for it in items]
    expecteds = [it.expected for it in items]
    raw_scores = score_many(
        raw_responses, metrics=metrics, judge=judge, model=judge_model,
        queries=queries, expecteds=expecteds, max_workers=max_workers,
    )
    curated_scores = score_many(
        curated_responses, metrics=metrics, judge=judge, model=judge_model,
        queries=queries, expecteds=expecteds, max_workers=max_workers,
    )
    # a_b_compare(baseline, optimized) — "baseline" = raw, "optimized" = curated,
    # so delta = curated - raw (positive delta = curation helped quality).
    quality_deltas = a_b_compare(raw_scores, curated_scores)
    quality_gate = evaluate_curator_quality_gate(raw_scores, curated_scores)
    # Enrich each metric's a_b_compare entry with the gate's CI/significance/
    # effect-size fields — additive, doesn't change the existing shape.
    for metric, gate_info in quality_gate["per_metric"].items():
        if metric in quality_deltas:
            quality_deltas[metric].update(gate_info)

    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": getattr(client, "PROVIDER", "unknown"),
        "model": model,
        "judge": judge_label,
        "n": len(items),
        "raw": _stats(raw_records, raw_precisions),
        "curated": _stats(curated_records, curated_precisions),
        "quality": quality_deltas,
        "quality_gate": quality_gate,
        "curation": {
            "mean_drop_rate": round(statistics.mean(drop_rates), 3) if drop_rates else 0.0,
            "total_dedup_drops": dedup_drop_count,
        },
    }
    if summary["raw"] and summary["curated"]:
        summary["delta"] = {
            "prompt_tokens_delta_mean": round(
                summary["curated"]["prompt_tokens_mean"] - summary["raw"]["prompt_tokens_mean"], 1
            ),
            "cost_delta_usd_per_call": round(
                summary["curated"]["cost_usd_per_call"] - summary["raw"]["cost_usd_per_call"], 6
            ),
            "context_precision_delta": round(
                summary["curated"]["context_precision_mean"] - summary["raw"]["context_precision_mean"], 3
            ),
        }
    return summary


def render_curator_summary(summary: dict, label: str = "curator_bench") -> str:
    """Fixed-width text summary, same style as `contextops_bench.runner.render_summary`."""
    lines = [f"=== {label} ==="]
    if summary.get("generated_at"):
        lines.append(f"generated_at: {summary['generated_at']}")
    lines.append(
        f"provider={summary.get('provider', '?')}  model={summary.get('model', '?')}  "
        f"n={summary.get('n', 0)}"
    )
    judge_label = summary.get("judge", "")
    if judge_label:
        lines.append(f"judge={judge_label}")
        if judge_label == "self" or judge_label.startswith("self:"):
            lines.append(
                "  [!] self-judged — the same model (or a sibling model on the same "
                "provider) graded its own answers; results may be biased toward this "
                "model's own answer style."
            )

    for side in ("raw", "curated"):
        s = summary.get(side) or {}
        if not s:
            continue
        lines.append(f"\n[{side.upper()}] (n={s.get('n', 0)})")
        lines.append(f"  prompt tokens:      mean={s['prompt_tokens_mean']:>8}")
        lines.append(f"  completion tokens:  mean={s['completion_tokens_mean']:>8}")
        lines.append(f"  cost / call:        ${s['cost_usd_per_call']:.6f}   total=${s['cost_usd_total']:.4f}")
        lines.append(f"  latency p50:        {s['latency_ms_p50']:.0f}ms")
        lines.append(f"  context precision:  {s['context_precision_mean']:.1%}")

    curation = summary.get("curation") or {}
    if curation:
        lines.append(
            f"\n[CURATION] mean drop rate={curation.get('mean_drop_rate', 0.0):.1%}  "
            f"total dedup drops={curation.get('total_dedup_drops', 0)}"
        )

    d = summary.get("delta") or {}
    if d:
        lines.append("\n[DELTA] (curated - raw)")
        lines.append(f"  prompt tokens:      {d['prompt_tokens_delta_mean']:+.1f}")
        lines.append(f"  cost / call:        ${d['cost_delta_usd_per_call']:+.6f}")
        lines.append(f"  context precision:  {d['context_precision_delta']:+.1%}")

    q = summary.get("quality") or {}
    if q:
        lines.append("\n[QUALITY] (judge deltas, curated vs raw)")
        for metric, delta in q.items():
            ci_str = ""
            if delta.get("ci_low") is not None:
                sig = "significant" if delta.get("significant") else "not significant"
                ci_str = (
                    f"  95% CI=[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}] ({sig}, "
                    f"effect={delta.get('effect_size_pct', 0.0):+.1f}%)"
                )
            lines.append(
                f"  {metric:<16s} raw={delta['baseline_mean']:.3f}  "
                f"curated={delta['optimized_mean']:.3f}  delta={delta['delta']:+.3f}{ci_str}"
            )

    qg = summary.get("quality_gate") or {}
    if qg:
        lines.append("\n[QUALITY GATE]")
        status = "PASS (verified)" if qg.get("verified") else "FAIL (unverified)"
        lines.append(f"  status:              {status}")
        lines.append(f"  n={qg.get('n', 0)}  min_n={qg.get('min_n', 0)}")
        sig_metrics = qg.get("significant_metrics") or []
        lines.append(f"  significant metrics: {', '.join(sig_metrics) or 'none'}")
        for reason in qg.get("reasons", []):
            lines.append(f"  - {reason}")

    return "\n".join(lines)
