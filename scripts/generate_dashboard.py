"""Generate a static HTML dashboard from bench result summaries.

Reads `bench/results/*.summary.json` and emits `docs/dashboard/index.html`.
The output is self-contained (CDN-hosted Tailwind + Chart.js) so it can be
served directly by GitHub Pages with no build step.

Usage:
    python scripts/generate_dashboard.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from contextops_bench.quality import evaluate_quality_gate

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "bench" / "results"
OUTPUT_DIR = ROOT / "docs" / "dashboard"


def _parse_filename(path: Path) -> tuple[str, str]:
    """Infer provider/model from a summary filename.

    Examples:
        cloud_openai_gpt-4o-mini.summary.json -> (openai, gpt-4o-mini)
        mac_zen_n5.summary.json -> (zen, n5)
        local_ollama.summary.json -> (ollama, local)
    """
    name = path.stem.replace(".summary", "")
    # Strip environment prefixes and summary suffixes.
    name = re.sub(r"^(cloud|local|mac)_", "", name)
    name = re.sub(r"_summary$", "", name)
    name = re.sub(r"\.summary$", "", name)

    # Try to split on the first underscore after a provider-like token.
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[0], "_".join(parts[1:])
    if name:
        return name, name
    return "unknown", "unknown"


# Filename-provider tokens that are actually model VENDOR names, not real
# API providers — these summaries were run via OpenRouter (the only client
# that dispatches "vendor/model"-style names) but the filename convention
# splits the vendor into the "provider" slot. Used only as a fallback for
# legacy summaries that predate the ground-truth `provider`/`model` fields
# runner.py now writes into every summary.json.
_VENDOR_LABELS_MEANING_OPENROUTER = {"anthropic", "openai", "google"}


def _runtime_provider_model(display_provider: str, display_model: str, data: dict) -> tuple[str, str]:
    """Resolve the (provider, model) pair to feed the quality gate.

    Prefers the ground-truth fields runner.py stores directly in the summary
    (`data["provider"]`, `data["model"]`) — these reflect exactly what the
    bench actually called at runtime. Falls back to reconstructing the
    likely runtime values from the filename-derived display labels for
    older summaries that predate those fields.
    """
    if data.get("provider") and data.get("model"):
        return str(data["provider"]), str(data["model"])
    if display_provider in _VENDOR_LABELS_MEANING_OPENROUTER:
        return "openrouter", f"{display_provider}/{display_model}"
    return display_provider, display_model


def _load_runs() -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.summary.json")):
        provider, model = _parse_filename(path)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        # Always (re-)compute the quality gate deterministically from the
        # summary's own stats, rather than trusting a possibly-stale stored
        # `quality` key — this keeps legacy summaries (generated before the
        # gate existed) and fresh ones held to the same standard. Use the
        # resolved runtime provider/model (not the display labels) so gates
        # like `cache_marker_dropped` match reality regardless of filename
        # convention quirks.
        runtime_provider, runtime_model = _runtime_provider_model(provider, model, data)
        quality = evaluate_quality_gate(data, provider=runtime_provider, model=runtime_model)
        runs.append(
            {
                "path": str(path.relative_to(ROOT)),
                "provider": provider,
                "model": model,
                "data": data,
                "quality": quality,
            }
        )
    return runs


def _safe(d: dict, *keys: str, default=0.0) -> float:
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    try:
        return float(d)
    except (TypeError, ValueError):
        return default


def _build_datasets(runs: list[dict]) -> dict:
    labels = [f"{r['provider']} · {r['model']}" for r in runs]
    optimized_hits = [
        _safe(r["data"], "optimized", "cache_hit_rate_mean") for r in runs
    ]
    baseline_hits = [
        _safe(r["data"], "baseline", "cache_hit_rate_mean") for r in runs
    ]
    optimized_costs = [
        _safe(r["data"], "optimized", "cost_usd_per_call") for r in runs
    ]
    baseline_costs = [
        _safe(r["data"], "baseline", "cost_usd_per_call") for r in runs
    ]
    deltas = [_safe(r["data"], "delta", "cache_hit_rate_delta") for r in runs]
    return {
        "labels": labels,
        "optimized_hits": optimized_hits,
        "baseline_hits": baseline_hits,
        "optimized_costs": optimized_costs,
        "baseline_costs": baseline_costs,
        "deltas": deltas,
    }


def _summary_stats(runs: list[dict]) -> dict:
    if not runs:
        return {
            "runs": 0,
            "best_hit": 0.0,
            "best_cost_delta": 0.0,
            "avg_delta": 0.0,
            "wins": 0,
            "verified": 0,
        }
    deltas = [_safe(r["data"], "delta", "cache_hit_rate_delta") for r in runs]
    cost_deltas = [
        -_safe(r["data"], "delta", "cost_per_call_delta_usd") for r in runs
    ]
    verified_flags = [r["quality"]["verified"] for r in runs]
    # A "win" requires BOTH a statistically significant cost delta (the
    # quality gate's `verified` flag) AND that the delta actually reduces
    # cost — a verified, significant *increase* is real but not a win.
    wins = sum(
        1
        for verified, c in zip(verified_flags, cost_deltas)
        if verified and c > 0
    )
    return {
        "runs": len(runs),
        "best_hit": max(
            _safe(r["data"], "optimized", "cache_hit_rate_mean") for r in runs
        ),
        "best_cost_delta": max(cost_deltas) if cost_deltas else 0.0,
        "avg_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "wins": wins,
        "verified": sum(verified_flags),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ContextOps Benchmark Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    .chart-container { position: relative; height: 320px; width: 100%; }
  </style>
</head>
<body class="bg-slate-50 text-slate-900">
  <div class="max-w-7xl mx-auto px-6 py-10">
    <header class="mb-10">
      <h1 class="text-4xl font-bold tracking-tight mb-2">ContextOps Benchmark Dashboard</h1>
      <p class="text-slate-600">
        Live-ish results from real provider runs. Optimized prompts reorder stable
        sections first; baseline uses a worst-case ordering.
      </p>
      <p class="text-xs text-slate-400 mt-2">
        Generated from bench/results/*.summary.json. Every run is scored by a
        deterministic quality gate: n &ge; 20 paired samples, error rate
        &le; 15%, and a cost-delta 95% CI that excludes zero. "Verified"
        means the gate passed; "Win" additionally requires the cost delta to
        be a reduction. Unverified runs are shown but flagged with why they
        didn't pass (see the Verified column).
      </p>
    </header>

    <section class="grid grid-cols-1 md:grid-cols-6 gap-4 mb-10">
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <p class="text-sm font-medium text-slate-500">Runs</p>
        <p class="text-3xl font-bold">__RUNS__</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <p class="text-sm font-medium text-slate-500">Verified</p>
        <p class="text-3xl font-bold text-blue-600">__VERIFIED__</p>
        <p class="text-xs text-slate-400">n&ge;20, err&le;15%, CI excludes 0</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <p class="text-sm font-medium text-slate-500">Verified wins</p>
        <p class="text-3xl font-bold text-emerald-600">__WINS__</p>
        <p class="text-xs text-slate-400">verified + cost &Delta; &lt; $0</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <p class="text-sm font-medium text-slate-500">Best cache hit rate</p>
        <p class="text-3xl font-bold text-emerald-600">__BEST_HIT__</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <p class="text-sm font-medium text-slate-500">Best cost reduction</p>
        <p class="text-3xl font-bold text-emerald-600">__BEST_COST_DELTA__</p>
        <p class="text-xs text-slate-400">per call</p>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <p class="text-sm font-medium text-slate-500">Avg cache-hit &Delta;</p>
        <p class="text-3xl font-bold text-blue-600">__AVG_DELTA__</p>
      </div>
    </section>

    <section class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <h2 class="text-lg font-semibold mb-4">Cache hit rate</h2>
        <div class="chart-container"><canvas id="hitChart"></canvas></div>
      </div>
      <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200">
        <h2 class="text-lg font-semibold mb-4">Cost per call</h2>
        <div class="chart-container"><canvas id="costChart"></canvas></div>
      </div>
    </section>

    <section class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <h2 class="text-lg font-semibold p-6 border-b border-slate-200">All runs</h2>
      <div class="overflow-x-auto">
        <table class="min-w-full text-sm">
          <thead class="bg-slate-100 text-slate-600">
            <tr>
              <th class="text-left px-6 py-3 font-medium">Provider</th>
              <th class="text-left px-6 py-3 font-medium">Model</th>
              <th class="text-right px-6 py-3 font-medium">n (opt/base)</th>
              <th class="text-right px-6 py-3 font-medium">Hit rate &Delta;</th>
              <th class="text-right px-6 py-3 font-medium">Cost &Delta;/call (mean)</th>
              <th class="text-right px-6 py-3 font-medium">Effect size (median)</th>
              <th class="text-right px-6 py-3 font-medium">Savings / 1k calls</th>
              <th class="text-right px-6 py-3 font-medium">Latency opt/base</th>
              <th class="text-left px-6 py-3 font-medium">Verified</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            __TABLE_ROWS__
          </tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    const labels = __LABELS_JSON__;
    const optimizedHits = __OPTIMIZED_HITS_JSON__;
    const baselineHits = __BASELINE_HITS_JSON__;
    const optimizedCosts = __OPTIMIZED_COSTS_JSON__;
    const baselineCosts = __BASELINE_COSTS_JSON__;
    const deltas = __DELTAS_JSON__;

    const commonOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        x: { ticks: { callback: (value, index) => {
              const label = labels[index] || '';
              return label.length > 20 ? label.slice(0, 20) + '...' : label;
            }, maxRotation: 45, minRotation: 30 } },
        y: { beginAtZero: true },
      },
    };

    new Chart(document.getElementById('hitChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Optimized', data: optimizedHits, backgroundColor: '#10b981' },
          { label: 'Baseline', data: baselineHits, backgroundColor: '#64748b' },
        ],
      },
      options: commonOptions,
    });

    new Chart(document.getElementById('costChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Optimized', data: optimizedCosts, backgroundColor: '#10b981' },
          { label: 'Baseline', data: baselineCosts, backgroundColor: '#64748b' },
        ],
      },
      options: commonOptions,
    });
  </script>
</body>
</html>
"""


def _render_table(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        opt = r["data"].get("optimized", {})
        base = r["data"].get("baseline", {})
        delta = r["data"].get("delta", {})
        quality = r["quality"]
        n_opt = opt.get("n", 0)
        n_base = base.get("n", 0)
        hit_delta = _safe(delta, "cache_hit_rate_delta")
        cost_delta = _safe(delta, "cost_per_call_delta_usd")
        effect_size = delta.get("effect_size_pct")
        lat_opt = _safe(opt, "latency_ms_p50")
        lat_base = _safe(base, "latency_ms_p50")
        hit_class = "text-emerald-600" if hit_delta > 0 else "text-slate-600"
        cost_class = "text-emerald-600" if cost_delta < 0 else "text-slate-600"
        savings_1k = -cost_delta * 1000
        effect_size_str = f"{effect_size:+.1f}%" if effect_size is not None else "n/a"

        if quality["verified"]:
            verified_badge = (
                '<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full '
                'text-xs font-medium bg-emerald-100 text-emerald-700">verified</span>'
            )
        else:
            reasons = "; ".join(quality["reasons"]) or "did not pass quality gate"
            verified_badge = (
                f'<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full '
                f'text-xs font-medium bg-amber-100 text-amber-700" title="{reasons}">'
                f'unverified</span>'
            )

        rows.append(
            f"""<tr class="hover:bg-slate-50">
  <td class="px-6 py-3 font-medium">{r['provider']}</td>
  <td class="px-6 py-3 text-slate-600">{r['model']}</td>
  <td class="px-6 py-3 text-right">{n_opt}/{n_base}</td>
  <td class="px-6 py-3 text-right {hit_class}">{hit_delta:+.1%}</td>
  <td class="px-6 py-3 text-right {cost_class}">${cost_delta:+.6f}</td>
  <td class="px-6 py-3 text-right {cost_class}">{effect_size_str}</td>
  <td class="px-6 py-3 text-right {cost_class}">${savings_1k:+.2f}</td>
  <td class="px-6 py-3 text-right">{lat_opt:.0f}ms / {lat_base:.0f}ms</td>
  <td class="px-6 py-3">{verified_badge}</td>
</tr>"""
        )
    return "\n".join(rows)


def main() -> None:
    runs = _load_runs()
    if not runs:
        print(f"No summary JSON files found in {RESULTS_DIR}")
        return

    datasets = _build_datasets(runs)
    stats = _summary_stats(runs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "index.html"

    html = (
        HTML_TEMPLATE
        .replace("__RUNS__", str(stats["runs"]))
        .replace("__VERIFIED__", str(stats["verified"]))
        .replace("__WINS__", str(stats["wins"]))
        .replace("__BEST_HIT__", f"{stats['best_hit']:.1%}")
        .replace("__BEST_COST_DELTA__", f"${stats['best_cost_delta']:.6f}")
        .replace("__AVG_DELTA__", f"{stats['avg_delta']:+.1%}")
        .replace("__LABELS_JSON__", json.dumps(datasets["labels"]))
        .replace("__OPTIMIZED_HITS_JSON__", json.dumps(datasets["optimized_hits"]))
        .replace("__BASELINE_HITS_JSON__", json.dumps(datasets["baseline_hits"]))
        .replace("__OPTIMIZED_COSTS_JSON__", json.dumps(datasets["optimized_costs"]))
        .replace("__BASELINE_COSTS_JSON__", json.dumps(datasets["baseline_costs"]))
        .replace("__DELTAS_JSON__", json.dumps(datasets["deltas"]))
        .replace("__TABLE_ROWS__", _render_table(runs))
    )

    output.write_text(html)
    print(f"Wrote dashboard to {output} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
