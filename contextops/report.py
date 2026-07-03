"""Report generation for v0.2 — turns raw scores into human-readable summaries."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable


def aggregate(scores: Iterable[dict]) -> dict[str, dict]:
    """Aggregate per-metric scores: mean, median, stdev, min, max, count, pass_rate.

    `pass_rate` = fraction of scores >= 0.7 (configurable later).
    """
    by_metric: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        by_metric[s["metric"]].append(float(s["score"]))

    summary: dict[str, dict] = {}
    for metric, vals in by_metric.items():
        if not vals:
            continue
        summary[metric] = {
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "count": len(vals),
            "pass_rate": round(sum(1 for v in vals if v >= 0.7) / len(vals), 3),
        }
    return summary


def render_table(summary: dict[str, dict]) -> str:
    """Render a fixed-width table for CLI output."""
    headers = ["Metric", "Mean", "Median", "Stdev", "Min", "Max", "Pass@0.7", "N"]
    rows = [headers]
    for metric, s in summary.items():
        rows.append(
            [
                metric,
                f"{s['mean']:.3f}",
                f"{s['median']:.3f}",
                f"{s['stdev']:.2f}",
                f"{s['min']:.3f}",
                f"{s['max']:.3f}",
                f"{s['pass_rate']:.1%}",
                str(s["count"]),
            ]
        )

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    out = []
    for i, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        out.append(line)
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def a_b_compare(
    baseline_scores: list[dict],
    optimized_scores: list[dict],
) -> dict:
    """Compare two score lists side-by-side, return deltas."""
    base = aggregate(baseline_scores)
    opt = aggregate(optimized_scores)
    deltas: dict[str, dict] = {}
    for metric in set(base) | set(opt):
        b = base.get(metric, {})
        o = opt.get(metric, {})
        deltas[metric] = {
            "baseline_mean": b.get("mean"),
            "optimized_mean": o.get("mean"),
            "delta": round((o.get("mean", 0) - b.get("mean", 0)), 3),
            "n": b.get("count", 0),
        }
    return deltas