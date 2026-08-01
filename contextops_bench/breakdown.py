"""Per-prompt breakdown of A/B bench results.

For each prompt the bench ran twice (optimized + baseline) this module
extracts a single paired row with cost/cache deltas on both sides. Sorting
by ``|Δ cost|`` descending surfaces the prompts where the reorder actually
helped vs hurt — the diagnostic signal headline summary stats can't show.

Top-N rows land in the rendered summary AND in ``<label>.breakdown.csv``.
The full per-row data already lives in ``<label>.csv`` from ``runner.save_csv``,
so the breakdown CSV is a derived diagnostic view, not a re-export.
"""

from __future__ import annotations

import csv
from pathlib import Path

from contextops_bench.clients import BenchResult


# Stable column order. Any downstream tooling that consumes the breakdown CSV
# should rely on this constant — do NOT reorder without checking consumers.
COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "model",
    "prompt_tokens",
    "baseline_cost",
    "optimized_cost",
    "delta_cost",
    "delta_pct",
    "baseline_cache_hit",
    "optimized_cache_hit",
)


def _cache_hit(r: BenchResult) -> float:
    """``cached / (cached + prompt)``. Returns 0.0 when the denominator is 0."""
    denom = r.cached_tokens + r.prompt_tokens
    if denom <= 0:
        return 0.0
    return r.cached_tokens / denom


def per_prompt_breakdown(
    optimized: list[BenchResult],
    baseline: list[BenchResult],
    *,
    top_n: int = 10,
) -> list[dict]:
    """Build per-prompt A/B rows, sorted by ``|Δ cost|`` descending, truncated.

    Pairing rule: rows are paired by ``prompt_id``. Prompt IDs that appear in
    only one arm are silently dropped (defensive — shouldn't happen given
    ``run_batch`` runs each prompt twice, but if it does, surface it via
    unequal ``len(optimized) != len(baseline)`` upstream).

    Error filter: rows where either arm has a non-empty ``error`` field are
    dropped (their cost/cache data is not meaningful).

    Edge cases:
      - Empty inputs → ``[]``
      - ``baseline_cost == 0`` → ``delta_pct = 0.0`` (% comparison undefined)
      - All paired deltas are 0 → rows still emit, sort order is stable on
        ``prompt_id`` ascending (deterministic, easy to diff across runs)

    Args:
      optimized: optimized-arm results
      baseline: baseline-arm results
      top_n: keep at most this many rows after sorting
    """
    opt_by_id = {r.prompt_id: r for r in optimized if not r.error}
    base_by_id = {r.prompt_id: r for r in baseline if not r.error}
    common_ids = sorted(set(opt_by_id) & set(base_by_id))

    rows: list[dict] = []
    for pid in common_ids:
        opt = opt_by_id[pid]
        base = base_by_id[pid]
        delta_cost = opt.cost_usd - base.cost_usd
        if base.cost_usd != 0.0:
            delta_pct = round((delta_cost / base.cost_usd) * 100.0, 2)
        else:
            delta_pct = 0.0
        rows.append({
            "prompt_id": pid,
            "model": opt.model,
            "prompt_tokens": opt.prompt_tokens,
            "baseline_cost": round(base.cost_usd, 6),
            "optimized_cost": round(opt.cost_usd, 6),
            "delta_cost": round(delta_cost, 6),
            "delta_pct": delta_pct,
            "baseline_cache_hit": round(_cache_hit(base), 3),
            "optimized_cache_hit": round(_cache_hit(opt), 3),
        })

    # Sort by |delta_cost| descending; secondary key prompt_id ascending keeps
    # the sort stable so equal-cost rows don't shuffle between runs.
    rows.sort(key=lambda r: (-abs(r["delta_cost"]), r["prompt_id"]))
    return rows[:top_n]


def save_breakdown_csv(rows: list[dict], path: str | Path) -> None:
    """Persist breakdown rows to CSV. Creates parent dirs. Always writes the
    header row even when ``rows`` is empty so downstream tooling doesn't
    choke on an empty file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        # Project only known columns in stable order; missing keys → empty cell.
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in COLUMNS})


def render_breakdown_table(rows: list[dict], *, title: str | None = None) -> str:
    """Render breakdown rows as a fixed-width text table for the terminal.

    Returns an empty string when there are no rows. Designed to slot in
    directly as additional lines in the existing ``render_summary`` output.
    """
    if not rows:
        return ""
    out: list[str] = []
    out.append("")
    out.append(f"[BREAKDOWN] {title or f'Top {len(rows)} prompts by |delta cost|'}")
    out.append(
        f"  {'pid':>5}  {'model':<24}  {'delta cost':>12}  "
        f"{'delta %':>7}  {'opt hit':>7}  {'base hit':>7}"
    )
    for r in rows:
        model = (r["model"] or "")[:24]
        out.append(
            f"  {r['prompt_id']:>5}  {model:<24}  "
            f"{r['delta_cost']:>+12.6f}  "
            f"{r['delta_pct']:>+6.1f}% "
            f"{r['optimized_cache_hit']:>6.0%}  "
            f"{r['baseline_cache_hit']:>6.0%}"
        )
    return "\n".join(out)
