"""Rich-based presentation helpers for the CLI.

Separated from `cli.py` so the command module is just wiring (Click decorators
+ argument parsing) and `report.py` stays pure-logic (no Rich dependency).
These functions own the visual layer: tables, colors, notes formatting.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from contextops.models import OptimizationResult

console = Console()


def render_optimization(result: OptimizationResult) -> None:
    """Render an OptimizationResult as a side-by-side Original vs Optimized table."""
    table = Table(title="ContextOps Optimization", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Original", justify="right")
    table.add_column("Optimized", justify="right")

    original_order = [s[0] for s in result.original_sections]
    optimized_order = [s[0] for s in result.optimized_sections]

    table.add_row(
        "Section order",
        " → ".join(original_order) or "(empty)",
        " → ".join(optimized_order) or "(empty)",
    )
    table.add_row("Tokens", str(result.original_tokens), str(result.optimized_tokens))
    table.add_row(
        "Cache hit rate (est)",
        f"{result.original_cache_hit_rate:.1%}",
        f"{result.estimated_cache_hit_rate:.1%}",
    )
    table.add_row(
        "Cost savings / 1k calls",
        "$0.00",
        f"${result.estimated_cost_savings_usd:.4f}",
    )

    console.print(table)
    if result.notes:
        console.print("\n[bold]Notes:[/bold]")
        for note in result.notes:
            console.print(f"  • {note}")


def render_eval_report(report: dict) -> None:
    """Render an A/B eval report (structural + quality deltas) as Rich tables."""
    console.rule("[bold]A/B Eval Report")

    # Structural deltas
    s = report["structural"]
    stable = Table(title="Structural deltas", show_lines=False)
    stable.add_column("Metric", style="bold")
    stable.add_column("Value", justify="right")
    stable.add_row("Tokens (Δ)", str(s["tokens"]))
    stable.add_row("Cache hit rate (Δ)", f"{s['cache_hit_rate']:+.1%}")
    stable.add_row("Cost savings / 1k calls (Δ)", f"${s['cost_savings_per_1k_usd']:+.4f}")
    console.print(stable)

    # Quality deltas
    quality_table = Table(title="Quality deltas (judge scores)", show_lines=False)
    quality_table.add_column("Metric", style="bold")
    quality_table.add_column("Baseline", justify="right")
    quality_table.add_column("Optimized", justify="right")
    quality_table.add_column("Δ", justify="right")
    quality_table.add_column("N", justify="right")

    for metric, d in report["quality"].items():
        b = f"{d['baseline_mean']:.3f}" if d.get("baseline_mean") is not None else "-"
        o = f"{d['optimized_mean']:.3f}" if d.get("optimized_mean") is not None else "-"
        delta = d["delta"]
        delta_str = f"{delta:+.3f}"
        if delta > 0.05:
            delta_str = f"[green]{delta_str}[/green]"
        elif delta < -0.05:
            delta_str = f"[red]{delta_str}[/red]"
        quality_table.add_row(metric, b, o, delta_str, str(d.get("n", 0)))
    console.print(quality_table)
