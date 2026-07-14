"""CLI: `contextops optimize / stats / recent / compare / eval`.

Built with Click + Rich for nice tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from contextops import __version__
from contextops.clients import EchoJudge, LiteLLMJudge
from contextops.cli_views import render_eval_report, render_optimization
from contextops.dataset import DatasetItem, load as load_dataset
from contextops.eval import compare as compare_prompts, evaluate_ab
from contextops.judge import list_metrics
from contextops.logger import DEFAULT_DB_PATH, Logger
from contextops.models import Prompt

console = Console()


@click.group()
@click.version_option(__version__, prog_name="contextops")
def main() -> None:
    """ContextOps — cache-aware prompt optimizer + local cost logger."""


@main.command()
@click.option("--system", default="", help="System prompt")
@click.option("--tools", default="", help="Tool definitions")
@click.option("--role", default="", help="Role / persona")
@click.option("--context", default="", help="Static context")
@click.option("--documents", default="", help="Retrieved documents")
@click.option("--history-file", type=click.Path(exists=True), default=None,
              help="JSONL file with {role, content} messages")
@click.option("--query", default="", help="User query")
@click.option("--model", default="gpt-4o", help="Target model")
@click.option("--goal", default="cache_friendly",
              type=click.Choice(["cache_friendly", "balanced", "quality"]))
@click.option("--from-json", "from_json", type=click.Path(exists=True), default=None,
              help="Load prompt from a JSON file")
def optimize(
    system: str,
    tools: str,
    role: str,
    context: str,
    documents: str,
    history_file: str | None,
    query: str,
    model: str,
    goal: str,
    from_json: str | None,
) -> None:
    """Optimize a prompt's section order for cache friendliness."""
    if from_json:
        data = json.loads(Path(from_json).read_text())
        p = Prompt(**data)
    else:
        history = []
        if history_file:
            for line in Path(history_file).read_text().splitlines():
                line = line.strip()
                if line:
                    history.append(json.loads(line))
        p = Prompt(
            system=system,
            tools=tools,
            role=role,
            context=context,
            documents=documents,
            history=history,
            query=query,
            model=model,
            goal=goal,
        )

    from contextops.optimizer import optimize as run_optimize

    result = run_optimize(p)
    render_optimization(result)


@main.command()
@click.option("--limit", default=100, help="How many recent calls to summarize")
@click.option("--db", type=click.Path(), default=None, help="Custom DB path")
def stats(limit: int, db: str | None) -> None:
    """Show aggregate stats from the local logger."""
    logger = Logger(Path(db) if db else None)
    s = logger.stats(limit=limit)

    table = Table(title=f"ContextOps Stats (last {s['limit']} calls)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total calls", str(s["total_calls"]))
    table.add_row("Prompt tokens", f"{s['total_prompt_tokens']:,}")
    table.add_row("Completion tokens", f"{s['total_completion_tokens']:,}")
    table.add_row("Cached tokens", f"{s['total_cached_tokens']:,}")
    table.add_row("Cache hit rate", f"{s['cache_hit_rate']:.1%}")
    table.add_row("Total cost (USD)", f"${s['total_cost_usd']:.4f}")
    table.add_row("Avg latency (ms)", f"{s['avg_latency_ms']:.1f}")

    console.print(table)

    if s["by_model"]:
        mtable = Table(title="By model")
        mtable.add_column("Model", style="bold")
        mtable.add_column("Calls", justify="right")
        mtable.add_column("Cost (USD)", justify="right")
        for row in s["by_model"]:
            mtable.add_row(row["model"], str(row["n"]), f"${row['cost']:.4f}")
        console.print(mtable)


@main.command()
@click.option("--limit", default=20, help="Number of recent calls to show")
@click.option("--db", type=click.Path(), default=None, help="Custom DB path")
def recent(limit: int, db: str | None) -> None:
    """Show recent logged calls."""
    logger = Logger(Path(db) if db else None)
    rows = logger.recent(limit=limit)

    table = Table(title=f"Last {len(rows)} calls")
    table.add_column("Timestamp", style="dim")
    table.add_column("Model")
    table.add_column("P tokens", justify="right")
    table.add_column("C tokens", justify="right")
    table.add_column("Cached", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Latency", justify="right")

    for r in rows:
        table.add_row(
            r["timestamp"],
            r["model"],
            f"{r['prompt_tokens']:,}",
            f"{r['completion_tokens']:,}",
            f"{r['cached_tokens']:,}",
            f"${r['cost_usd']:.4f}",
            f"{r['latency_ms']:.0f}ms" if r["latency_ms"] else "-",
        )
    console.print(table)


@main.command()
@click.option("--days", default=30, help="Number of days to show")
@click.option("--by-model", is_flag=True, help="Break down by model")
@click.option("--db", type=click.Path(), default=None, help="Custom DB path")
def trend(days: int, by_model: bool, db: str | None) -> None:
    """Show daily cache-hit-rate / cost / call-count trend."""
    logger = Logger(Path(db) if db else None)
    rows = logger.trend(days=days, by_model=by_model)
    if not rows:
        console.print("[dim]No calls in the selected window.[/dim]")
        return

    title = f"Daily trend (last {days} days{'  ·  by model' if by_model else ''})"
    table = Table(title=title, show_lines=False)
    table.add_column("Date", style="dim")
    if by_model:
        table.add_column("Model")
    table.add_column("Calls", justify="right")
    table.add_column("P tokens", justify="right")
    table.add_column("Cache hit", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Avg latency", justify="right")

    for r in rows:
        cells = [r["date"]]
        if by_model:
            cells.append(r["model"])
        cells += [
            f"{r['calls']:,}",
            f"{r['prompt_tokens']:,}",
            f"{r['cache_hit_rate']:.1%}",
            f"${r['cost_usd']:.4f}",
            f"{r['avg_latency_ms']:.0f}ms" if r["avg_latency_ms"] else "-",
        ]
        table.add_row(*cells)
    console.print(table)


@main.command()
@click.option("--format", "fmt", type=click.Choice(["csv", "jsonl"]), default="csv")
@click.option("--since", default=None,
              help="Limit to recent N days (e.g. 7). Default: all rows.")
@click.option("--out", type=click.Path(), required=True, help="Output file path")
@click.option("--db", type=click.Path(), default=None, help="Custom DB path")
def export(fmt: str, since: str | None, out: str, db: str | None) -> None:
    """Export logged calls to CSV or JSONL for analysis elsewhere."""
    logger = Logger(Path(db) if db else None)
    days = int(since) if since else None
    written = logger.export(fmt=fmt, path=Path(out), days=days)
    n = len(logger.recent(limit=10**9))
    console.print(f"[green]Exported {n} calls → {written}[/green]")


@main.command()
@click.argument("baseline_json", type=click.Path(exists=True))
@click.argument("optimized_json", type=click.Path(exists=True), required=False)
def compare(baseline_json: str, optimized_json: str | None) -> None:
    """Compare two prompt JSON files (or auto-optimize the baseline)."""
    baseline = Prompt(**json.loads(Path(baseline_json).read_text()))
    optimized = None
    if optimized_json:
        optimized = Prompt(**json.loads(Path(optimized_json).read_text()))
    report = compare_prompts(baseline, optimized)
    console.print_json(data=report)


@main.command(name="eval")
@click.option("--baseline", "baseline_json", type=click.Path(exists=True), required=True,
              help="Baseline prompt JSON")
@click.option("--optimized", "optimized_json", type=click.Path(exists=True), default=None,
              help="Optimized prompt JSON (auto-generated if omitted)")
@click.option("--dataset", "dataset_path", type=click.Path(exists=True), required=True,
              help="Golden dataset (.json/.jsonl/.csv)")
@click.option("--metrics", default="relevance,completeness,faithfulness",
              help=f"Comma-separated metrics. Available: {', '.join(list_metrics())}")
@click.option("--judge-model", default="gpt-4o-mini",
              help="Model used as judge")
@click.option("--echo", is_flag=True,
              help="Use offline echo judge (for demos/CI)")
@click.option("--run-fn", "run_fn_choice",
              type=click.Choice(["echo", "echo-fixed", "stub"]), default="echo",
              help="Which stub run_fn to use for the demo")
@click.option("--output", "output_path", type=click.Path(), default=None,
              help="Write full JSON report here")
def eval(
    baseline_json: str,
    optimized_json: str | None,
    dataset_path: str,
    metrics: str,
    judge_model: str,
    echo: bool,
    run_fn_choice: str,
    output_path: str | None,
) -> None:
    """Run an A/B eval: two prompts over a dataset, judged by LLM-as-judge."""
    baseline = Prompt(**json.loads(Path(baseline_json).read_text()))
    if optimized_json:
        optimized = Prompt(**json.loads(Path(optimized_json).read_text()))
    else:
        from contextops.optimizer import reorder
        optimized = reorder(baseline)

    dataset = load_dataset(dataset_path)
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    for m in metric_list:
        if m not in list_metrics():
            raise click.BadParameter(f"Unknown metric: {m}. Available: {list_metrics()}")

    judge = EchoJudge() if echo else _pick_real_judge()
    if isinstance(judge, LiteLLMJudge):
        console.print(f"[dim]Using real judge: {judge_model}[/dim]")
    else:
        console.print("[yellow]Using offline echo judge (no API calls)[/yellow]")

    run_fn = _pick_run_fn(run_fn_choice, dataset)

    progress_state = {"current": ""}

    def _on_progress(i: int, n: int, phase: str) -> None:
        progress_state["current"] = phase

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running A/B eval...", total=len(dataset) * 2 * len(metric_list))

        def _progress_wrapper(i: int, n: int, phase: str) -> None:
            progress.update(task, completed=i, description=f"[cyan]{phase}[/cyan]")

        report = evaluate_ab(
            baseline,
            optimized,
            run_fn=run_fn,
            dataset=dataset,
            metrics=metric_list,
            judge=judge,
            judge_model=judge_model,
            on_progress=_progress_wrapper,
        )

    render_eval_report(report)
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        console.print(f"\n[dim]Report written to {output_path}[/dim]")


def _pick_real_judge():
    try:
        return LiteLLMJudge()
    except RuntimeError:
        console.print(
            "[yellow]litellm not installed — falling back to offline EchoJudge. "
            "Install with: pip install 'contextops[integrations]'[/yellow]"
        )
        return EchoJudge()


def _pick_run_fn(choice: str, dataset: list[DatasetItem]):
    """Pick a stub run_fn. In a real app, the user passes their own LLM client."""
    if choice == "echo":
        # Pretend the model returns the expected answer with high accuracy.
        def run_fn(prompt_str: str) -> str:
            for item in dataset:
                if item.query and item.query in prompt_str:
                    return item.expected or "I don't know."
            return "I don't know."

        return run_fn
    if choice == "echo-fixed":
        # Pretend the model always returns a fixed, mediocre answer.
        def run_fn(prompt_str: str) -> str:
            return "Here is a generic answer."

        return run_fn
    if choice == "stub":
        def run_fn(prompt_str: str) -> str:
            return ""
        return run_fn
    raise ValueError(f"Unknown run_fn: {choice}")


def run_doctor(client, *, system_content: str, n_calls: int = 3) -> dict:
    """Send `n_calls` with the same stable prefix; report cache activation.

    Returns a dict: {calls: [{i, prompt_tokens, cached_tokens, cache_hit_rate}],
                     activated: bool, summary: str}.
    `activated=True` if any call after the first has cached_tokens > 0.

    Pure function (no I/O of its own beyond the client) so it's easy to unit
    test with a mock client.
    """
    calls = []
    for i in range(n_calls):
        resp = client.complete(
            model=getattr(client, "DEFAULT_MODEL", "claude-haiku-4.5"),
            messages=[{"role": "user", "content": f"Question {i}: what is 2+2?"}],
            temperature=0.0,
            max_tokens=32,
            system=system_content,
        )
        hit_rate = (resp.cached_tokens / resp.prompt_tokens) if resp.prompt_tokens > 0 else 0.0
        calls.append({
            "i": i,
            "prompt_tokens": resp.prompt_tokens,
            "cached_tokens": resp.cached_tokens,
            "cache_hit_rate": round(hit_rate, 3),
        })
    warm = [c for c in calls[1:] if c["cached_tokens"] > 0]
    activated = len(warm) >= 1
    if activated:
        best = max(c["cache_hit_rate"] for c in warm)
        summary = f"cache active: best repeat call served {best:.0%} from cache"
    else:
        summary = "cache NOT active — stable prefix may be under the provider's token minimum"
    return {"calls": calls, "activated": activated, "summary": summary}


@main.command()
@click.option("--provider", default="direct_anthropic",
              help="Provider to probe (direct_anthropic, openrouter, ...)")
@click.option("--n", default=3, help="Number of calls to send")
def doctor(provider: str, n: int) -> None:
    """Verify that prompt caching actually activates on your provider.

    Sends N calls with a large stable prefix and reports whether the provider
    serves calls 2+ from cache. This is the fastest way to confirm ContextOps
    will deliver savings for your workload before integrating.
    """
    console.rule("[bold]ContextOps Doctor — cache self-check")
    # Build the client from the bench registry.
    try:
        from contextops_bench.clients import get_client
        client = get_client(provider)
    except Exception as e:
        console.print(f"[red]Could not build provider '{provider}': {e}[/red]")
        console.print(
            "[dim]Run offline instead: pip install the bench extras and try "
            "`python -m contextops_bench smoke`.[/dim]"
        )
        return

    # Load the realistic preset (large enough to clear cache token minimums).
    try:
        from contextops_bench.prompt_factory import AGENT_PRESETS
        preset = AGENT_PRESETS["realistic"]
        system_content = preset["system"] + "\n\n" + preset["tools"]
    except Exception:
        console.print("[yellow]Could not load realistic preset; using a fallback prefix.[/yellow]")
        system_content = "You are a helpful assistant. " * 500

    console.print(f"[dim]Provider: {provider}  ·  sending {n} calls...[/dim]")
    result = run_doctor(client, system_content=system_content, n_calls=n)

    for c in result["calls"]:
        tag = "cold" if c["i"] == 0 else ("warm" if c["cached_tokens"] > 0 else "miss")
        console.print(
            f"  call {c['i'] + 1}: prompt={c['prompt_tokens']:>5}  "
            f"cached={c['cached_tokens']:>5}  hit={c['cache_hit_rate']:.0%}  [{tag}]"
        )
    if result["activated"]:
        console.print(f"\n[green]✓ {result['summary']}[/green]")
    else:
        console.print(f"\n[red]✗ {result['summary']}[/red]")
        console.print(
            "[dim]Anthropic requires ≥1024 tokens (Sonnet/Opus) or ≥2048 (Haiku) "
            "in the stable prefix to engage caching.[/dim]"
        )


@main.group()
def budget() -> None:
    """Set or show spend budgets (CLI-only alerts — v1, no webhooks)."""


@budget.command(name="set")
@click.option("--daily", type=float, default=None, help="Daily USD limit")
@click.option("--monthly", type=float, default=None, help="Monthly USD limit")
@click.option("--db", type=click.Path(), default=None, help="Custom DB path")
def budget_set(daily: float | None, monthly: float | None, db: str | None) -> None:
    """Configure daily/monthly spend limits."""
    if daily is None and monthly is None:
        raise click.UsageError("Provide --daily and/or --monthly.")
    logger = Logger(Path(db) if db else None)
    logger.set_budget(daily_usd=daily, monthly_usd=monthly)
    console.print("[green]Budget saved.[/green]")
    status = logger.budget_status()
    for k, v in status.items():
        if v:
            console.print(f"  {k}: ${v['spent']:.2f} / ${v['limit']:.2f} ({v['pct']:.0%})")


@budget.command(name="status")
@click.option("--db", type=click.Path(), default=None, help="Custom DB path")
def budget_status_cmd(db: str | None) -> None:
    """Show current spend against configured budgets."""
    logger = Logger(Path(db) if db else None)
    status = logger.budget_status()
    if not any(status.values()):
        console.print("[dim]No budgets configured. Set one with: contextops budget set --daily 50[/dim]")
        return
    for k, v in status.items():
        if v:
            color = "red" if v["over"] else ("yellow" if v["pct"] >= 0.8 else "green")
            console.print(
                f"  {k}: ${v['spent']:.2f} / ${v['limit']:.2f} ({v['pct']:.0%}) [{color}]"
            )


@main.command()
@click.option("--db", type=click.Path(), default=None, help="Custom DB path")
@click.confirmation_option(prompt="Are you sure you want to delete all logs?")
def reset(db: str | None) -> None:
    """Delete the local SQLite database."""
    path = Path(db) if db else DEFAULT_DB_PATH
    if path.exists():
        path.unlink()
        console.print(f"[red]Deleted {path}[/red]")
    else:
        console.print(f"Nothing to delete at {path}")


if __name__ == "__main__":
    main()