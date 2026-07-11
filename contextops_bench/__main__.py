"""Run benchmarks from the CLI.

Usage:
    python -m contextops_bench smoke
    python -m contextops_bench local --provider ollama --n 100 --model llama3.1:8b
    python -m contextops_bench direct --provider direct_anthropic --n 100
    python -m contextops_bench cloud --provider openrouter --n 1000 --model gpt-4o-mini,anthropic/claude-haiku-4.5
    python -m contextops_bench run_all --provider echo --n 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from contextops_bench.clients import CLIENTS, get_client
from contextops_bench.prompt_factory import generate_many, EDGE_CASES, AGENT_PRESETS
from contextops_bench.runner import (
    render_summary,
    run_batch,
    save_csv,
    summarize,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="echo",
                        choices=sorted(CLIENTS))
    parser.add_argument("--model", default=None,
                        help="Model name (provider-specific). If unset, uses provider default. "
                             "For `cloud` subcommand, comma-separated runs each model.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--fixed-system", default=None,
                        help="Lock the system prompt across all generated prompts "
                             "(simulates a real agent workload — enables cache hits)")
    parser.add_argument("--fixed-tools", default=None,
                        help="Lock the tools section across all generated prompts")
    parser.add_argument("--preset-agent", default=None, choices=list(AGENT_PRESETS.keys()),
                        help="Load a preset system prompt + tool schema. "
                             "Required for cache_hit_rate to be non-zero on cloud providers "
                             "since both OpenAI and Anthropic have token minimums "
                             "(1024 for Sonnet/Opus, 2048 for Haiku). "
                             "Overrides --fixed-system/--fixed-tools if set.")


def _make_client_and_model(args) -> tuple:
    client = get_client(args.provider)
    model = args.model
    if model is None:
        if args.provider == "ollama":
            models = client.list_models()
            model = models[0] if models else "llama3.1"
        elif args.provider == "lmstudio":
            model = "local-model"
        elif args.provider == "openrouter":
            model = "openai/gpt-4o-mini"
        else:
            model = "echo-model"
    return client, model


def _resolve_preset(args) -> tuple[str | None, str | None]:
    """Resolve --preset-agent / --fixed-system / --fixed-tools into (system, tools)."""
    fixed_system = args.fixed_system
    fixed_tools = args.fixed_tools
    if args.preset_agent:
        preset = AGENT_PRESETS[args.preset_agent]
        fixed_system = fixed_system or preset["system"]
        fixed_tools = fixed_tools or preset["tools"]
        print(f"[bench] preset-agent={args.preset_agent}  "
              f"system~{len(fixed_system)} chars  tools~{len(fixed_tools)} chars")
    return fixed_system, fixed_tools


def _build_prompt_list(
    *, n: int, fixed_system: str | None, fixed_tools: str | None, model: str,
    include_edge_cases: bool,
) -> tuple[list, list[int]]:
    """Generate `n` prompts, optionally interleaving edge cases.

    Returns (prompts, edge_ids) where edge_ids is the list of prompt indices
    that are degenerate edge cases (excluded from headline stats). Edge cases
    are split half-before / half-after the generated prompts so half land in
    the optimized phase and half in the baseline phase of `run_batch`.
    """
    generated = list(generate_many(
        n=n, seed=42,
        fixed_system=fixed_system,
        fixed_tools=fixed_tools,
        fixed_model=model,
    ))
    if not include_edge_cases:
        return generated, []
    half_edge = len(EDGE_CASES) // 2
    prompts = list(EDGE_CASES[:half_edge]) + generated + list(EDGE_CASES[half_edge:])
    edge_ids = list(range(0, half_edge)) + list(range(n + half_edge, n + len(EDGE_CASES)))
    return prompts, edge_ids


def _write_artifacts(out_dir: Path, label: str, results, summary: dict) -> Path:
    """Write CSV + summary JSON to `out_dir`. Returns the CSV path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{label}.csv"
    save_csv(results, csv_path)
    (out_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2))
    return csv_path


def _execute(args, *, label: str, n: int, include_edge_cases: bool = False) -> int:
    client, model = _make_client_and_model(args)
    fixed_system, fixed_tools = _resolve_preset(args)
    print(f"[bench] provider={args.provider}  model={model}  n={n}  parallel={args.parallel}")

    prompts, edge_ids = _build_prompt_list(
        n=n, fixed_system=fixed_system, fixed_tools=fixed_tools,
        model=model, include_edge_cases=include_edge_cases,
    )

    def _on_progress(done: int, total: int) -> None:
        if done % max(1, total // 10) == 0 or done == total:
            print(f"  progress: {done}/{total}", flush=True)

    t0 = time.time()
    results = run_batch(
        prompts,
        client=client,
        parallel=args.parallel,
        label=label,
        on_progress=_on_progress,
    )
    elapsed = time.time() - t0

    summary = summarize(results, exclude_ids=set(edge_ids))
    csv_path = _write_artifacts(args.out, label, results, summary)

    print()
    print(render_summary(summary, label))
    print(f"\n[bench] elapsed: {elapsed:.1f}s  ({elapsed / max(1, n):.2f}s/call)")
    print(f"[bench] csv:     {csv_path}")
    return 0


def smoke(args) -> int:
    """Tiny offline bench — must run in <30s, no LLM.

    Operates on a shallow copy of `args` so mutations don't leak back to the
    caller (critical for `run_all`, which calls `smoke` then reuses `args`).
    """
    args = argparse.Namespace(**vars(args))  # shallow copy
    args.provider = "echo"
    args.parallel = 1
    args.n = 10
    return _execute(args, label="smoke", n=10, include_edge_cases=True)


def local(args) -> int:
    """Local bench — Ollama / LM Studio."""
    return _execute(args, label=f"local_{args.provider}", n=args.n)


def cloud(args) -> int:
    """Cloud bench — OpenRouter with multiple models."""
    models = (args.model or "openai/gpt-4o-mini").split(",")
    rc = 0
    for m in models:
        args.model = m
        rc |= _execute(args, label=f"cloud_{m.replace('/', '_')}", n=args.n)
    return rc


def run_all(args) -> int:
    """Run smoke first, then whatever provider is requested."""
    rc = smoke(args)
    if args.provider != "echo":
        rc |= _execute(args, label=f"all_{args.provider}", n=args.n)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(prog="bench", description="ContextOps benchmark runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke", help="10 prompts, no LLM, <30s, CI-friendly")
    _add_common_args(p_smoke)
    p_smoke.set_defaults(func=smoke)

    p_local = sub.add_parser("local", help="Run against local LLM (Ollama/LM Studio)")
    _add_common_args(p_local)
    p_local.set_defaults(func=local)

    p_direct = sub.add_parser("direct", help="Run against direct API (no OpenRouter) — definitive cache signal")
    _add_common_args(p_direct)
    p_direct.set_defaults(func=local)

    p_cloud = sub.add_parser("cloud", help="Run against OpenRouter (1+ models)")
    _add_common_args(p_cloud)
    p_cloud.set_defaults(func=cloud)

    p_all = sub.add_parser("run_all", help="Smoke + selected provider")
    _add_common_args(p_all)
    p_all.set_defaults(func=run_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())