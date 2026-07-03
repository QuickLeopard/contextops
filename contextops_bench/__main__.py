"""Run benchmarks from the CLI.

Usage:
    python -m bench.smoke
    python -m bench.local --provider ollama --n 100 --model llama3.1:8b
    python -m bench.cloud --provider openrouter --n 1000 --models gpt-4o-mini,claude-3.5-haiku
    python -m bench.run_all --provider echo --n 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contextops_bench.clients import get_client
from contextops_bench.prompt_factory import generate_many, EDGE_CASES
from contextops_bench.runner import (
    render_summary,
    run_batch,
    save_csv,
    summarize,
)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="echo",
                        choices=["echo", "ollama", "lmstudio", "openrouter"])
    parser.add_argument("--model", default=None,
                        help="Model name (provider-specific). If unset, uses provider default.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--fixed-system", default=None,
                        help="Lock the system prompt across all generated prompts "
                             "(simulates a real agent workload — enables cache hits)")
    parser.add_argument("--fixed-tools", default=None,
                        help="Lock the tools section across all generated prompts")


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


def _execute(args, *, label: str, n: int, include_edge_cases: bool = False) -> int:
    client, model = _make_client_and_model(args)

    # Override model on client if possible
    if hasattr(client, "default_model"):
        model = client.default_model  # type: ignore[attr-defined]

    print(f"[bench] provider={args.provider}  model={model}  n={n}  parallel={args.parallel}")

    generated = list(generate_many(
        n=n, seed=42,
        fixed_system=args.fixed_system,
        fixed_tools=args.fixed_tools,
    ))
    prompts: list = []
    edge_ids: list[int] = []

    if include_edge_cases:
        # Interleave edge cases evenly with generated prompts so half go to
        # optimized phase and half to baseline phase. With 10 edge cases and
        # n generated, drop ~half of edge cases into the first half of the
        # prompts list and the rest into the second half.
        half_edge = len(EDGE_CASES) // 2
        # Take the first half_edge edge cases and prepend them to generated.
        # Take the second half_edge edge cases and append them to generated.
        prompts = list(EDGE_CASES[:half_edge]) + generated + list(EDGE_CASES[half_edge:])
        # Edge case prompt_ids: 0..half_edge-1 and n+half_edge..n+len(EDGE_CASES)-1
        edge_ids = list(range(0, half_edge)) + list(range(n + half_edge, n + len(EDGE_CASES)))
    else:
        prompts = generated

    def _on_progress(done: int, total: int) -> None:
        if done % max(1, total // 10) == 0 or done == total:
            print(f"  progress: {done}/{total}", flush=True)

    t0 = __import__("time").time()
    results = run_batch(
        prompts,
        client=client,
        parallel=args.parallel,
        label=label,
        on_progress=_on_progress,
    )
    elapsed = __import__("time").time() - t0

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{label}.csv"
    save_csv(results, csv_path)

    summary = summarize(results, exclude_ids=set(edge_ids))
    text = render_summary(summary, label)
    print()
    print(text)
    print(f"\n[bench] elapsed: {elapsed:.1f}s  ({elapsed / max(1, n):.2f}s/call)")
    print(f"[bench] csv:     {csv_path}")

    (out_dir / f"{label}.summary.json").write_text(
        __import__("json").dumps(summary, indent=2)
    )
    return 0


def smoke(args) -> int:
    """Tiny offline bench — must run in <30s, no LLM."""
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