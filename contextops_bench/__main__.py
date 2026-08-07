"""Run benchmarks from the CLI.

Usage:
    python -m bench.smoke
    python -m bench.local --provider ollama --n 100 --model llama3.1:8b
    python -m bench.cloud --provider openrouter --n 1000 --models gpt-4o-mini,claude-3.5-haiku
    python -m bench.run_all --provider echo --n 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from contextops.curator import CuratorConfig
from contextops.judge import JudgeClient
from contextops_bench.clients import get_client
from contextops_bench.curator_bench import (
    _ClientAsJudge,
    generate_curation_dataset,
    render_curator_summary,
    run_curator_bench,
)
from contextops_bench.prompt_factory import generate_many, EDGE_CASES, AGENT_PRESETS
from contextops_bench.runner import (
    render_summary,
    run_batch,
    save_csv,
    summarize,
)
from contextops_bench.breakdown import save_breakdown_csv


# Providers that actually charge for prompt cache. On these, the cache key
# MUST be stable across calls or every request becomes a cold cache_write.
# The `echo` / `ollama` / `lmstudio` providers don't have cache at all — for
# those we keep the random default since cache hit rate is meaningless there.
CACHE_BEARING_PROVIDERS = frozenset({
    "openrouter", "direct_anthropic", "direct_zen", "direct_openai", "direct_google",
})

# Sentinel for `--preset-agent` to opt out of the auto-default below. Lives
# outside AGENT_PRESETS so it doesn't pollute the preset dictionary semantics.
_PRESET_NONE = "none"


def _resolve_preset_args(args) -> tuple[str | None, str | None, str | None, str, str | None]:
    """Resolve (fixed_system, fixed_tools, fixed_role, preset_label, warning).

    `preset_label` is the human-readable name for log lines ("realistic",
    "realistic (auto-default)", "none", "no-preset"). `warning` is a multi-line
    string to print BEFORE the bench starts if a non-default fallback fires,
    or None for the happy path.

    Resolution rules (in order):
      1. `--preset-agent <name>` → load AGENT_PRESETS[name], merge with any
         explicit `--fixed-*` overrides. Role comes from the preset (so it's
         pinned — see v0.3.1 postmortem).
      2. `--preset-agent none` → explicit opt-out; use whatever `--fixed-*`
         flags the user passed (or randomized defaults if they passed none).
      3. No `--preset-agent`:
         a. Cache-bearing provider + no `--fixed-*` overrides → auto-apply
            `realistic` so the cache key stays constant. Print a warning so
            the user knows this happened (and how to opt out).
         b. Anything else → use whatever `--fixed-*` flags were passed (or
            randomized defaults if they passed none).
    """
    name = args.preset_agent
    fixed_system = args.fixed_system
    fixed_tools = args.fixed_tools
    fixed_role = None
    warning: str | None = None

    if name and name != _PRESET_NONE:
        preset = AGENT_PRESETS[name]
        fixed_system = fixed_system or preset["system"]
        fixed_tools = fixed_tools or preset["tools"]
        fixed_role = preset.get("role")  # role may be absent in some presets
        label = name
    elif name == _PRESET_NONE:
        label = "none"
    else:
        # No preset passed. Safety net for cache-bearing providers: without a
        # pinned role/system/tools, every prompt randomizes a piece of the
        # cache key and we get 0% cache hit rate while looking legitimate.
        # This is the exact failure mode v0.3.1 fixed for explicit
        # --preset-agent users; the no-preset path was still exposed.
        if args.provider in CACHE_BEARING_PROVIDERS and not (fixed_system or fixed_tools):
            preset = AGENT_PRESETS["realistic"]
            fixed_system = preset["system"]
            fixed_tools = preset["tools"]
            fixed_role = preset["role"]
            label = "realistic (auto-default)"
            warning = (
                "[bench] WARNING: no --preset-agent passed on cache-bearing provider "
                f"{args.provider!r}.\n"
                "[bench]          Auto-applying 'realistic' preset so the cache key stays "
                "constant.\n"
                "[bench]          To silence this: pass --preset-agent realistic "
                "(explicit) or --preset-agent none (opt out of the default)."
            )
        else:
            label = "no-preset"

    return fixed_system, fixed_tools, fixed_role, label, warning


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default="echo",
                        choices=["echo", "ollama", "lmstudio", "vllm", "tgi",
                                 "openrouter", "direct_anthropic", "direct_zen",
                                 "direct_openai", "direct_google"])
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
    parser.add_argument("--preset-agent", default=None,
                        choices=list(AGENT_PRESETS.keys()) + [_PRESET_NONE],
                        help="Load a preset system prompt + tool schema. Required for "
                             "cache_hit_rate to be non-zero on cloud providers since both "
                             "OpenAI and Anthropic have token minimums (1024 for Sonnet/Opus, "
                             "2048 for Haiku). On cache-bearing providers (openrouter, "
                             "direct_*), defaults to 'realistic' if unset — pass "
                             "'none' to opt out and use randomized prompts.")


def _make_client_and_model(args) -> tuple:
    client = get_client(args.provider)
    model = args.model
    if model is None:
        if args.provider == "ollama":
            models = client.list_models()
            model = models[0] if models else "llama3.1"
        elif args.provider == "lmstudio":
            model = "local-model"
        elif args.provider in ("vllm", "tgi"):
            models = client.list_models()
            model = models[0] if models else "default"
        elif args.provider == "openrouter":
            model = "openai/gpt-4o-mini"
        else:
            model = "echo-model"
    return client, model


def _execute(args, *, label: str, n: int, include_edge_cases: bool = False) -> int:
    client, model = _make_client_and_model(args)

    # Resolve preset / fixed_* args. May auto-default to 'realistic' on
    # cache-bearing providers and emit a warning — see _resolve_preset_args
    # docstring + v0.3.1 postmortem.
    fixed_system, fixed_tools, fixed_role, preset_label, warning = _resolve_preset_args(args)
    if warning:
        # Multi-line warning — print each line on its own to keep the bench
        # output readable.
        for line in warning.splitlines():
            print(line)
    print(f"[bench] preset-agent={preset_label}  "
          f"system~{len(fixed_system) if fixed_system else 0} chars  "
          f"tools~{len(fixed_tools) if fixed_tools else 0} chars  "
          f"role={fixed_role!r}")

    print(f"[bench] provider={args.provider}  model={model}  n={n}  parallel={args.parallel}")

    generated = list(generate_many(
        n=n, seed=42,
        fixed_system=fixed_system,
        fixed_tools=fixed_tools,
        fixed_model=model,
        fixed_role=fixed_role,
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
    # Per-prompt breakdown CSV (v0.3.3, item 2). Always write — even when
    # `breakdown` is empty, the file exists with just the header so downstream
    # tools have a stable contract.
    save_breakdown_csv(summary.get("breakdown", []), out_dir / f"{label}.breakdown.csv")
    return 0


def smoke(args) -> int:
    """Tiny offline bench — must run in <30s, no LLM."""
    args.provider = "echo"
    args.parallel = 1
    args.n = 10
    return _execute(args, label="smoke", n=10, include_edge_cases=True)


def local(args) -> int:
    """Local bench — Ollama / LM Studio."""
    # If --label was passed, use it verbatim; else default to local_<provider>.
    label = args.label or f"local_{args.provider}"
    return _execute(args, label=label, n=args.n)


def cloud(args) -> int:
    """Cloud bench — OpenRouter with multiple models."""
    models = (args.model or "openai/gpt-4o-mini").split(",")
    rc = 0
    for m in models:
        args.model = m
        # Honor --label: if user passed one, use it (optionally suffixed with
        # model name when running multiple models in one invocation).
        if args.label:
            label = (
                args.label if len(models) == 1
                else f"{args.label}_{m.replace('/', '_')}"
            )
        else:
            label = f"cloud_{m.replace('/', '_')}"
        rc |= _execute(args, label=label, n=args.n)
    return rc


def run_all(args) -> int:
    """Run smoke first, then whatever provider is requested."""
    rc = smoke(args)
    if args.provider != "echo":
        rc |= _execute(args, label=f"all_{args.provider}", n=args.n)
    return rc


# Defaults for the synthetic-dataset-only curator flags. Shared between
# `_add_curator_args` (so argparse defaults stay in sync) and
# `_check_dataset_mutual_exclusivity` (so it can detect an explicit
# non-default override without a separate hardcoded copy).
_SYNTHETIC_ONLY_DEFAULTS: dict[str, Any] = {
    "noise_ratio": 0.7,
    "chunks_per_item": 6,
    "dup_rate": 0.15,
    "seed": 42,
    "prompt_style": "qa_short",
    "embedder": "none",
}


def _add_curator_args(parser: argparse.ArgumentParser) -> None:
    """Args for the `curator` subcommand — deliberately separate from
    `_add_common_args`: the RAG curator bench has no system/tools/cache
    angle at all, so `--fixed-system`/`--fixed-tools`/`--preset-agent`
    would be dead, confusing options here.
    """
    parser.add_argument("--provider", default="echo",
                        choices=["echo", "ollama", "lmstudio", "vllm", "tgi",
                                 "openrouter", "direct_anthropic", "direct_zen",
                                 "direct_openai", "direct_google"],
                        help="LLM under test (the 'client' whose responses are compared "
                             "raw vs curated). Independent of judge selection below.")
    parser.add_argument("--model", default=None,
                        help="Model name (provider-specific). If unset, uses provider default.")
    parser.add_argument("--n", type=int, default=50, help="Number of synthetic dataset items")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent judge calls")
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--noise-ratio", type=float, default=_SYNTHETIC_ONLY_DEFAULTS["noise_ratio"],
                        help="Fraction of each item's chunks that are noise (0..1). "
                             "Mutually exclusive with --dataset.")
    parser.add_argument("--chunks-per-item", type=int, default=_SYNTHETIC_ONLY_DEFAULTS["chunks_per_item"],
                        help="Mutually exclusive with --dataset.")
    parser.add_argument("--dup-rate", type=float, default=_SYNTHETIC_ONLY_DEFAULTS["dup_rate"],
                        help="Probability of injecting a near-duplicate relevant chunk. "
                             "Mutually exclusive with --dataset.")
    parser.add_argument("--seed", type=int, default=_SYNTHETIC_ONLY_DEFAULTS["seed"],
                        help="Mutually exclusive with --dataset.")
    parser.add_argument("--prompt-style", default=_SYNTHETIC_ONLY_DEFAULTS["prompt_style"],
                        help="Synthetic dataset prompt-style profile: qa_short (default), "
                             "long_document, code, structured, multilingual, multi_turn, "
                             "adversarial_noise. See contextops_bench.curator_bench."
                             "PROMPT_STYLES. Mutually exclusive with --dataset.")
    parser.add_argument("--embedder", default=_SYNTHETIC_ONLY_DEFAULTS["embedder"],
                        choices=["none", "tfidf", "openai"],
                        help="Compute REAL similarity scores for the synthetic dataset "
                             "instead of synthetic random ranges: 'tfidf' (offline, free) "
                             "or 'openai' (requires OPENAI_API_KEY, costs a small amount "
                             "per run). Default 'none' keeps the original synthetic "
                             "similarity ranges. Mutually exclusive with --dataset (real "
                             "data already has its own similarity scores).")
    parser.add_argument("--dataset", default=None, type=Path,
                        help="Path to a JSON file of real/production dataset items "
                             "(list of {query, expected?, chunks:[{text,similarity,...}]}) "
                             "instead of generating a synthetic dataset — see "
                             "contextops_bench.curator_bench.load_curation_dataset. "
                             "Mutually exclusive with --noise-ratio/--chunks-per-item/"
                             "--dup-rate/--seed/--prompt-style/--embedder.")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="curate() strict cutoff score")
    parser.add_argument("--dedup-threshold", type=float, default=0.9,
                        help="curate() Jaccard dedup threshold")
    parser.add_argument("--metrics", default="relevance,completeness",
                        help="Comma-separated judge metrics")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens for the LLM-under-test's answers. Short answers "
                             "starve both the judge and context_precision of signal.")
    parser.add_argument("--judge-model", default=None,
                        help="Model used to judge. Defaults to the same model as --model "
                             "(self-judging). Can point at a DIFFERENT model on the same "
                             "provider (e.g. a stronger sibling model) to partially reduce "
                             "self-judging bias without needing a separate API key. "
                             "Ignored if --echo-judge or --litellm-judge is set.")
    parser.add_argument("--echo-judge", action="store_true",
                        help="Use offline echo judge for scoring (no judge API calls). "
                             "Independent of --provider — you can bench a real LLM "
                             "under test while still using a free/offline judge.")
    parser.add_argument("--litellm-judge", action="store_true",
                        help="Use LiteLLMJudge (requires a separate litellm-compatible "
                             "credential, e.g. OPENAI_API_KEY) instead of self-judging "
                             "via the same client/key as --provider.")


def _select_curator_judge(
    *, client: Any, model: str, provider: str,
    echo_judge: bool, litellm_judge: bool, judge_model: str | None,
) -> tuple[JudgeClient, str]:
    """Pick the `JudgeClient` + human-readable label for the curator bench.

    Precedence: `--echo-judge` > `--litellm-judge` > provider=="echo"
    fallback > default self-judging via `_ClientAsJudge(client)`. Pulled
    out of `curator()` so it's independently unit-testable without a
    real network call or argparse plumbing.
    """
    from contextops.clients import EchoJudge, LiteLLMJudge

    resolved_judge_model = judge_model or model

    if echo_judge:
        return EchoJudge(), "echo"
    if litellm_judge:
        try:
            return LiteLLMJudge(), f"litellm:{resolved_judge_model}"
        except RuntimeError:
            print("[bench] litellm not installed — falling back to offline EchoJudge. "
                  "Install with: pip install 'contextops[integrations]'")
            return EchoJudge(), "echo (fallback)"
    if provider == "echo":
        # Self-judging against EchoClient's fixed "echo" text is meaningless
        # (every response is identical) — fall back to the offline judge.
        return EchoJudge(), "echo"
    # Default: self-judge using the SAME real client/key already configured
    # for the LLM-under-test. --judge-model can point at a different model
    # on the same provider to partially reduce self-judging bias without a
    # separate credential.
    label = "self" if resolved_judge_model == model else f"self:{resolved_judge_model}"
    return _ClientAsJudge(client), label


def _check_dataset_mutual_exclusivity(
    *, dataset: Path | None, noise_ratio: float, chunks_per_item: int,
    dup_rate: float, seed: int, prompt_style: str, embedder: str,
) -> None:
    """`--dataset` (real production data, own similarity scores) is
    mutually exclusive with every synthetic-generator-only flag. Raises
    `ValueError` naming exactly which flags were explicitly overridden,
    rather than silently ignoring them — a silently-ignored `--embedder`
    in particular would otherwise look like it worked while quietly never
    touching the user's real similarity scores.
    """
    if dataset is None:
        return
    provided = {
        "noise_ratio": noise_ratio, "chunks_per_item": chunks_per_item,
        "dup_rate": dup_rate, "seed": seed, "prompt_style": prompt_style,
        "embedder": embedder,
    }
    flag_names = {
        "noise_ratio": "--noise-ratio", "chunks_per_item": "--chunks-per-item",
        "dup_rate": "--dup-rate", "seed": "--seed", "prompt_style": "--prompt-style",
        "embedder": "--embedder",
    }
    overrides = [
        flag_names[key] for key, value in provided.items()
        if value != _SYNTHETIC_ONLY_DEFAULTS[key]
    ]
    if overrides:
        raise ValueError(
            f"--dataset is mutually exclusive with: {', '.join(overrides)}. "
            "--dataset supplies its own real chunks/similarity/query text; "
            "remove these synthetic-generator flags (or don't pass --dataset)."
        )


def _load_or_generate_curator_dataset(
    *, dataset: Path | None, n: int, noise_ratio: float, chunks_per_item: int,
    dup_rate: float, seed: int, prompt_style: str, embedder: str,
) -> list:
    """Resolve the curator bench's dataset: `load_curation_dataset(dataset)`
    if `--dataset` was given, else `generate_curation_dataset(...)` with the
    requested style/embedder. Pulled out of `curator()` for testability
    (mirrors `_select_curator_judge`).
    """
    from contextops_bench.curator_bench import load_curation_dataset
    from contextops_bench.embedders import get_embedder

    _check_dataset_mutual_exclusivity(
        dataset=dataset, noise_ratio=noise_ratio, chunks_per_item=chunks_per_item,
        dup_rate=dup_rate, seed=seed, prompt_style=prompt_style, embedder=embedder,
    )
    if dataset is not None:
        return load_curation_dataset(str(dataset))

    embedder_obj = get_embedder(embedder) if embedder != "none" else None
    return generate_curation_dataset(
        n, noise_ratio=noise_ratio, chunks_per_item=chunks_per_item,
        dup_rate=dup_rate, seed=seed, style=prompt_style, embedder=embedder_obj,
    )


def curator(args) -> int:
    """Bench the RAG curator: raw (uncurated) vs curated chunks on a real LLM,
    measuring token/cost savings AND answer-quality impact.
    """
    client, model = _make_client_and_model(args)
    judge_model = args.judge_model or model
    judge, judge_label = _select_curator_judge(
        client=client, model=model, provider=args.provider,
        echo_judge=args.echo_judge, litellm_judge=args.litellm_judge,
        judge_model=args.judge_model,
    )

    dataset_desc = f"dataset={args.dataset}" if args.dataset else (
        f"noise_ratio={args.noise_ratio}  style={args.prompt_style}  embedder={args.embedder}"
    )
    print(f"[bench] curator: provider={args.provider}  model={model}  n={args.n}  "
          f"{dataset_desc}  judge={judge_label}  max_tokens={args.max_tokens}")

    items = _load_or_generate_curator_dataset(
        dataset=args.dataset,
        n=args.n,
        noise_ratio=args.noise_ratio,
        chunks_per_item=args.chunks_per_item,
        dup_rate=args.dup_rate,
        seed=args.seed,
        prompt_style=args.prompt_style,
        embedder=args.embedder,
    )
    config = CuratorConfig(threshold=args.threshold, dedup_threshold=args.dedup_threshold)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    summary = run_curator_bench(
        items,
        client=client,
        model=model,
        judge=judge,
        metrics=metrics,
        config=config,
        judge_model=judge_model,
        max_workers=args.parallel,
        max_tokens=args.max_tokens,
        judge_label=judge_label,
    )

    label = args.label or f"curator_{args.provider}"
    text = render_curator_summary(summary, label)
    print()
    print(text)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.curator_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[bench] curator summary: {out_path}")
    return 0


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

    p_curator = sub.add_parser(
        "curator", help="Bench the RAG curator: raw vs curated chunks on a real LLM"
    )
    _add_curator_args(p_curator)
    p_curator.set_defaults(func=curator)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())