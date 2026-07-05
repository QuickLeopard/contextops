#!/usr/bin/env python3
"""Bench regression gate for CI.

Runs the bench against a real (or echo) LLM provider with the `realistic` agent
preset — the exact configuration that exposed the v0.3.1 cache-key bug. Exits
non-zero if the optimized arm's `cache_hit_rate_p50` is below the configured
threshold, or if the bench run itself fails.

Env vars
---------
BENCH_PROVIDER   default: "direct_zen"
BENCH_MODEL      default: "claude-sonnet-4-6"
BENCH_N          default: 5  (small for fast CI; raise in workflow_dispatch input)
BENCH_THRESHOLD  default: 0.50  (conservative; e.g. 0.90 for stricter gating)
BENCH_LABEL      default: derived from provider + n

The provider's API key must be set: ZEN_API_KEY, ANTHROPIC_API_KEY, or
OPENAI_API_KEY. The CI workflow should `exit 0` BEFORE invoking this script
if no key is configured — otherwise this script will surface a clean ValueError
from the client factory.

Why this exists
---------------
Before this gate, the v0.3.1 realistic-preset cache-key bug (role randomly
rotated the cache prefix) passed all 39 unit tests because tests use
EchoClient (no real network, no real cache). This script runs the bench
against a real provider with real cache semantics — so a cache-key-rotating
regression surfaces as `cache_hit_rate_p50 < BENCH_THRESHOLD` and fails the PR.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    provider = os.environ.get("BENCH_PROVIDER", "direct_zen")
    model = os.environ.get("BENCH_MODEL", "claude-sonnet-4-6")
    n = int(os.environ.get("BENCH_N", "5"))
    threshold = float(os.environ.get("BENCH_THRESHOLD", "0.50"))
    label = os.environ.get("BENCH_LABEL", f"ci_gate_{provider}_{n}")

    print(
        f"[bench-gate] provider={provider}  model={model}  n={n}  "
        f"threshold={threshold:.2f}  label={label}",
        flush=True,
    )

    # Find the bench dir relative to this script (scripts/ci_bench_gate.py).
    repo_root = Path(__file__).resolve().parent.parent
    bench_results = repo_root / "bench" / "results"
    bench_results.mkdir(parents=True, exist_ok=True)

    # Run the bench via `python -m` so it picks up the installed package and the
    # same venv as the caller. Use `sys.executable` so we honour the active venv.
    cmd = [
        sys.executable,
        "-m", "contextops_bench", "cloud",
        "--provider", provider,
        "--model", model,
        "--n", str(n),
        "--preset-agent", "realistic",
        "--label", label,
        "--parallel", "1",
    ]
    print(f"[bench-gate] running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(repo_root))
    if rc != 0:
        print(
            f"[bench-gate] FAIL: bench run exited with code {rc}. "
            f"This usually means API auth failed or the network was unreachable. "
            f"Re-run locally with the same command to debug.",
            file=sys.stderr,
            flush=True,
        )
        return rc

    # The bench writes a {label}.summary.json next to {label}.csv.
    summary_path = bench_results / f"{label}.summary.json"
    if not summary_path.exists():
        print(
            f"[bench-gate] FAIL: no summary file at {summary_path}. "
            f"The bench finished without errors but didn't write summary.json — "
            f"this is a bench bug.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as e:
        print(
            f"[bench-gate] FAIL: summary.json at {summary_path} is malformed: {e}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    optimized = summary.get("optimized", {})
    p50 = optimized.get("cache_hit_rate_p50", 0.0)
    errors = optimized.get("errors", 0)
    n_actual = optimized.get("n", 0)
    print(
        f"[bench-gate] optimized: n={n_actual} errors={errors} "
        f"cache_hit_rate_p50={p50:.3f}  threshold={threshold:.3f}",
        flush=True,
    )

    if errors >= n_actual / 2:
        print(
            f"[bench-gate] FAIL: {errors}/{n_actual} optimized calls errored. "
            f"Bench regression is masked by upstream errors. Check API key + network.",
            file=sys.stderr,
            flush=True,
        )
        return 3

    if p50 < threshold:
        print(
            f"[bench-gate] FAIL: cache_hit_rate_p50={p50:.3f} < threshold={threshold:.3f}. "
            f"A regression likely rotated the cache key (system/tools/role) "
            f"or disabled cache_control. See docs/POSTMORTEM_realistic_cache.md.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print("[bench-gate] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
