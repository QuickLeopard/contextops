#!/usr/bin/env python3
"""CI gate: block newly added/modified `bench/results/*.summary.json` files
that fail the deterministic quality gate (`contextops_bench.quality.
evaluate_quality_gate`) from ever reaching `main` / the public dashboard.

This is deliberately separate from `scripts/ci_bench_gate.py` (which runs a
FRESH bench call against a real provider to catch cache-key regressions).
This script does no network calls at all — it only re-evaluates the quality
gate against whatever `*.summary.json` files a PR is adding or changing, so
it can run on every PR with no API key required.

Usage
-----
    python scripts/ci_bench_data_gate.py [--base-ref REF]

`--base-ref` (or the `GITHUB_BASE_REF` / `CI_BENCH_GATE_BASE_REF` env vars,
in that priority order) selects the git ref to diff against. Defaults to
"origin/main". Exits 0 with "no changed summary files" if the diff is empty
(this is normal for most PRs) — exits 1 if any changed/added
`bench/results/*.summary.json` fails `evaluate_quality_gate`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from contextops_bench.quality import evaluate_quality_gate  # noqa: E402
from scripts.generate_dashboard import _parse_filename, _runtime_provider_model  # noqa: E402


def changed_summary_files(base_ref: str) -> list[Path]:
    """Return absolute paths of `bench/results/*.summary.json` files added or
    modified relative to `base_ref` (git diff, added+modified only — deleted
    files are irrelevant to this gate).
    """
    try:
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", base_ref.removeprefix("origin/")],
            cwd=REPO_ROOT, capture_output=True, check=False,
        )
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=AM", base_ref, "HEAD",
             "--", "bench/results/*.summary.json"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[data-gate] WARN: git diff against {base_ref!r} failed ({e}); "
              f"treating as no changed files.", file=sys.stderr)
        return []
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line.strip()]


def check_summary_file(path: Path) -> tuple[bool, list[str]]:
    """Evaluate one summary.json's quality gate.

    Returns (ok, reasons). `ok=False` means this file must not merge as-is;
    `reasons` explains why (empty list if `ok=True` or the file is
    malformed/missing — malformed files are reported as failures too, since
    a broken summary.json shouldn't reach the dashboard either).
    """
    if not path.exists():
        return False, [f"{path} does not exist (deleted after diff was computed?)"]
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return False, [f"{path} is not valid JSON: {e}"]
    if not isinstance(data, dict):
        return False, [f"{path} does not contain a JSON object"]

    provider, model = _parse_filename(path)
    runtime_provider, runtime_model = _runtime_provider_model(provider, model, data)
    quality = evaluate_quality_gate(data, provider=runtime_provider, model=runtime_model)
    if quality["verified"]:
        return True, []
    return False, quality["reasons"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default=None, help="git ref to diff against")
    args = parser.parse_args(argv)

    base_ref = (
        args.base_ref
        or os.environ.get("CI_BENCH_GATE_BASE_REF")
        or (f"origin/{os.environ['GITHUB_BASE_REF']}" if os.environ.get("GITHUB_BASE_REF") else None)
        or "origin/main"
    )
    print(f"[data-gate] diffing bench/results/*.summary.json against {base_ref!r}", flush=True)

    changed = changed_summary_files(base_ref)
    if not changed:
        print("[data-gate] PASS: no changed/added bench/results/*.summary.json files.", flush=True)
        return 0

    failures: list[tuple[Path, list[str]]] = []
    for path in changed:
        ok, reasons = check_summary_file(path)
        rel = path.relative_to(REPO_ROOT)
        if ok:
            print(f"[data-gate] OK: {rel}", flush=True)
        else:
            print(f"[data-gate] FAIL: {rel}", file=sys.stderr, flush=True)
            for reason in reasons:
                print(f"[data-gate]   - {reason}", file=sys.stderr, flush=True)
            failures.append((path, reasons))

    if failures:
        print(
            f"\n[data-gate] {len(failures)}/{len(changed)} changed summary file(s) "
            f"failed the quality gate. Fix or remove them before merging — "
            f"unverified bench data must not reach the public dashboard.",
            file=sys.stderr, flush=True,
        )
        return 1

    print(f"[data-gate] PASS: all {len(changed)} changed summary file(s) verified.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
