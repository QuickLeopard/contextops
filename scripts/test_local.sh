#!/usr/bin/env bash
# test_local.sh — runs the full local test + bench suite.
# Use this before committing or opening a PR.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> 1. Unit tests"
python -m pytest -v --tb=short

echo ""
echo "==> 2. Smoke bench (10 prompts, offline echo)"
python -m contextops_bench smoke

echo ""
echo "==> 3. 100-prompt bench (parallel=4, offline echo)"
python -m contextops_bench smoke --n 100 --parallel 4 --label smoke_100

echo ""
echo "==> 4. CLI sanity"
contextops optimize --system "You are a helpful assistant." --query "Hi" --model gpt-4o | head -10

echo ""
echo "✅ All checks passed. Ready to commit."
echo "   bench/results/ has your CSV + summary outputs if you want to inspect them."