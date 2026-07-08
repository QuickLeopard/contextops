# v0.3.3 — Bench Reliability Patch

Status: **Locked**, awaiting implementation. Decisions recorded 2026-07-08.

## Goal

Make the bench harness produce trustworthy A/B measurements and lower the barrier
to cross-provider cache-mechanics comparison. No new providers, no architectural
changes — pure measurement quality.

## Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Stat method for A/B cost-delta | **Bootstrap CI** (10k samples, stdlib only, deterministic seed) |
| 2 | Effect size center | **Median** of paired Δ cost, expressed as % of median baseline |
| 3 | Replay `--max-cost-usd` cap | **No cap.** User responsibility. Flag dropped from CLI. |
| 4 | CHANGELOG section | **Keep `[Unreleased]`** until acceptance criteria are met |

## In scope

### 1. Bootstrap CI on cost delta (`contextops_bench.stats`)

- New module `contextops_bench/stats.py` with:
  - `bootstrap_ci(values, n_boot=10_000, ci=0.95, seed=0) -> tuple[float, float]`
  - `effect_size_pct(optimized, baseline) -> float` (median-based)
- Stdlib only (`statistics`, `random`). No scipy.
- Auto-scale `n_boot` to `1_000` when paired N < 20 to keep wall time sane.
- Added to `summarize()` output:
  - `cost_delta_ci_low_usd`
  - `cost_delta_ci_high_usd`
  - `effect_size_pct`
- Rendered in summary as: `[low, high] @ 95%, effect: X.X%`.

### 2. Per-prompt breakdown (`contextops_bench.breakdown`)

- New module `contextops_bench/breakdown.py` with
  `per_prompt_breakdown(pairs) -> list[dict]`.
- Top-N prompts by `|Δ cost|` rendered as a Rich table at the end of summary.
- Saved to `bench/results/<label>.breakdown.csv`.
- Columns: `prompt_id, model, prompt_tokens, baseline_cost, optimized_cost,
  delta_cost, delta_pct, baseline_cache_hit, optimized_cache_hit`.

### 3. `bench replay <csv>` subcommand

- New module `contextops_bench/replay.py` with
  `replay_from_csv(path, client) -> list[BenchResult]`.
- Re-runs only the LLM call, reusing prompt structures saved in the source CSV.
- Pair ordering preserved (source row N → still row N in replay), so the
  cost-delta CI is directly comparable across replays.
- Property test: the `system + tools + role` prefix hash must be byte-identical
  between original and replayed prompts.
- Writes `<label>.replay.csv` and `<label>.replay.summary.json`.
- **No cost cap.** README and CLI help text warn the user that they are
  responsible for cost control — especially on cloud replays.

## Out of scope (deferred)

- New providers (DeepSeek, Bedrock) — separate PR.
- Web UI / dashboard — separate plan.
- RAG curator — v0.4.
- Latency regression CI — already covered by v0.3.2 bench-regression gate.
- New presets in `AGENT_PRESETS` — separate PR if needed.

## File layout

```
contextops_bench/
  stats.py         NEW     bootstrap_ci(), effect_size_pct()
  breakdown.py     NEW     per_prompt_breakdown(pairs)
  replay.py        NEW     replay_from_csv(path, client)
  runner.py        EDIT    summarize() emits CI keys + breakdown rows
  __main__.py      EDIT    add `bench breakdown` + `bench replay` subcommands

tests/
  test_bench_stats.py     NEW     ~4 tests
  test_bench_breakdown.py NEW     ~3 tests
  test_bench_replay.py    NEW     ~3 tests
  test_bench_unit.py      UNCHANGED

docs/
  PLAN_v0.3.3.md          NEW     this file

CHANGELOG.md [Unreleased] FILLED IN
```

## Acceptance criteria

1. `summary["optimized"]["cost_delta_ci_low_usd"]`, `["cost_delta_ci_high_usd"]`,
   `["effect_size_pct"]` exist in `.summary.json`.
2. Bootstrap CI collapses to a point when all paired deltas are equal (sanity
   unit test).
3. Breakdown CSV at `bench/results/<label>.breakdown.csv` exists with the
   documented columns.
4. Replay: `bench replay <csv>` produces rows where `prompt_id` and pair
   ordering match the source; `system+tools+role` prefix hash is byte-identical
   between source and replayed prompts (property test).
5. **≥63 tests pass** (53 + ~10 new). All existing tests still green.
6. CI bench regression gate still passes (no regression on the v0.3.2 gate).

## Migration / backwards compatibility

- Additive. New summary JSON keys; existing keys unchanged.
- New CLI subcommands; existing commands unchanged.
- SQLite logger schema unchanged.
- Existing bench results (`.csv`, `.summary.json`) remain readable.

## Risks

- Bootstrap n_boot=10k on tiny N is overkill — auto-scale to 1k when N<20.
- No cost cap on replay means a stale CSV can blow a cloud budget — README must
  warn loudly; CLI help text reinforces it.
