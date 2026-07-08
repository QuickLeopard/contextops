# Changelog

All notable changes to ContextOps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Per-prompt breakdown** (`contextops.breakdown`, new module `contextops_bench/breakdown.py`): top-N prompts by `|Δ cost|` rendered as a Rich table at the end of the bench summary, and also saved to `bench/results/<label>.breakdown.csv`. Columns: `prompt_id, model, prompt_tokens, baseline_cost, optimized_cost, delta_cost, delta_pct, baseline_cache_hit, optimized_cache_hit`. Diagnoses which prompt shapes the reorder helps vs hurts.
- **Bootstrap CI on A/B cost delta** (new module `contextops_bench/stats.py`): new keys in `.summary.json` — `cost_delta_ci_low_usd`, `cost_delta_ci_high_usd`, `effect_size_pct` — rendered as `[low, high] @ 95%, effect: X.X%`. `n_boot=10_000` default (auto-scales to 1k when N<20). Deterministic via seed. Stdlib only — no scipy. Effect-size center is **median** of paired Δ cost, robust to skewed cost distributions.
- **`bench replay <csv>` subcommand** (new module `contextops_bench/replay.py`): re-runs the LLM call only, reusing the prompt structures saved in the source CSV. Preserves pair ordering so the cost-delta CI is directly comparable across replays. Useful for cross-provider comparison (Anthropic vs OpenAI vs Gemini on the same prompt set). Writes `<label>.replay.csv` and `<label>.replay.summary.json`. **No cost cap** — user responsibility, called out in README and CLI help text.

### Changed
- `contextops_bench.runner.summarize()` output schema extended additively; existing keys unchanged.

### Tests
- ~10 new tests across `tests/test_bench_stats.py`, `tests/test_bench_breakdown.py`, `tests/test_bench_replay.py`. Target: ≥63 passing (was 53).

See `docs/PLAN_v0.3.3.md` for the full plan, decisions, and acceptance criteria.

## [0.3.2] — 2026-07-07

This release ships the v0.3.1 cache-key regression fix plus the bench infrastructure needed to actually measure it (CI regression gate, direct OpenAI/Google providers), and adds a safety-net auto-default that closes the latent version of the same bug on the no-preset cloud path. See [`docs/POSTMORTEM_realistic_cache.md`](docs/POSTMORTEM_realistic_cache.md) for the full story.

### Fixed
- **Bench harness — realistic preset cache key regression:** the `realistic` agent preset pinned `system` and `tools` to constants but left `role` randomized (`random.choice(["weather-agent", "code-assistant", ...])`). Since the bench sends the cacheable prefix as `system + "\n\n" + tools + "\n\n" + role`, role rotation silently invalidated the cache key on every call — every optimized call became a cold `cache_creation` (1.25× write surcharge) with zero `cache_read`s, making the optimized arm more expensive per call than baseline. Pinned `role: "code-assistant"` in `AGENT_PRESETS["realistic"]`. After the fix (verified on OpenCode-ZEN, `--preset-agent realistic`, n=30): optimized arm is **90% cheaper per call** ($0.00107 vs baseline $0.01062) with mean cache hit rate 89.2% (the cache mechanism works correctly on ZEN once the prefix is stable across calls). Total run cost dropped from $0.319 → $0.032 across the 60-call A/B — saved $0.287.

- **Bench harness — no-preset cloud path silently randomized role too.** The fix above only covered the explicit `--preset-agent realistic` path; anyone running `bench cloud --provider direct_openai` *without* `--preset-agent` and without `--fixed-*` overrides still got the same bug because `generate_one` randomizes `role` by default. Added a safety net in `__main__._resolve_preset_args`: on cache-bearing providers (`openrouter`, `direct_anthropic`, `direct_zen`, `direct_openai`, `direct_google`), if no preset/fixed args are passed, the `realistic` preset is auto-applied and a loud warning explains what happened and how to opt out. Echo / Ollama / LM Studio unchanged (they have no cache, so the default is meaningless there). New `--preset-agent none` flag for the explicit opt-out.

### Changed
- `contextops_bench.prompt_factory.generate_one` / `generate_many` now accept a `fixed_role` parameter to mirror `fixed_system` / `fixed_tools` / `fixed_model`. Presets can lock agent identity the same way they lock system prompt and tool schema.
- Bench startup log now reports the resolved `role` along with `system` and `tools` sizes, so future regressions in preset-pinning are obvious at a glance.

### Fixed (tests)
- Four unit tests in `tests/test_bench_unit.py` were authored for the original single-run-per-prompt `run_batch` behavior. After the cache-control refactor, `run_batch` runs each prompt twice (optimized + baseline) for paired A/B. Updated expected counts from N → 2N to match. No production code change required; the tests had drifted from actual semantics.

### Added
- **CI bench regression gate** (`.github/workflows/bench-regression.yml` + `scripts/ci_bench_gate.py`): runs the realistic-preset bench against a real provider with a small N (default 5) and fails the workflow if `optimized.cache_hit_rate_p50 < BENCH_THRESHOLD` (default 0.50). This is the meta-fix for the cache-key regression above — unit tests use EchoClient (no real cache, no real network) and would never have caught it. Triggered on PRs to `main`, push to `main` (paths-filtered to bench source), and `workflow_dispatch` for manual runs with custom `n`/`threshold`/`provider`/`model` inputs. Skipped with a warning if no API key secret is configured; add one of `ZEN_API_KEY` (recommended, cheapest), `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` as a repo secret to enable.

### Fixed
- `contextops_bench/__main__.py`: `cloud` and `local` subcommands now honor `--label` when set (previously hard-coded the label, making the CLI flag a silent no-op). If `--label` is set with a single model, it's used verbatim; with multiple models, the model name is appended to keep artifacts unique.

### Added
- **Direct OpenAI provider** (`contextops_bench.clients.OpenAIDirectClient`, alias `direct_openai` / `openai`): bypasses OpenRouter entirely. OpenAI's prompt caching is AUTOMATIC — no `cache_control` markers, just `usage.prompt_tokens_details.cached_tokens` reporting which prompt tokens came from cache at 50% off input. This is the opposite cache shape from Anthropic (which we already support via `direct_anthropic` and `direct_zen`) — useful for users who need to verify both flavors of cache mechanics in one tool. Auth: `OPENAI_API_KEY` env var.
- **Direct Google Gemini provider** (`contextops_bench.clients.GoogleDirectClient`, alias `direct_google` / `google`): bypasses OpenRouter entirely, talks to `generativelanguage.googleapis.com` (Google AI Studio) directly. Gemini's caching is also IMPLICIT (`cachedContentTokenCount` in `usageMetadata`) — no markers, no separate `system` message, just a `systemInstruction` top-level field that maps from the runner's `system=` kwarg. Cache reads cost 10% of input on the paid tier. Auth: `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) env var. Supports `--preset-agent realistic` end-to-end (path: `bench cloud --provider direct_google --model google/gemini-2.5-flash ...`).
- Together with the existing `direct_zen` and `direct_anthropic` providers, the bench now has dedicated measurement paths for the four major cache mechanics flavors: Anthropic-style explicit (`cache_control: ephemeral`), OpenAI-style automatic-with-discount, Gemini-style automatic-with-implicit-system-field, and Zen's pass-through (same shape as Anthropic, different URL).
- 14 new unit tests in `tests/test_bench_unit.py` (8 for the direct providers, 6 for `_resolve_preset_args`). Total: **53 passing**.

## [0.3.0] — 2026-07-04

### Changed
- **Package renamed on PyPI:** the project is now published as `contextops-tool` instead of `contextops`. The previous name was already registered on PyPI by a different project (Abhijeet Baug's "deterministic context linter", latest 0.3.2), so we couldn't keep publishing under it. **Install:** `pip install contextops-tool`. The CLI command (`contextops optimize / stats / recent / compare / eval / reset`) and internal Python module imports (`from contextops import ...`) are unchanged for discoverability. If a user has both `contextops` and `contextops-tool` installed, the `contextops` CLI will be claimed by whichever was installed last — recommend uninstalling the other `contextops` to avoid the script-name collision.

## [0.2.3] — 2026-07-04

### Changed
- CI: end-to-end PyPI publish verified — added `PYPI_TOKEN` repository secret. Same artifact as 0.2.1; this release was a re-tag to confirm the CI publishing path works. Note: PyPI later returned 403 because the `contextops` package name was already owned by a different project, which led to the rename in 0.3.0.

## [0.2.1] — 2026-07-03

### Fixed
- **Bench harness:** client request latency now correctly propagates to `BenchResult.latency_ms`. Previously both `OllamaClient` and `OpenRouterClient` computed latency but discarded it, so latency p50/p95 in bench summaries always reported `0`. `runner.run_one` already expected `resp.raw["_latency_ms"]` — the clients just needed to populate it.

## [0.2.0] — 2026-07-03

### Added
- Acceptance criteria document (`docs/ACCEPTANCE.md`) with 30+ formal pass/fail criteria.
- Bench harness (`contextops_bench/`) supporting Ollama, LM Studio, OpenRouter, and offline echo.
- Smoke suite (10 prompts, <30s) and stress suite (1000+ prompts) for CI and pre-release.

### Added
- LLM-as-judge eval (`contextops.judge`) with 4 built-in metrics: `faithfulness`, `relevance`, `completeness`, `conciseness`.
- Dataset loaders for `.json`, `.jsonl`, `.csv` (`contextops.dataset`).
- Judge clients: `EchoJudge` (offline), `CallableJudge` (any function), `LiteLLMJudge` (real LLM).
- `evaluate()` and `evaluate_ab()` entry points with structural + quality delta reporting.
- Aggregation: mean / median / stdev / pass_rate@0.7 per metric.
- New CLI command: `contextops eval` with progress bar and JSON output.
- `CallableJudge` for plugging in custom judges.
- `on_render(prompt, item) -> str` hook for full control over dataset-row injection.

### Changed
- Bumped version to 0.2.0.
- `__init__.py` now exports the full public API including `Prompt`, `OptimizationResult`, judge clients, dataset helpers.

## [0.1.0] — 2026-07-03

### Added
- Initial release.
- Cache-aware prompt reordering (`contextops.optimizer.reorder`).
- Token counting via `tiktoken` (`count_tokens`) with `cl100k_base` fallback.
- Cost and cache hit rate estimation (`optimize`).
- Local SQLite logger (`contextops.logger.Logger`) at `~/.contextops/calls.db`.
- CLI: `contextops optimize / stats / recent / compare / reset` with Rich tables.
- Optional LiteLLM auto-callback (`contextops.integrations.install_callback`).
- 9 unit tests, 3 working examples, full README.