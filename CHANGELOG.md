# Changelog

All notable changes to ContextOps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Bench harness — realistic preset cache key regression:** the `realistic` agent preset pinned `system` and `tools` to constants but left `role` randomized (`random.choice(["weather-agent", "code-assistant", ...])`). Since the bench sends the cacheable prefix as `system + "\n\n" + tools + "\n\n" + role`, role rotation silently invalidated the cache key on every call — every optimized call became a cold `cache_creation` (1.25× write surcharge) with zero `cache_read`s, making the optimized arm more expensive per call than baseline. Pinned `role: "code-assistant"` in `AGENT_PRESETS["realistic"]`. After the fix (verified on OpenCode-ZEN, `--preset-agent realistic`, n=30): optimized arm is **90% cheaper per call** ($0.00107 vs baseline $0.01062) with mean cache hit rate 89.2% (the cache mechanism works correctly on ZEN once the prefix is stable across calls). Total run cost dropped from $0.319 → $0.032 across the 60-call A/B — saved $0.287.

### Changed
- `contextops_bench.prompt_factory.generate_one` / `generate_many` now accept a `fixed_role` parameter to mirror `fixed_system` / `fixed_tools` / `fixed_model`. Presets can lock agent identity the same way they lock system prompt and tool schema.
- Bench startup log now reports the resolved `role` along with `system` and `tools` sizes, so future regressions in preset-pinning are obvious at a glance.

### Fixed (tests)
- Four unit tests in `tests/test_bench_unit.py` were authored for the original single-run-per-prompt `run_batch` behavior. After the cache-control refactor, `run_batch` runs each prompt twice (optimized + baseline) for paired A/B. Updated expected counts from N → 2N to match. No production code change required; the tests had drifted from actual semantics.

### Added
- **CI bench regression gate** (`.github/workflows/bench-regression.yml` + `scripts/ci_bench_gate.py`): runs the realistic-preset bench against a real provider with a small N (default 5) and fails the workflow if `optimized.cache_hit_rate_p50 < BENCH_THRESHOLD` (default 0.50). This is the meta-fix for the cache-key regression above — unit tests use EchoClient (no real cache, no real network) and would never have caught it. Triggered on PRs to `main`, push to `main` (paths-filtered to bench source), and `workflow_dispatch` for manual runs with custom `n`/`threshold`/`provider`/`model` inputs. Skipped with a warning if no API key secret is configured; add one of `ZEN_API_KEY` (recommended, cheapest), `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` as a repo secret to enable.

### Fixed
- `bench/__main__.py`: `cloud` and `local` subcommands now honor `--label` when set (previously hard-coded the label, making the CLI flag a silent no-op). If `--label` is set with a single model, it's used verbatim; with multiple models, the model name is appended to keep artifacts unique.

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