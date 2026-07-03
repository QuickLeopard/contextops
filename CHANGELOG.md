# Changelog

All notable changes to ContextOps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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