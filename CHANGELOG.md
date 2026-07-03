# Changelog

All notable changes to ContextOps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Acceptance criteria document (`docs/ACCEPTANCE.md`) with 30+ formal pass/fail criteria.
- Bench harness (`contextops_bench/`) supporting Ollama, LM Studio, OpenRouter, and offline echo.
- Smoke suite (10 prompts, <30s) and stress suite (1000+ prompts) for CI and pre-release.

## [0.2.0] — 2026-07-03

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