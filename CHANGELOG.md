# Changelog

All notable changes to ContextOps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`Prompt.render_order` field** (public API): when set, `Prompt.sections()` yields in that order instead of declaration order. Replaces the private `_bench_render_order` monkeypatch that previously crossed the library/bench package boundary.
- **Shared `contextops.pricing` module** — a single source of truth for model pricing (`Price`, `PRICING`, `estimate_cost()`). Reconciles three previously-divergent pricing tables. Exported from the top-level package.
- **`BenchClient` Protocol** (`contextops_bench.client_protocol`): the implicit client contract is now explicit and runtime-checkable. `EchoClient` conforms (LSP fixed — it now accepts the `system` kwarg).
- **`CLIENTS` provider registry** (`contextops_bench.clients`): single source for the provider list; the CLI `--provider` choices derive from it.
- **`contextops_bench/types.py`**: `BenchResult` and `CompletionResponse` moved out of `clients.py` to break an import cycle with the Protocol.
- Tests: `_percentile`, `save_csv` empty-input, `run_one` system-passing, EchoClient LSP regression, OpenRouter `_shape_messages`/`_apply_provider_pinning`/`_maybe_debug`, registry, Protocol conformance, and full CLI coverage (`test_cli.py`). Test count: 39 → 73.

### Changed
- **`OpenRouterClient.complete` decomposed** from a 135-line god method into `_shape_messages`, `_apply_provider_pinning`, `_maybe_debug`. Environment-variable reads moved from per-call to the constructor.
- **Cost estimation centralized**: `AnthropicDirectClient` and `OpenRouterClient` now call the shared `estimate_cost()` instead of maintaining their own `PRICING` dicts and inline `1_000_000` cost math.
- **`runner.run_one` simplified**: the `**complete_kwargs` conditional is gone (every client now accepts `system=None`).
- **`_render_prompt` simplified**: now 5 lines — `Prompt.sections()` respects `render_order` directly, so the `getattr` ladder is removed.
- **`_make_fresh_client` → `_reset_client_state`**: honestly named (it mutates in place, doesn't return a fresh instance).
- **`prompt_factory`**: global `random.seed()` calls replaced with local `random.Random` instances (no longer perturbs the global RNG). Realistic-agent preset content moved to `contextops_bench/data/` data files (module shrank from ~422 to ~250 lines).
- **`__main__._execute` decomposed** into `_resolve_preset`, `_build_prompt_list`, `_write_artifacts`. Stale docstring updated (`bench.smoke` → `contextops_bench smoke`).

### Fixed
- **Pricing drift**: `claude-haiku-4.5` was `$0.80/M` in `optimizer.py` but `$1.00/M` in the bench clients. Now a single reconciled value (`$1.00/M`) via the shared module.
- **Fake p95 percentile**: `runner._stats` used `sorted(latencies)[int(len*0.95)]`, which silently returned `0.0` for `n<20` and indexed incorrectly. Replaced with a correct nearest-rank `_percentile` helper.
- **`smoke`/`run_all` args mutation**: `smoke()` mutated the shared `args` Namespace, corrupting the subsequent `_execute` call in `run_all`. Now operates on a shallow copy.
- **`save_csv` silent no-op**: empty input previously wrote a headerless file indistinguishable from "no rows"; now raises `ValueError`.
- **Narrowed bare `except`** in `OllamaClient.list_models` to `(URLError, ValueError, KeyError)`.
- **Stale `__version__`**: `__init__.py` reported `0.2.0` while packaging said `0.3.0` (already corrected in-tree pre-refactor).
- Removed dead code: `_maybe` helper (buggy + unused), `EDGE_CASE_PROMPT_IDS` global (never populated), `__import__("time"/"json")` dances, redundant `or 0` in cache-token parsing.

### Removed
- `_bench_render_order` private attribute and both associated `# type: ignore[attr-defined]` suppressions.
- Duplicate `PRICING` dicts from `AnthropicDirectClient` and `OpenRouterClient`.

### Internal
- Diagnostic cache-control probes (`diag_*.py`) moved from repo root to `scripts/diag/` with a README. Fixed the broken hardcoded path in `diag_pinned_v2.py`.

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