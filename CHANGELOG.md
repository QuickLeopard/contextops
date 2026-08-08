# Changelog

All notable changes to ContextOps are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **`contextops.optimizer.estimate_cache_hit` is now position-aware**: bonuses only accrue for the longest stable-ordered prefix. Previously the estimator summed bonuses for every stable section regardless of position, overstating hit rate for prompts with broken render orders. Uses `_bench_render_order` when present.
- **`contextops.report.a_b_compare` now reports both baseline and optimized counts** (`n_baseline` / `n_optimized`) while keeping `n` as the baseline count for backwards compatibility.
- **Removed redundant token-counting work in `optimize()`**: section content is memoized across original and reordered passes.
- **Collapsed dead branch in `contextops.eval._render_prompt`**: `history` is rendered by `Prompt.sections()`, so it no longer needs a special-cased append path.
- **CLI progress bar totals are now correct and monotonic**: `evaluate_ab` reports a single increasing progress count across both arms and both phases, fixing the previous reset/total-mismatch bug.
- **Deduplicated the baseline cache-hit constant** in the CLI; it now reads from the centralized optimizer config instead of redefining a local value.

### Added
- **Retry/timeout logic in `LiteLLMJudge.complete`** with exponential backoff, handling transient provider failures gracefully.
- **Concurrent judge scoring in `score_many`** via `ThreadPoolExecutor`; output order is preserved by submission order and progress callbacks are thread-safe and monotonic. Exposed through the `contextops eval --parallel N` CLI flag.
- **`lru_cache` on `_get_encoding`** to avoid repeatedly loading tiktoken BPE rank files.
- **Dataset validation warnings** in `contextops.dataset.load` for empty queries, empty expected answers, duplicate queries, and entirely empty datasets.
- **SQLite WAL mode + connection timeout** in `contextops.logger.Logger` to eliminate "database is locked" errors under concurrent writes.
- **New module `contextops.pricing`** centralizing per-model token prices, default price, and cache-read discount.
- **`contextops.optimizer.OptimizerConfig`** injectable configuration object for stability ordering, hit-rate bounds, section bonuses, cache-read discount, and pricing. `optimize()`, `reorder()`, and `estimate_cache_hit()` all accept an optional `config` argument while preserving existing defaults.
- **Public benchmark dashboard** (`scripts/generate_dashboard.py`) generates a self-contained `docs/dashboard/index.html` from `bench/results/*.summary.json`, with summary cards, comparison charts, and per-run savings.
- **Dashboard auto-regeneration workflow** (`.github/workflows/dashboard.yml`) regenerates and commits the dashboard whenever bench results or the generator change.
- **One-line OpenAI SDK integration** (`contextops.integrations.openai.patch`) wraps an `openai.OpenAI` client so every `chat.completions.create` call is reordered for cache friendliness and optionally logged to the local SQLite database.
- **`vllm` and `tgi` bench providers** (`contextops_bench/clients.py`): `VLLMClient` (OpenAI-compatible `/v1/chat/completions`) and `TGIClient` (native `/generate`, with `tiktoken`-based token estimation since TGI reports no usage). Both are local/self-hosted with `cost_usd=0.0` — used to measure token/latency savings, not cache/cost savings. Wired into `get_client()` and the `--provider` CLI choice.
- **`safety` and `format_compliance` judge metrics** (`contextops/judge.py`): `safety` fails closed (defaults to `0.0`, not `0.5`, on unparseable judge output); `format_compliance` reuses the existing `expected` field to carry the required-format spec (e.g. `"valid JSON with keys: name, age"`) instead of an expected answer.
- **RAG Curator** (`contextops/curator.py`): `DocumentChunk`/`CuratorConfig`/`CurationResult`/`curate()` — multi-signal (similarity + recency half-life decay + trust) scoring, bag-of-words Jaccard dedup, strict threshold, optional `max_chunks` cap, human-readable drop reasons. `Prompt.from_chunks()` builds a `Prompt` directly from curated chunks with zero changes to `optimize()`/`reorder()`. New `contextops curate` CLI command. Deterministic (no LLM call) `context_precision()` n-gram-overlap heuristic pluggable into `evaluate()`/`evaluate_ab()` via the new optional `curated_chunks`/`curated_chunks_baseline`/`curated_chunks_optimized` params — surfaces automatically as a `context_precision` row in A/B quality deltas since `a_b_compare` is metric-agnostic.
- **RAG Curator real-LLM bench** (`contextops_bench/curator_bench.py`): synthetic, deterministic noisy-retrieval QA dataset generator (`generate_curation_dataset`, mixes relevant facts with injected noise chunks + near-duplicates at configurable ratios) and `run_curator_bench()`, which runs raw (uncurated) vs `curate()`-filtered chunks through the same LLM client for the same query, measuring token/cost deltas AND answer-quality impact (judge metrics via `contextops.report.a_b_compare` + deterministic `context_precision`). New `python -m contextops_bench curator` subcommand (independent `--provider`/`--echo-judge` flags — bench a real LLM under test with a free offline judge, or vice versa) writes `bench/results/<label>.curator_summary.json`. `scripts/generate_dashboard.py` gained a "RAG Curator Bench" section (own HTML fragment, own table) that renders these results independently of the existing cache-hit-rate charts; the dashboard no longer early-returns when only curator results exist.
- **Real judge + statistical significance gate for the curator bench** (`contextops_bench/curator_bench.py`): `_ClientAsJudge` adapts any bench client (`ZenDirectClient`, `OllamaClient`, etc.) to the `JudgeClient` protocol, so the curator bench can self-judge using the SAME real API key/client already configured for the LLM-under-test — replacing the previous default of a fixed-score `EchoJudge` with zero new credentials required. Handles Anthropic-native clients (`supports_split_messages=True`) correctly by moving the judge prompt's system-role message into the `system=` kwarg instead of leaving it in `messages` (which Anthropic's native API rejects with HTTP 400). New `evaluate_curator_quality_gate()` computes a bootstrap-CI (reusing `contextops_bench.stats.bootstrap_ci`/`effect_size_pct`) significance check per judge metric, paired by `(metric, index)`; folded into `run_curator_bench()`'s summary as `quality_gate` and per-metric `ci_low`/`ci_high`/`significant`/`effect_size_pct` fields on `summary["quality"]`. `render_curator_summary()` gained `[JUDGE]` (flags self-judging bias) and `[QUALITY GATE]` (verified/unverified + reasons) sections. New CLI flags: `--max-tokens` (default 200, up from a hardcoded 64 — short answers were starving both the judge and `context_precision`'s n-gram heuristic of signal), `--litellm-judge` (opt-in to the old separate-credential judge path), `--judge-model` (now also usable to pick a *different* model on the *same* provider as judge, partially mitigating self-judging bias without a new key). Judge-selection precedence logic extracted into standalone `_select_curator_judge()` in `contextops_bench/__main__.py` for testability. Dashboard's curator table gained a "Quality gate" verified/unverified badge column.
- **Hardened curator testing for production readiness**: property-based fuzz tests (`tests/test_curator_fuzz.py`, new `hypothesis` dev dependency) check `contextops.curator`'s pure functions' invariants — `score_chunk` always in `[0,1]`, `curate` partition/threshold/max_chunks/determinism invariants, never raises on unicode/empty/10k-word text, all-identical-chunk dedup, word-reorder-paraphrase-dedup vs. synonym-swap-not-deduped (documents the bag-of-words dedup's known semantic limitation), `context_precision` bounds. **Diversified synthetic prompt styles** (`contextops_bench/curator_bench.py`): `generate_curation_dataset(..., style=...)` now supports `qa_short` (default), `long_document` (200+ word paragraphs), `code`, `structured` (JSON blobs), `multilingual`, `multi_turn` (fabricated conversation-prefixed queries), and `adversarial_noise` (per-fact lexically-close decoys that share keywords with the query but are factually wrong — the hard case for similarity-threshold-only filtering) via new `PROMPT_STYLES`/`ALL_PROMPT_STYLES`; new `--prompt-style` CLI flag. **Pluggable real-embedding similarity** (`contextops_bench/embedders.py`, new module): `TfidfEmbedder` (stdlib-only, offline, free — replaces the dataset's synthetic `random.uniform(...)` similarity ranges with actually-computed per-item TF-IDF cosine similarity) and `OpenAIEmbedder` (`text-embedding-3-small` via the same `urllib`-only HTTP pattern as the existing direct provider clients, opt-in, requires `OPENAI_API_KEY`); `get_embedder()` factory; new `--embedder {none,tfidf,openai}` CLI flag, wired into `generate_curation_dataset(..., embedder=...)` via `_assign_embedded_similarity()`. Note: real embedding cosine similarity is on a much lower numeric scale (~0.03-0.15 for TF-IDF) than the default synthetic range (0.10-0.98) — using `--embedder` without lowering `--threshold` will drop most/all chunks; documented in `README.md`. **Production data harness**: `load_curation_dataset()` parses a user-supplied JSON file using the same chunk schema as `contextops curate --chunks`, with mutual exclusivity checks against synthetic-generation flags (`--dataset` vs. `--n`/`--noise-ratio`/`--prompt-style`) in the CLI. New tests cover dataset loading, validation, and CLI mutual-exclusivity enforcement.
- **Bench/dashboard maturity (Track C)**: confidence-based sorting for the "All runs" table (`scripts/generate_dashboard.py`) surfaces verified/high-confidence runs first. New CI data-quality gate (`scripts/ci_bench_data_gate.py` + `.github/workflows/bench-regression.yml`) blocks unverified `bench/results/*.summary.json` files from merging. `contextops_bench/quality.py` now treats local/self-hosted providers (`ollama`, `vllm`, `tgi`, `lmstudio`) as verified when token-count or latency deltas are statistically significant, since their cost is identically zero; `contextops_bench/runner.py` computes paired bootstrap CIs for both. Padded the `realistic` agent preset (`contextops_bench/prompt_factory.py`) to exceed Anthropic Haiku's 2048-token cache-minimum, restoring cache hit rate on that path.

### Changed
- **`Section` literal ordering in `contextops.models`** now matches the canonical stability order used by the optimizer (`documents` before `history`).
- `contextops.cli.optimize` table now reads the baseline hit-rate from `DEFAULT_CONFIG` rather than a private module constant.

### Tests
- Added regression tests for position-aware hit-rate estimation, A/B compare counts, concurrent `score_many` order/progress monotonicity, dataset validation warnings, SQLite WAL concurrency, `LiteLLMJudge` retry behavior, injectable `OptimizerConfig`, and aligned `Section` literal ordering.
- Added tests for the dashboard generator (filename parsing, dataset building, summary stats, end-to-end HTML generation).
- Added tests for the OpenAI SDK integration (message conversion, reordering, logging, idempotent patch/unpatch).
- Added tests for `VLLMClient`/`TGIClient` (factory dispatch, response parsing, message flattening, token estimation) and the `safety`/`format_compliance` judge metrics (fail-closed default, `expected`-as-format-spec convention).
- Added tests for the RAG Curator (`tests/test_curator.py`): scoring math, recency half-life decay, threshold edge cases (exactly-at, all-above, all-below), dedup (drops lower-scored duplicate, keeps distinct chunks), `max_chunks` cap, `Prompt.from_chunks()`, CLI end-to-end via `CliRunner`, and `context_precision()` (used/unused/mixed/empty cases); plus `evaluate()`/`evaluate_ab()` `curated_chunks` wiring tests in `tests/test_v02_eval.py`.
- Added tests for the RAG Curator bench (`tests/test_curator_bench.py`): deterministic dataset generation (item count, noise ratio, dedup injection, seed reproducibility), `run_curator_bench()` summary shape, curated-uses-fewer-tokens and curated-context-precision-higher-than-raw assertions (using a local content-aware stub client — the real `EchoClient` always returns fixed text, which would make `context_precision` meaningless), and `render_curator_summary()`. Extended `tests/test_dashboard_generator.py` with `_load_curator_runs()`/`_render_curator_section()` unit tests plus end-to-end fixtures covering curator-only and mixed (cache + curator) result sets.
- Added tests for `_ClientAsJudge` (returns `.text`, satisfies `JudgeClient` duck-typing for `score_many()`, judge-side `max_tokens` independent of the under-test model's), `evaluate_curator_quality_gate` (significant-improvement, identical-scores-not-significant, low-n, and multi-metric-paired-by-index cases), and `max_tokens` plumbing to the LLM-under-test's `client.complete()` calls, in `tests/test_curator_bench.py`. New `tests/test_bench_curator_cli.py` unit-tests `_select_curator_judge()`'s precedence logic (`--echo-judge` > `--litellm-judge` > provider=="echo" fallback > default self-judging, plus the `--judge-model`-differs-from-model-under-test labeling) without a real network call. Extended `tests/test_dashboard_generator.py`'s curator fixture with a `quality_gate` field and added tests for the verified/unverified/n-a badge rendering.
- New `tests/test_curator_fuzz.py`: `hypothesis`-driven property tests for `score_chunk`/`curate`/`context_precision` invariants (unit-interval bounds, determinism, partition/threshold/max_chunks correctness, never-raises on unicode/empty/huge text) plus explicit dedup edge cases (all-identical collapse, word-reorder-still-dedups, synonym-swap-not-deduped). New `tests/test_embedders.py`: `cosine_similarity` bounds/clamping, `TfidfEmbedder` determinism and relevant-vs-noise separation, `get_embedder()` dispatch, `OpenAIEmbedder` via a mocked HTTP call (no real network/key). Extended `tests/test_curator_bench.py` with prompt-style generation/determinism/transform tests (all 7 styles), embedder-wiring tests, and `load_curation_dataset()` round-trip/validation tests. Extended `tests/test_bench_curator_cli.py` with `_check_dataset_mutual_exclusivity()` and `_load_or_generate_curator_dataset()` tests.
- Extended `tests/test_bench_quality.py` with local-provider verification tests: token-delta significance, latency-delta significance, failure when neither is significant, and `is_local_provider()` dispatch.
- **Access-aware context + audit trail (Track B v1.0 MVP)**: `contextops/access.py` adds `Principal`, `TaggedContent`, `AccessDecision`, `RedactionResult`, `RoleBasedPolicyEngine`, and `apply_access_policy()` — pure, role-based redaction before optimization. `contextops/logger.py` extends the SQLite schema with an `access_audit` table that stores only content hashes, never raw text. `contextops optimize` gains `--principal-id`, `--principal-role`, and `--access-tags` flags that apply redaction before reordering and persist decisions to the audit log. New `contextops audit` CLI command renders recent decisions from the local audit log with optional `--principal` filter. Tests in `tests/test_access.py`, `tests/test_logger.py`, and `tests/test_cli.py`.
- Full suite: **281 passing**.

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