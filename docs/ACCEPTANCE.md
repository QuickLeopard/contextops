# ContextOps — Acceptance Criteria

> Formal pass/fail criteria for the ContextOps package.
> Every release MUST pass all `MUST` criteria before publishing.
> `SHOULD` and `MAY` are tracked but don't block release.

## 1. Functional correctness

### 1.1 Reordering (`MUST`)

- **AC-R-01**: For any `Prompt` with non-empty sections in arbitrary init order, `reorder(p)` MUST produce a `Prompt` whose `sections()` are in canonical stability order: `system → tools → role → context → documents → history → query`.
- **AC-R-02**: `reorder(p)` MUST NOT change total token count of any individual section (pure reorder, no text mutation).
- **AC-R-03**: `reorder(p)` MUST preserve the user's `history` list verbatim — no re-rendering, no splitting, no merging of messages.
- **AC-R-04**: When `goal="quality"`, `reorder(p)` MUST preserve the user's input order and return a prompt whose `sections()` follow declaration order: `system → tools → role → context → documents → history → query`.
- **AC-R-05**: For an empty `Prompt()`, `reorder(p)` MUST return an empty `Prompt()` with zero sections.

### 1.2 Token counting (`MUST`)

- **AC-T-01**: `count_tokens(text, model)` MUST return the same number as `tiktoken.encoding_for_model(model).encode(text)` for all models tiktoken knows.
- **AC-T-02**: For unknown models, `count_tokens` MUST fall back to `cl100k_base` without raising.
- **AC-T-03**: `count_tokens("", model)` MUST return 0.

### 1.3 Optimization (`MUST`)

- **AC-O-01**: `optimize(p).optimized_tokens == optimize(p).original_tokens` when only reordering is needed (no text mutation in v0.x).
- **AC-O-02**: `optimize(p).estimated_cache_hit_rate` MUST be in `[0, 1]`.
- **AC-O-03**: For a prompt with ≥ 3 sections, `estimate_cache_hit(reordered=True) > estimate_cache_hit(reordered=False)`.
- **AC-O-04**: `optimize(p)` MUST always include a `notes` field (list of strings, possibly empty).

### 1.4 Backwards compatibility (`MUST`)

- **AC-B-01**: All v0.1 public symbols (`optimize`, `reorder`, `count_tokens`, `estimate_cache_hit`, `Logger`, `CallLog`, `compare`) MUST remain importable from `contextops` in v0.2+.
- **AC-B-02**: The output schema of `compare(baseline, optimized)` from v0.1 MUST NOT change in v0.2.

### 1.5 Judge and eval (`MUST`)

- **AC-J-01**: For known metrics (`faithfulness`, `relevance`, `completeness`, `conciseness`), `score_one` MUST return a dict with `score ∈ [0, 1]`, `metric`, `reason`, `raw`.
- **AC-J-02**: For unknown metric names, `score_one` MUST raise `ValueError` (not silently default).
- **AC-J-03**: If the judge returns text that contains valid JSON, `score_one` MUST extract it and use the `score` field.
- **AC-J-04**: If the judge returns text with no parseable JSON, `score_one` MUST return the metric's default score (0.5) and set `reason` to a non-empty diagnostic string.
- **AC-J-05**: `evaluate(prompt, run_fn, dataset, metrics)` MUST invoke `run_fn` exactly `len(dataset)` times.
- **AC-J-06**: `evaluate_ab(b, o, run_fn, dataset)` MUST invoke `run_fn` exactly `2 * len(dataset)` times.
- **AC-J-07**: `evaluate_ab` result MUST contain keys: `baseline`, `optimized`, `structural`, `quality`.

### 1.6 Dataset loading (`MUST`)

- **AC-D-01**: `load(path)` MUST support `.json`, `.jsonl`, `.csv` extensions.
- **AC-D-02**: For `.json`, the file MUST be either a list of objects or `{"items": [...]}`.
- **AC-D-03**: Field aliases MUST work: `query|question|q` for query, `expected|answer|a` for expected, `context|ctx` for context.
- **AC-D-04**: Missing optional fields MUST default to empty string.
- **AC-D-05**: Unsupported extension MUST raise `ValueError` with extension name in the message.

### 1.7 Logger (`MUST`)

- **AC-L-01**: `Logger.log(entry)` MUST persist to a SQLite database at the configured path.
- **AC-L-02**: `Logger.stats()` MUST return a dict with `total_calls`, `total_prompt_tokens`, `cache_hit_rate`, `total_cost_usd`.
- **AC-L-03**: `Logger.recent(n)` MUST return at most `n` rows, most-recent first.
- **AC-L-04**: Database MUST be created automatically if it does not exist, including parent directories.

## 2. Performance (`SHOULD`)

- **AC-P-01**: `reorder(p)` for a prompt with ≤ 10 sections MUST complete in < 1 ms on a typical laptop.
- **AC-P-02**: `count_tokens(text, model)` for 100k-token text MUST complete in < 100 ms.
- **AC-P-03**: `Logger.stats()` for a 10k-row database MUST complete in < 200 ms.
- **AC-P-04**: `evaluate_ab` for a 100-row dataset with 3 metrics MUST complete end-to-end (excluding model latency) in < 5 s when using EchoJudge.

## 3. Robustness (`MUST`)

- **AC-X-01**: All public functions MUST accept and gracefully handle empty strings, empty lists, and empty dicts.
- **AC-X-02**: `Prompt(history=["raw string", {"role": "user", "content": "hi"}])` MUST coerce both forms to `HistoryMessage` without raising.
- **AC-X-03**: `Prompt(query=None)` (if pydantic allows) MUST coerce to empty string.
- **AC-X-04**: Concurrent writes to the same SQLite logger MUST NOT corrupt the database (single-process safe).
- **AC-X-05**: When the judge client raises an exception, the exception MUST propagate (no silent swallowing in the core pipeline).

## 4. Documentation (`SHOULD`)

- **AC-DOC-01**: README MUST show a working `pip install` + 5-line usage example.
- **AC-DOC-02**: Every CLI command MUST have a `--help` text describing all flags.
- **AC-DOC-03**: Public API MUST be exported via `contextops/__init__.py`.

## 5. Large-scale acceptance (`MUST` for v0.3+)

> Run from `bench/` directory.

- **AC-LS-01**: Smoke suite (10 prompts, no LLM) MUST pass in < 30 seconds.
- **AC-LS-02**: Local suite (100 prompts × 2 models on Ollama/LM Studio) MUST complete without unhandled exceptions.
- **AC-LS-03**: Cloud suite (≥ 1000 prompts via OpenRouter across ≥ 3 models) MUST produce a CSV report with no NaN/None in the critical columns: `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`, `cache_hit_rate`.
- **AC-LS-04**: Reorder improvement (median cache hit rate delta across 1000+ prompts) MUST be ≥ +20 percentage points.
- **AC-LS-05**: At least 3 different model providers MUST be exercised in the cloud suite.
- **AC-LS-06**: Concurrent suite (5 parallel run_fn tasks) MUST produce the same totals as a serial run (modulo LLM-side nondeterminism).

## 6. Pass criteria summary

| Stage | Required to pass | Required to publish |
|---|---|---|
| Unit tests (24 tests) | All 24 green | Yes |
| Bench smoke (10 prompts) | All green | Yes |
| Bench local (100 prompts, 2 models) | No exceptions | Yes |
| Bench cloud (1000 prompts, 3+ models) | No NaN, +20pp cache delta | Pre-release only |
| Bench concurrent | Equal totals | Pre-release only |

## 7. Release gating

A version may be published to PyPI only when:

1. All `MUST` criteria pass.
2. All unit tests pass.
3. Smoke bench passes.
4. Local bench runs without exceptions.
5. CHANGELOG.md is updated.