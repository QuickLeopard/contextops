# Functional Decomposition Review — `contextops`

**Scope:** Independent review of how the project is split into modules/functions, and whether those boundaries are sound. Covers both the core library (`contextops/`) and the benchmark harness (`contextops_bench/`). Written fresh from the code; reconciled against the existing `REFACTOR_PLAN.md` only at the end (§6).

**Method:** Every claim below is grounded in a `file:line` reference I read directly. No summary was taken on faith from the README or the refactor plan.

**Codebase at time of review:** `contextops/` ~1,526 LOC across 11 modules; `contextops_bench/` ~1,570 LOC across 4 modules. Working tree has an uncommitted cache-control feature (~+570 lines across bench) layered on top of the 0.3.0 release.

---

## 1. Executive summary

The decomposition is **above average and structurally sound**. Both packages are organized into a clean, acyclic dependency DAG with sensible tiers. Modules are small and single-purpose (median ~145 LOC); there is no god module in the classic sense. The library exposes a deliberate, flat public surface, and — critically — the bench package depends *downward* on the library (only `models` + `optimizer`), never the reverse.

What drags the score down is a **handful of well-defined seams**, not a pervasive mess. In rough priority order:

1. A **private-attribute monkeypatch** (`_bench_render_order`) that crosses the library/bench boundary — the single worst smell, identifiable by `# type: ignore[attr-defined]` on *both* sides.
2. **Three pricing tables that have already drifted** (`claude-haiku-4.5` = `$0.80` in one place, `$1.00` in another) — decomposition failure by duplication.
3. A **135-line method** doing five unrelated jobs (`OpenRouterClient.complete`), and a **hand-synced dual provider list**.
4. **Correctness bugs rooted in the structure**: a fake p95 percentile, and a missing model field hidden by a hardcoded constant in the CLI.
5. **Real test gaps**: the entire 375-line CLI, `integrations.py`, and all four real HTTP clients have zero tests.

None of this is hard to fix, and the fixes are mostly local. The architecture is worth preserving and tightening, not rethinking.

### Scorecard

| Dimension | Grade | One-line justification |
|---|---|---|
| Layering / dependency direction | **A−** | Clean acyclic DAG in both packages; bench→core is one-way; one illegal private-attr edge drags it from A. |
| Cohesion / single-responsibility | **B+** | Modules are focused; `cli.py` mixes parsing+rendering+selection, and `OpenRouterClient.complete` mixes five jobs. |
| Coupling / information hiding | **C+** | Private-attr contract across package boundary; no client Protocol; capability via duck-typed class flag. |
| Naming / boundaries | **A−** | Names map cleanly to responsibilities; `EDGE_CASE_PROMPT_IDS` and `Logger.run` are dead weight. |
| Correctness | **B** | Fake p95, hardcoded baseline hit rate, pricing drift — all structural in origin. |
| Testability / coverage | **C** | Core logic is tested; CLI, integrations, and all HTTP clients are not. |
| Dependency hygiene (packaging) | **B** | `setup.cfg` duplicates `pyproject.toml`; diagnostic scripts leak into repo root. |

**Overall: B+.** Sound bones, fixable seams.

---

## 2. Architecture map

### 2.1 Core library — `contextops/`

```
Tier 0  (leaves — no internal imports)
  models.py        Pydantic models: Prompt, OptimizationResult, CallLog, HistoryMessage
  judge.py         JudgeClient Protocol + 4 metric prompt templates + scoring
  dataset.py       Golden-dataset loaders (json/jsonl/csv)
  report.py        Score aggregation + table rendering

Tier 1  (depend on Tier 0)
  optimizer.py     → models        token counting, cache-hit estimate, reorder
  logger.py        → models        SQLite append-only call log
  clients.py       → judge         EchoJudge / CallableJudge / LiteLLMJudge (imports Protocol only)

Tier 2  (orchestration)
  eval.py          → models, optimizer, judge, clients, dataset, report
  integrations.py  → models (lazy: logger)   litellm auto-logging callback

Tier 3  (interface)
  cli.py           → models, optimizer, dataset, eval, judge, logger, clients
  __init__.py      re-exports 18 symbols from 7 submodules
```

**Verified acyclic.** I traced every `from contextops…` import. The only "sideways" edge is `clients → judge`, and it's one-directional (clients needs the `JudgeClient` Protocol). No cycles.

### 2.2 Benchmark harness — `contextops_bench/`

```
Tier 0  (leaves)
  clients.py         BenchResult, CompletionResponse, 5 provider clients, get_client()
  prompt_factory.py  synthetic + edge-case + realistic-agent prompt generator

Tier 1
  runner.py          → clients, contextops.{models, optimizer}    A/B loop, stats, CSV

Tier 2  (composition root)
  __main__.py        → clients.get_client, prompt_factory.*, runner.*
```

### 2.3 The cross-package edge

```
contextops_bench/runner.py  ──imports──▶  contextops.models.Prompt
contextops_bench/runner.py  ──imports──▶  contextops.optimizer.{reorder, count_tokens}
contextops_bench/prompt_factory.py ──imports──▶ contextops.models.{Prompt, HistoryMessage}
```

The bench depends on exactly **two** core modules. `__main__.py` and `clients.py` have no `contextops` import at all. This is the correct direction and a narrow surface — good.

**But there is a second, hidden edge** (see F-1): the optimizer stamps a private `_bench_render_order` attribute onto `Prompt`, and the bench reads it back. This is a coupling that doesn't show up in the import graph and isn't part of any type signature.

### 2.4 Public API surface

`contextops/__init__.py:13-33` exports 18 names from 7 submodules. The surface is intentionally broad and flat — every submodule's "interesting" symbol is hoisted. `install_callback` (from `integrations`) is correctly kept opt-in (not re-exported). Clean and conventional.

---

## 3. What's working well

These are genuine strengths, stated precisely so the remediation plan doesn't accidentally erode them.

1. **Strict acyclic layering in both packages.** Leaf modules (`models`, `judge`, `dataset`, `report` in core; `clients`, `prompt_factory` in bench) have zero internal dependencies. The orchestrator (`eval`) sits above its dependencies, not beside them.
2. **Correct dependency direction at the package boundary.** The bench consumes the library; the library never imports the bench. The bench's import surface into the library is just `models` + `optimizer`.
3. **Small, cohesive modules.** Median ~145 LOC. Nothing is artificially split (no `utils.py` dumping ground), and nothing is bloated to the point of unreadability. `clients.py` (66 LOC) and `integrations.py` (59 LOC) are small but each does exactly one thing.
4. **Deliberate public surface.** `__init__.py` is a curated re-export layer, not a leaky `import *`. Opt-in integrations are kept out of the top level.
5. **Concern separation in the eval pipeline.** `eval.py` composes `optimizer` (structure) + `judge`/`clients` (scoring) + `dataset` (input) + `report` (output) without any of those knowing about each other. This is the cleanest part of the codebase.
6. **`Prompt.sections()` as the single rendering primitive.** Every consumer renders through one method, so reordering logic lives in exactly one place (`optimizer._reorder_sections`).

---

## 4. Findings

Each finding: **location**, **why it's a decomposition problem**, **severity**. Severities are relative to *decomposition health* (a correctness bug gets a higher severity if it exists *because* of a structural choice).

### 4.1 Coupling smells

#### F-1 — Private-attribute monkeypatch across the library/bench boundary `[Critical]`
- **Where:** set in `contextops/optimizer.py:125` (`new._bench_render_order = ...  # type: ignore[attr-defined]`); read in `contextops_bench/runner.py:30` (`getattr(p, "_bench_render_order", None)`); also set by `runner._reverse_prompt` at `runner.py:87` (same `# type: ignore`).
- **Why it's structural:** This is a hidden contract between two packages, established through an undocumented private attribute and *two* type-checker suppressions. The import graph looks clean (bench → `models`/`optimizer` only), but there's a second data-flow edge the graph hides. If `reorder()` ever stops setting the attribute, the bench silently degrades to declaration order with no error. The `# type: ignore[attr-defined]` on both ends is the tell-tale that the type system knows this is wrong.
- **Root cause:** `Prompt` has no field for "render in this order," so the optimizer reaches into a private slot to communicate ordering to the bench renderer. The bench can't derive render order itself because `Prompt.sections()` always returns declaration order.

#### F-2 — No client Protocol; capability via duck-typed class flag `[High]`
- **Where:** `contextops_bench/runner.py:125` (`getattr(client, "supports_split_messages", False)`); the flag is a plain class attribute on `BaseHTTPClient` (`clients.py:69`) and overridden on `OpenRouterClient` (`clients.py:304`). `EchoClient` (`clients.py:465`) doesn't inherit from `BaseHTTPClient` at all.
- **Why it's structural:** Five providers share an *informal* `complete(*, model, messages, temperature, max_tokens, system=None)` contract with no ABC or Protocol to enforce it. `run_one` discovers capabilities with `getattr` rather than a typed interface, and the `system` kwarg is threaded through `**complete_kwargs` conditionally (`runner.py:125-132`) because not all clients accept it. This is the Liskov-substitution failure the structure invites: `EchoClient.complete` silently lacks a parameter the others have.
- **Root cause:** No shared contract type; clients grew independently and the runner papered over the differences with `getattr`/`**kwargs`.

### 4.2 Duplication / drift

#### F-3 — Three pricing tables that have already diverged `[Critical]`
- **Where:**
  - `contextops/optimizer.py:25-35` — `_PRICING` (input-only, $/M). `claude-haiku-4.5 = 0.80`.
  - `contextops_bench/clients.py:187-193` — `AnthropicDirectClient.PRICING` (input, output tuples). `claude-haiku-4-5 = (1.00, 5.00)`.
  - `contextops_bench/clients.py:306-317` — `OpenRouterClient.PRICING` (input, output tuples). `anthropic/claude-haiku-4.5 = (1.00, 5.00)`.
- **Why it's structural:** The *same model* is priced differently across three locations, and two of them use a different *shape* (scalar vs tuple) and a different *key* (`claude-haiku-4.5` vs `claude-haiku-4-5` vs `anthropic/claude-haiku-4.5`). The drift has already happened: `$0.80` vs `$1.00` for input. There is no single source of truth, so cost estimates computed by `optimize()` and by the bench are not even comparable. This is the textbook failure mode of duplicated data.
- **Root cause:** Cost estimation was implemented three times in three places instead of being a shared service.

#### F-4 — Triplicated cost math `[High]`
- **Where:** `optimizer.py:164-169`, `clients.py:277-286` (AnthropicDirectClient), `clients.py:433-441` (OpenRouterClient). The `1_000_000` divisor, the `0.1` cache-read factor, and the `1.25` cache-write factor are inlined in each.
- **Why it's structural:** The cache-pricing formula (read at 0.1× input, write at 1.25× input) is encoded inline three times. If Anthropic changes cache pricing, three sites must be found and edited in lockstep — and the bench's two copies use a *different* formula shape than the optimizer's (the optimizer has no cache-write term at all).
- **Root cause:** Same as F-3 — no shared `estimate_cost()`.

#### F-5 — `_baseline_hit` is a no-op duplicating a constant `[Medium]`
- **Where:** `contextops/cli.py:127-128` (`def _baseline_hit(model): return 0.05`) duplicates `optimizer._BASELINE_HIT_RATE = 0.05` (`optimizer.py:40`). It ignores its `model` argument entirely.
- **Why it's structural:** The CLI's "Original cache hit rate" column (`cli.py:111`) is a hardcoded `0.05` because `OptimizationResult` has no `original_cache_hit_rate` field — the model only stores the *optimized* rate (`models.py:85`). The CLI papers over the missing field by duplicating the constant. Two copies of a magic number that can drift.
- **Root cause:** The result model is incomplete, and the CLI hides the gap.

### 4.3 God module / mixed concerns

#### F-6 — `OpenRouterClient.complete` is a 135-line god method `[High]`
- **Where:** `contextops_bench/clients.py:327-462`.
- **Why it's structural:** One method does five distinct jobs: (1) reads `OPENROUTER_CACHE_MODE` env and shapes Anthropic messages with cache-control blocks, (2) reads `OPENROUTER_PROVIDER_PIN` env and pins providers, (3) builds the payload, (4) posts and extracts usage fields, (5) computes cost and optionally debug-prints. The env reads happen mid-method. This is the single hardest function in the codebase to reason about, and it's untestable in isolation (it constructs `urllib.request.Request` directly with no seam).
- **Root cause:** A client grew by accretion as the cache-control feature was discovered.

#### F-7 — `cli.py` mixes four concerns `[Medium]`
- **Where:** `contextops/cli.py` (375 LOC, the largest core module — ~24% of the package).
- **Why it's structural:** It does argument parsing (Click decorators), prompt construction from JSON/files (`optimize`, L66-90), presentation (`_render_optimization` L94, `_render_eval_report` L327), and service selection (`_pick_real_judge` L292, `_pick_run_fn` L303). The presentation helpers in particular belong with `report.py`, which already owns table rendering. Not egregious at 375 lines, but it's the one module that imports *every* other module and carries presentation logic that has nothing to do with CLI wiring.
- **Root cause:** Convenience — rendering was written next to the command that calls it.

#### F-8 — `Prompt` model bundles content with optimization config `[Low]`
- **Where:** `contextops/models.py:33-53`. `Prompt` carries content fields (`system`, `tools`, …, `query`) *and* optimization knobs (`model` L52, `goal` L53).
- **Why it's structural:** "What the prompt is" is coupled to "how to optimize it." This forces every place that builds a `Prompt` (including the bench's `prompt_factory`) to supply a `model`/`goal` even when it only cares about content, and it means `model`/`goal` get deep-copied by `reorder()` (`optimizer.py:109`) for no reason. Mild, but it muddies the model's single responsibility.
- **Root cause:** No separate config object.

### 4.4 Correctness bugs rooted in decomposition

#### F-9 — Fake p95 percentile `[High]`
- **Where:** `contextops_bench/runner.py:372-375`: `sorted(latencies)[int(len(latencies) * 0.95)]` with `if len(latencies) >= 20 else 0.0`.
- **Why it's structural:** For `n < 20`, p95 silently reports `0.0` — a misleading "no data" that looks like "fast." For `n ≥ 20` it indexes by `int(n*0.95)`, which for `n=20` picks index 19 (the max), not the 95th percentile. This bug exists because `_stats` is a bag of inline aggregations rather than a place where a correct percentile helper would naturally live and be reused.
- **Root cause:** Inline stats with no shared/validated aggregation utility.

#### F-10 — Missing `original_cache_hit_rate` on `OptimizationResult` `[Medium]`
- **Where:** `contextops/models.py:78-88` (field absent); `cli.py:111` (hardcodes the column).
- **Why it's structural:** The result reports the optimized hit rate but not the original, so consumers can't compute the real delta. The CLI fills the gap with the duplicated `0.05` (F-5). A complete model would have made the duplication unnecessary.
- **Root cause:** Incomplete result model; consumer patches over it.

### 4.5 Dead code & clutter

#### F-11 — Dead `EDGE_CASE_PROMPT_IDS` global `[Low]`
- `contextops_bench/runner.py:262` — a module-level empty `set()` with a comment "populated by smoke benchmark" that is never written to and never read (`summarize` takes `exclude_ids` as a parameter instead). Dead.

#### F-12 — `Logger.run` context manager is a stub `[Low]`
- `contextops/logger.py:139-145` — yields a `Logger` and does nothing in `finally`; docstring says "Mostly here for future expansion." Dead weight.

#### F-13 — `_maybe` helper is buggy and unused-as-intended `[Low]`
- `contextops_bench/prompt_factory.py:141-143` — calls `generator_fn()` twice (once in the condition, once in the else-branch via `type(generator_fn())()`). Wastes work and can produce inconsistent results if the generator is stateful.

#### F-14 — Stray diagnostic scripts in repo root `[Medium]`
- `diag_anthropic.py`, `diag_cache.py`, `diag_pinned.py`, `diag_pinned_v2.py` at the repo root, untracked. They are throwaway network probes (hit OpenRouter with `urllib`, print `usage` blocks). None are imported by the package. `diag_pinned_v2.py:18` has a **broken hardcoded path** (`/Volumes/My Data/Work/Minimax Code/contextops` — the repo is under `…/Z/contextops`), so it can't even run. They clutter the top-level namespace alongside `pyproject.toml`.
- **Root cause:** Investigation scripts that never got promoted to `scripts/` or replaced by a test.

#### F-15 — Stale `__init__.py` docstring `[Low]`
- `contextops_bench/__init__.py:1` says "bench is a script package, not importable" — but `__main__.py:16-23` and the tests import it normally. The docstring is wrong.

### 4.6 Packaging / config hygiene

#### F-16 — `setup.cfg` duplicates `pyproject.toml` `[Medium]`
- `setup.cfg` (45 lines) and `pyproject.toml` both fully declare name, version, description, dependencies, classifiers, and the `contextops = contextops.cli:main` entry point. `pyproject.toml` is canonical (PEP 621, has the build-system). `setup.cfg` is a drift risk — they must be hand-synced on every release.
- Note `setup.cfg` omits the `integrations`/`bench`/`dev` extras that `pyproject.toml:31-35` defines, so they've *already* drifted slightly.

#### F-17 — Hand-synced dual provider list `[Medium]`
- `get_client` (`clients.py:526-542`) is an `if/elif` chain over provider names; `__main__.py:27-28` independently hardcodes `choices=["echo","ollama","lmstudio","openrouter","direct_anthropic"]`. Adding a provider requires editing both. They can fall out of sync (and `get_client` accepts `"anthropic"` as an alias that the CLI `choices` doesn't list).

### 4.7 Test gaps

#### F-18 — CLI, integrations, and all HTTP clients are untested `[High]`
| Module | LOC | Tested? |
|---|---|---|
| `contextops/cli.py` | 375 | **No** — no `click.testing.CliRunner` anywhere |
| `contextops/integrations.py` | 59 | **No** |
| `contextops_bench/clients.py` (4 HTTP clients) | ~370 | **No** — only `EchoClient` + `get_client` |
| `contextops_bench/__main__.py` | 195 | **No** |
| `contextops/clients.py` (`LiteLLMJudge`, `default_judge`) | — | **No** |

- The four real HTTP clients contain the most complex untested logic in the project: URL construction, Anthropic message splitting, the `per_block`/`top_level` cache-mode branching, provider pinning, and the cost-estimation formulas (F-4). The cost math is exactly where drift (F-3) hides, and it has no regression test.
- The CLI contains the `_baseline_hit` hardcode (F-5) and the presentation logic — both unverified.

---

## 5. Remediation plan

Prioritized into tiers. Each item lists **files/functions**, the **exact change**, **effort** (S ≤ 1h, M = ½–1 day, L = 1–2 days), and **dependencies**.

> ⚠️ **Pre-flight (do this first):** The working tree carries an uncommitted cache-control feature (~+570 lines across `contextops_bench/` and `tests/test_bench_unit.py`: `AnthropicDirectClient`, the `direct_anthropic` provider, `--preset-agent`/`AGENT_PRESETS`, `supports_split_messages`). **Commit (or stash) this before Tier 2/3** — those tiers touch the same files and will conflict badly with an uncommitted feature.

### Tier 0 — Hygiene (no behavioral risk) — effort: S each

| # | Change | Files / functions | Notes |
|---|---|---|---|
| T0-1 | Move the 4 `diag_*.py` to `scripts/diag/`; fix the broken path in `diag_pinned_v2.py:18`; add a one-line `scripts/diag/README.md` | repo root → `scripts/diag/` | Addresses F-14. |
| T0-2 | Delete `EDGE_CASE_PROMPT_IDS` (`runner.py:262`) | `contextops_bench/runner.py` | F-11. Grep confirms zero readers. |
| T0-3 | Delete `Logger.run` (`logger.py:139-145`) | `contextops/logger.py` | F-12. |
| T0-4 | Delete/rewrite `_maybe` (`prompt_factory.py:141-143`) — or inline its single call site | `contextops_bench/prompt_factory.py` | F-13. |
| T0-5 | Fix `__init__.py` docstring to reflect that the package *is* imported | `contextops_bench/__init__.py:1` | F-15. |
| T0-6 | Remove `setup.cfg`; keep `pyproject.toml` as the sole source | `setup.cfg` (delete) | F-16. Verify `pip install -e .` still works. |
| T0-7 | Replace the 3 `__import__("time"/"json")` dances with top-level imports | `contextops_bench/__main__.py:107,115,130` | Style only. |

### Tier 1 — Correctness (small, high-value) — effort: S–M each

| # | Change | Files / functions | Notes |
|---|---|---|---|
| T1-1 | Implement a correct percentile (use `statistics.quantiles` or a 2-line nearest-rank) and drop the `n>=20 else 0.0` gate; return `None`/`NaN` when insufficient data | `contextops_bench/runner.py:_stats` (L341-378) | F-9. Add a unit test (pairs with T4-1). |
| T1-2 | Add `original_cache_hit_rate: float` to `OptimizationResult`; compute it in `optimize()` from `estimate_cache_hit(p, reordered=False)` | `contextops/models.py:78`, `contextops/optimizer.py:129` | F-10. Then T1-3. |
| T1-3 | Delete `cli._baseline_hit`; render `result.original_cache_hit_rate` in the "Original" column | `contextops/cli.py:111,127` | F-5. Depends on T1-2. |

### Tier 2 — Architecture (the high-leverage changes) — effort: M–L

| # | Change | Files / functions | Notes |
|---|---|---|---|
| **T2-1** | **Replace `_bench_render_order` with a real `Prompt.render_order` field.** Add `render_order: list[Section] = []` to `Prompt` (`models.py`); `reorder()` and bench's `_reverse_prompt()` set it as a normal field; `_render_prompt` reads `p.render_order` instead of `getattr(p, "_bench_render_order", None)`. Delete both `# type: ignore[attr-defined]`. | `contextops/models.py`, `contextops/optimizer.py:125`, `contextops_bench/runner.py:30,87` | **F-1 — the top-priority fix.** Removes the hidden cross-package contract. The field should be empty-by-default so declaration order remains the default. |
| **T2-2** | **Consolidate pricing into `contextops/pricing.py`.** New module with a `Price` model (input, output), a single `PRICING` dict keyed by canonical model name, model-name normalization, and `estimate_cost(model, prompt_tokens, completion_tokens, cached_tokens=0, cache_write_tokens=0) -> float`. Re-export from `__init__.py`. `optimizer.optimize`, `AnthropicDirectClient`, and `OpenRouterClient` all call it; delete their local `PRICING` dicts and inline math. | new `contextops/pricing.py`; `optimizer.py:25-35,162-169`; `clients.py:187-193,277-286,306-317,433-441` | **F-3, F-4.** Fixes the `$0.80` vs `$1.00` drift by construction. Decide the canonical key format once (recommend the bare model name, e.g. `claude-haiku-4.5`). |
| **T2-3** | **Introduce a `BenchClient` Protocol** (`complete(*, model, messages, temperature, max_tokens, system=None) -> CompletionResponse`, plus `PROVIDER: str`, `supports_split_messages: bool`). Make all 5 clients conform; `EchoClient` gains the `system=None` param. Replace the `**complete_kwargs` dance in `run_one` with a direct `system=` call. | new `contextops_bench/client_protocol.py`; all clients in `clients.py`; `runner.py:125-141` | F-2. Removes the duck-typing and the LSP violation. |
| T2-4 | **Add a `CLIENTS` registry** (`dict[str, type[BenchClient]]`) in `clients.py`; `get_client` does `CLIENTS[provider](**kwargs)`; `__main__` derives `choices=list(CLIENTS)` from it. | `clients.py:526`; `__main__.py:27-28` | F-17. Single source for the provider list. |
| T2-5 | **Decompose `OpenRouterClient.complete`** into `_extract_usage(raw)`, `_shape_messages(model, messages, system, cache_mode)`, `_apply_provider_pinning(payload, model, cache_mode, pin_provider)`, `_maybe_debug(raw, cached, prompt_tokens)`. Read env vars in `__init__`, not mid-method. | `contextops_bench/clients.py:327-462` | F-6. Makes each piece unit-testable (pairs with T4-2). |

### Tier 3 — Decomposition polish — effort: M

| # | Change | Files / functions | Notes |
|---|---|---|---|
| T3-1 | Extract CLI presentation helpers (`_render_optimization`, `_render_eval_report`) into a `cli_views.py` or fold into `report.py` (which already owns table rendering). | `contextops/cli.py:94,327`; `contextops/report.py` | F-7. Leaves `cli.py` as pure wiring + command handlers. |
| T3-2 | Split `Prompt` content from optimization config: introduce `OptimizationConfig(model, goal)` and have `optimize(config, prompt)` take both, or make `model`/`goal` optional-with-default on `Prompt` and stop deep-copying them in `reorder`. | `contextops/models.py`, `contextops/optimizer.py` | F-8. Lower priority — larger blast radius across consumers (bench, eval, CLI). Consider carefully. |

### Tier 4 — Test backfill (fill the gaps the above expose) — effort: M

| # | Change | Files | Covers |
|---|---|---|---|
| T4-1 | `test_stats.py` — percentile correctness for n<20 and n≥20; CSV round-trip; `summarize` exclude semantics | new `tests/test_stats.py` | F-9, T1-1 |
| T4-2 | `test_bench_clients.py` — offline tests of `_shape_messages`, `_apply_provider_pinning`, `_extract_usage`, cost math (mock `_post`); assert the consolidated pricing matches for a shared model | new `tests/test_bench_clients.py` | F-3, F-4, F-6, T2-2, T2-5 |
| T4-3 | `test_pricing.py` — `estimate_cost()` for cached/uncached/write cases; regression test pinning `claude-haiku-4.5` to the single canonical value | new `tests/test_pricing.py` | F-3, T2-2 |
| T4-4 | `test_cli.py` — `click.testing.CliRunner` for each subcommand; assert the "Original" hit-rate column reads `original_cache_hit_rate` (not `0.05`) | new `tests/test_cli.py` | F-5, F-10, F-18 |
| T4-5 | `test_render_order.py` — `reorder()` sets `render_order`; bench `_render_prompt` honors it; the reverse baseline produces worst-case order | new `tests/test_render_order.py` | F-1, T2-1 |
| T4-6 | Network-gated `test_anthropic_integration.py` — capture what `diag_pinned_v2.py` probed, skipped unless `ANTHROPIC_API_KEY` set | new `tests/test_anthropic_integration.py` | Retires the diag scripts durably |

### Suggested sequencing

```
Pre-flight (commit feature)
  └─ Tier 0 (all parallel, no deps)
       └─ Tier 1
            ├─ T1-2 → T1-3
            └─ Tier 2
                 ├─ T2-1 (independent)
                 ├─ T2-2 (independent)
                 ├─ T2-3 → T2-4
                 └─ T2-5 (after T2-3)
                      └─ Tier 3 (T3-1 independent; T3-2 last, largest blast radius)
                           └─ Tier 4 (tests land alongside their Tier-1/2 change)
```

**Effort estimate:** Tier 0 ≈ ½ day; Tier 1 ≈ ½–1 day; Tier 2 ≈ 2–3 days (T2-2 and T2-1 are the bulk); Tier 3 ≈ 1 day; Tier 4 ≈ 1–1½ days. **Total ~5–6 focused days.**

---

## 6. Reconciliation with `REFACTOR_PLAN.md`

The repo contains an untracked `REFACTOR_PLAN.md` (658 lines) that was written from a prior architecture review. This section compares its findings with this independent review. **Conclusion: strong agreement on the diagnosis; this review adds a few findings the plan missed and frames the sequencing differently.**

| Finding (this review) | In `REFACTOR_PLAN.md`? | Notes |
|---|---|---|
| F-1 `_bench_render_order` monkeypatch | ✅ Yes (its Phase 2) | Plan proposes the same fix (real `render_order` field). Agree. |
| F-3 three pricing tables drifted | ✅ Yes (Phase 3+4) | Plan proposes `contextops/pricing.py`. Agree. This review independently confirmed the `$0.80`/`$1.00` drift. |
| F-6 `OpenRouterClient.complete` god method | ✅ Yes (Phase 7) | Plan proposes the same decomposition into ~5 helpers. Agree. |
| F-2 no client Protocol / LSP | ✅ Yes (Phase 5) | Plan proposes a `BenchClient` Protocol. Agree. |
| F-17 dual provider list | ✅ Yes (Phase 6) | Plan proposes a registry. Agree. |
| F-9 fake p95 | ✅ Yes (Phase 8.1) | Agree. |
| F-11, F-13, F-14, F-15 dead code/diag scripts | ✅ Yes (Phase 0 + Phase 1) | Agree. |
| F-16 `setup.cfg` duplication | ❌ **Not in plan** | This review adds it. |
| F-5 / F-10 `_baseline_hit` + missing `original_cache_hit_rate` | ❌ **Not in plan** | This review adds it — the plan doesn't note the incomplete result model or the CLI hardcode that papers over it. |
| F-7 `cli.py` mixed concerns | ❌ **Not in plan** | This review adds it. |
| F-8 `Prompt` content/config coupling | ❌ **Not in plan** | This review adds it (lower priority). |
| F-18 test gaps (CLI, HTTP clients, integrations) | ✅ Partial (Phase 11) | Plan proposes `test_pricing.py`, `test_bench_clients.py`, `test_cli.py`. This review agrees and adds the regression test pinning the canonical price (T4-3). |
| **Uncommitted cache-control feature** | ❌ **Not flagged** | **This review's key addition:** the plan doesn't note that ~570 lines of uncommitted bench changes overlap the code it wants to refactor. Committing first (pre-flight) is a prerequisite the plan omits. |

**Net:** the existing plan is a sound and largely complete execution checklist for the *mechanical* refactors. This review contributes (a) four findings the plan missed (F-5/F-10, F-7, F-8, F-16), (b) the pre-flight prerequisite about the uncommitted feature, and (c) an independent confirmation of the drift and the monkeypatch by reading the current tree (not the plan's description of it). The two documents are complementary; neither contradicts the other on any shared finding.

---

*Review based on the tree as of commit `516cddf` (0.3.0) plus the uncommitted working-tree changes. All `file:line` references were read directly.*
