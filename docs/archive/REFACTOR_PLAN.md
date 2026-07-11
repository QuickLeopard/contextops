# ContextOps — Refactor Plan

> **Status: EXECUTED (2026-07-11).** All 13 phases landed in commit
> `1f836f5` on branch `refactor/architecture-cleanup`. Tests went 39 → 73.
> Retained here as a historical record. See `docs/CODE_REVIEW.md` for the
> independent review that motivated it (and 4 additional findings not
> covered by this plan, addressed separately).

A detailed, step-by-step fix plan derived from the architecture review. Every
step is **atomic** (one small change → one verification), and every phase ends
**green** (`pytest` + `python -m contextops_bench smoke`). Commit after each
phase. Phases are ordered so earlier phases unblock later ones; nothing depends
on un-built code.

---

## Verification baseline (run before starting, expect all green)

```bash
python -m pytest -v --tb=short
python -m contextops_bench smoke
```

## Decisions locked with the user

| Decision | Choice |
|---|---|
| `_bench_render_order` monkeypatch | **Add a real `Prompt.render_order` field** (Pydantic-validated, public, type-checked) |
| Three divergent pricing tables | **One shared `contextops/pricing` module** used by both the library and the bench |
| The 4 `diag_*.py` probes | **Move to `scripts/diag/` and clean** (fix broken path, generalize) |

---

## Phase 0 — Trivial cleanups (zero-risk, no logic change)

Each is a one-commit no-op-behavior cleanup. Do these first to get a clean baseline.

### 0.1 Bump stale `__version__`
- **WHAT:** `contextops/__init__.py:11` says `0.2.0` while packaging says `0.3.0`.
- **FILES:** `contextops/__init__.py`
- **CHANGE:** `__version__ = "0.2.0"` → `__version__ = "0.3.0"`
- **VERIFY:** `python -c "import contextops; print(contextops.__version__)"` → `0.3.0`
- **TEST:** `python -m pytest tests/ -q` still green.

### 0.2 Delete dead `_maybe` helper
- **WHAT:** `prompt_factory.py:141-143` — unused, and its `type(generator_fn())()` body calls the generator twice (buggy even if dead). Confirmed zero call sites via grep.
- **FILES:** `contextops_bench/prompt_factory.py`
- **VERIFY:** `grep -rn "_maybe" --include="*.py"` returns nothing.
- **TEST:** `pytest -q` green.

### 0.3 Delete dead global `EDGE_CASE_PROMPT_IDS`
- **WHAT:** `runner.py:260-262` — global declared, never written, never read. Comment references a population that no longer happens.
- **FILES:** `contextops_bench/runner.py`
- **VERIFY:** `grep -n "EDGE_CASE_PROMPT_IDS" contextops_bench/runner.py` returns nothing.
- **TEST:** `pytest -q` green; `python -m contextops_bench smoke` works.

### 0.4 Remove redundant `or 0` in AnthropicDirectClient
- **WHAT:** `clients.py:269-272` — `usage.get("cache_read_input_tokens", 0) or 0` — the `or 0` is redundant after `.get(..., 0)`.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** Collapse to `cached_tokens = usage.get("cache_read_input_tokens", 0)`.
- **VERIFY:** read the edited lines.
- **TEST:** `pytest -q` green.

### 0.5 Replace `__import__` dance with real imports
- **WHAT:** `__main__.py:107, 115, 130` use `__import__("time")` / `__import__("json")` instead of top-level imports.
- **FILES:** `contextops_bench/__main__.py`
- **CHANGE:** Add `import json` and `import time` to the top import block; replace the three `__import__(...)` calls with `time` / `json`.
- **VERIFY:** `grep -n "__import__" contextops_bench/__main__.py` returns nothing.
- **TEST:** `python -m contextops_bench smoke` works.

---

## Phase 1 — Move diag scripts to `scripts/` (user-approved)

### 1.1 Relocate the 4 diag scripts
- **WHAT:** Move `diag_anthropic.py`, `diag_cache.py`, `diag_pinned.py`, `diag_pinned_v2.py` from repo root to `scripts/diag/`. They are scratch debug probes, not part of the package.
- **FILES:** `git mv diag_*.py scripts/diag/` (create `scripts/diag/` first).
- **VERIFY:** `ls scripts/diag/` shows the four files; repo root no longer has them.
- **TEST:** `python -m pytest -q` green (they're not imported by anything).

### 1.2 Fix broken hardcoded path in `diag_pinned_v2.py`
- **WHAT:** Line 18 has `sys.path.insert(0, "/Volumes/My Data/Work/Minimax Code/contextops")` — wrong path, the repo is at `.../Z/contextops`.
- **FILES:** `scripts/diag/diag_pinned_v2.py`
- **CHANGE:** Replace the absolute path with `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` so it works from any checkout.
- **VERIFY:** `python scripts/diag/diag_pinned_v2.py` no longer fails with `ImportError` on `AGENT_PRESETS` (will fail later on missing `OPENROUTER_API_KEY` — that's expected and fine).

### 1.3 Add a short README in `scripts/diag/`
- **WHAT:** One paragraph explaining what each diag script probes and that they require live API keys. Captures the findings so future readers know why they exist.
- **FILES:** `scripts/diag/README.md`
- **VERIFY:** file exists, readable.

---

## Phase 2 — Library: add real `Prompt.render_order` field

> **Unblocks Phases 3 & 5.** This is the high-leverage architectural fix — it
> removes the cross-package `_bench_render_order` monkeypatch set from **two**
> places (`optimizer.py` and `runner.py`).
>
> **Key insight:** `Prompt.sections()` is the *only* producer of section order
> (consumed in 5 files: optimizer ×3, eval ×2, runner ×5). Making `sections()`
> itself respect `render_order` makes the monkeypatch redundant everywhere at
> once — no caller needs a `getattr` ladder anymore.

### 2.1 Add the field
- **WHAT:** Add `render_order: list[Section] | None = None` to `Prompt`.
- **FILES:** `contextops/models.py`
- **CHANGE:** After the `goal:` line (~line 53), add:
  ```python
  # When set, `sections()` yields in this order instead of declaration order.
  # `reorder()` and the bench's `_reverse_prompt()` set this so callers see the
  # new order without a private-attr monkeypatch. None = declaration order.
  render_order: list[Section] | None = None
  ```
- **VERIFY:** `python -c "from contextops.models import Prompt; p = Prompt(); print(p.render_order)"` → `None`.
- **TEST:** `pytest -q` green (additive, backwards-compatible).

### 2.2 Make `sections()` respect `render_order`
- **WHAT:** `Prompt.sections()` currently yields in hardcoded order (system, tools, role, …). Make it yield in `render_order` when set.
- **FILES:** `contextops/models.py` — the `sections()` method.
- **CHANGE:** Keep the current body building `out` in declaration order; then at the end:
  ```python
  if self.render_order is not None:
      by_name = dict(out)
      return [(name, by_name[name]) for name in self.render_order if name in by_name]
  return out
  ```
- **VERIFY:**
  ```python
  from contextops.models import Prompt
  p = Prompt(system="S", query="Q")
  print([s[0] for s in p.sections()])                  # ['system', 'query']
  p.render_order = ["query", "system"]
  print([s[0] for s in p.sections()])                  # ['query', 'system']
  ```
- **TEST:** `pytest -q` green. **Important:** add a new test
  `test_prompt_sections_respects_render_order` here so the contract is locked
  before dependents change.

### 2.3 Update `optimizer.reorder` to use the public field
- **WHAT:** `optimizer.py:125` sets `new._bench_render_order = ...`; switch to `new.render_order = ...`.
- **FILES:** `contextops/optimizer.py`
- **CHANGE:** Replace
  `new._bench_render_order = [s[0] for s in new_sections]  # type: ignore[attr-defined]`
  with `new.render_order = [s[0] for s in new_sections]`.
  Update the docstring at line ~106 that references `_bench_render_order`.
- **VERIFY:** `grep -n "_bench_render_order" contextops/optimizer.py` returns nothing.
- **TEST:** `pytest tests/test_optimizer.py -q` green.

### 2.4 Update `runner._reverse_prompt` to use the public field
- **WHAT:** `runner.py:87` sets `new._bench_render_order = ...`; switch to `new.render_order = ...`.
- **FILES:** `contextops_bench/runner.py`
- **CHANGE:** Replace the line and remove the `# type: ignore[attr-defined]`.
- **VERIFY:** `grep -n "_bench_render_order" contextops_bench/runner.py` returns nothing.

### 2.5 Simplify `runner._render_prompt`
- **WHAT:** Now that `sections()` respects `render_order`, the entire `getattr(p, "_bench_render_order", None)` ladder in `_render_prompt` (lines 30-37) is redundant.
- **FILES:** `contextops_bench/runner.py`
- **CHANGE:** Replace the whole function body with:
  ```python
  def _render_prompt(p: Prompt) -> tuple[str, list[str]]:
      sections = p.sections()
      parts = [content for _, content in sections]
      order = [name for name, _ in sections]
      return "\n\n".join(parts), order
  ```
- **VERIFY:** read the new function; it's ~5 lines.
- **TEST:** `pytest tests/test_bench_unit.py -q` green; `python -m contextops_bench smoke` works.

### 2.6 Final grep — monkeypatch gone everywhere
- **WHAT:** Confirm no `_bench_render_order` references remain anywhere.
- **VERIFY:** `grep -rn "_bench_render_order" --include="*.py" .` returns nothing (excluding `.venv`).
- **TEST:** full `pytest -q` + `python -m contextops_bench smoke` green.
- **COMMIT Phase 2 here** — this is the high-leverage architectural fix.

---

## Phase 3 — Library: shared pricing module

> **Unblocks Phase 4.** Reconciles the three divergent pricing tables into one
> source of truth.

### 3.1 Create `contextops/pricing.py`
- **WHAT:** New module holding the canonical pricing table + cost function.
- **FILES:** `contextops/pricing.py` (new)
- **CONTENT:**
  ```python
  """Canonical model pricing + cache-aware cost estimation. Single source of truth."""
  from __future__ import annotations
  from dataclasses import dataclass


  @dataclass(frozen=True)
  class Price:
      input_per_m: float
      output_per_m: float
      cache_read_factor: float = 0.10   # cached tokens billed at 10% of input
      cache_write_factor: float = 1.25  # 5-min TTL write surcharge


  # $/M tokens. Refresh quarterly. Keyed by canonical model id.
  # Prefixed aliases (e.g. "anthropic/claude-haiku-4.5") are resolved by
  # `estimate_cost` stripping the provider prefix before lookup.
  PRICING: dict[str, Price] = {
      "gpt-4o":                  Price(2.50, 10.00),
      "gpt-4o-mini":             Price(0.15, 0.60),
      "gpt-5":                   Price(5.00, 15.00),
      "claude-opus-4.6":         Price(15.00, 75.00),
      "claude-sonnet-4.6":       Price(3.00, 15.00),
      "claude-haiku-4.5":        Price(1.00, 5.00),   # was 0.80 in optimizer — reconciled
      "claude-haiku-4-5":        Price(1.00, 5.00),   # alias: Anthropic native id
      "claude-3-haiku":          Price(0.25, 1.25),
      "claude-3-haiku-20240307": Price(0.25, 1.25),
      "qwen3-30b":               Price(0.20, 0.20),
      "gigachat":                Price(0.10, 0.10),
      "yandexgpt":               Price(0.10, 0.10),
      # OpenRouter-only entries (provider-prefixed):
      "meta-llama/llama-3.1-70b-instruct": Price(0.59, 0.79),
      "meta-llama/llama-3.1-8b-instruct":  Price(0.06, 0.06),
      "qwen/qwen-2.5-72b-instruct":        Price(0.40, 0.40),
      "google/gemini-2.0-flash-exp":       Price(0.10, 0.40),
  }


  def _normalize(model: str) -> str:
      """Strip a single provider prefix (`anthropic/`, `openai/`, ...) for lookup."""
      return model.split("/", 1)[-1] if "/" in model else model


  def estimate_cost(*, prompt_tokens: int, completion_tokens: int,
                    cached_tokens: int, cache_creation_tokens: int,
                    model: str) -> float:
      """Estimate USD cost from token usage + model pricing.

      Tries the model id as-is first (so `anthropic/claude-haiku-4.5` matches a
      prefixed key), then the prefix-stripped form (so it also matches
      `claude-haiku-4.5`), then falls back to a default Price.
      """
      price = PRICING.get(model) or PRICING.get(_normalize(model)) or Price(1.0, 1.0)
      non_cached_input = max(0, prompt_tokens - cached_tokens)
      return (
          (non_cached_input / 1_000_000) * price.input_per_m
          + (cached_tokens / 1_000_000) * price.input_per_m * price.cache_read_factor
          + (cache_creation_tokens / 1_000_000) * price.input_per_m * price.cache_write_factor
          + (completion_tokens / 1_000_000) * price.output_per_m
      )
  ```
- **VERIFY:** `python -c "from contextops.pricing import estimate_cost; print(estimate_cost(prompt_tokens=2000, completion_tokens=100, cached_tokens=1000, cache_creation_tokens=0, model='claude-haiku-4.5'))"` returns a sensible float.
- **TEST:** none yet (added in 3.4).

### 3.2 Wire `optimizer.py` to use the shared table
- **WHAT:** `optimizer.py:25-35` has its own `_PRICING` dict (single-value). Replace its usage in `optimize()` (line ~163) with `pricing.PRICING`.
- **FILES:** `contextops/optimizer.py`
- **CHANGE:** Delete `_PRICING`; at usage site, `from contextops.pricing import PRICING, Price` and `price_per_m = PRICING.get(p.model, Price(1.0, 1.0)).input_per_m`. Note: `optimize()` only uses input price for its rough savings estimate, so `.input_per_m` preserves current behavior.
- **VERIFY:** `pytest tests/test_optimizer.py -q` green. **Note in the commit message** that `claude-haiku-4.5` savings estimate changes from $0.80/M to $1.00/M (intentional reconciliation).
- **TEST:** `pytest -q` green.

### 3.3 Export `pricing` from the package
- **WHAT:** Surface the new module via `contextops/__init__.py`.
- **FILES:** `contextops/__init__.py`
- **CHANGE:** Add `from contextops.pricing import Price, PRICING, estimate_cost` and the three names to `__all__`.
- **VERIFY:** `python -c "from contextops import estimate_cost; print('ok')"`.

### 3.4 Test: pricing consistency + cost math
- **WHAT:** Add `tests/test_pricing.py` covering (a) `estimate_cost` on hand-computed values, (b) that `claude-haiku-4.5` and `claude-haiku-4-5` give the same cost, (c) unknown model falls back, (d) provider-prefixed id resolves.
- **FILES:** `tests/test_pricing.py` (new)
- **VERIFY:** `pytest tests/test_pricing.py -q` green.

---

## Phase 4 — Bench: dedup the cost math

> Both `AnthropicDirectClient.complete` and `OpenRouterClient.complete` have
> inline cost formulas with `1_000_000` × 8, `0.1`, `1.25`. Replace with the
> shared `estimate_cost`.

### 4.1 `AnthropicDirectClient` uses `estimate_cost`
- **WHAT:** Replace lines 275-287 of `clients.py` with a call to `estimate_cost(...)`.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** Add `from contextops.pricing import estimate_cost` at top. Replace the cost block with:
  ```python
  cost = estimate_cost(
      prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
      cached_tokens=cached_tokens, cache_creation_tokens=cache_creation_tokens,
      model=native_model,
  )
  ```
  Delete the `PRICING` dict at lines 187-193 (now redundant — pricing module is the source of truth).
- **VERIFY:** `grep -n "1_000_000" contextops_bench/clients.py` should not appear in `AnthropicDirectClient`.
- **TEST:** `pytest -q` green.

### 4.2 `OpenRouterClient` uses `estimate_cost`
- **WHAT:** Replace lines 427-441 of `clients.py` with the same call. Pass `model` (the OpenRouter id, e.g. `anthropic/claude-haiku-4.5`) — `estimate_cost` resolves the prefix (Phase 3.1).
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** Replace the cost block. Delete the `PRICING` dict at clients.py:306-317.
- **VERIFY:** `grep -n "input_cost \* 0.1\|input_cost \* 1.25\|/ 1_000_000" contextops_bench/clients.py` returns nothing.
- **TEST:** `pytest -q` green.

### 4.3 Confirm both `PRICING` dicts gone from clients.py
- **VERIFY:** `grep -n "PRICING = {" contextops_bench/clients.py` returns nothing.
- **TEST:** full `pytest -q` + `python -m contextops_bench smoke` green.
- **COMMIT Phase 4** — DRY win, worst duplication gone.

---

## Phase 5 — Bench: `BenchClient` Protocol + LSP fix for EchoClient

### 5.0 Move shared types to `types.py` (cycle prevention)
- **WHAT:** `CompletionResponse` and `BenchResult` currently live in `clients.py`. The Protocol needs to reference `CompletionResponse`, and `clients.py` will need to reference the Protocol → cycle. Move the two dataclasses to a new `contextops_bench/types.py` and re-export from `clients.py` for backwards compat.
- **FILES:** new `contextops_bench/types.py`; modified `contextops_bench/clients.py`.
- **CHANGE:** Move `BenchResult` and `CompletionResponse` dataclasses to `types.py`. In `clients.py` add `from contextops_bench.types import BenchResult, CompletionResponse` so existing imports (`from contextops_bench.clients import BenchResult, ...`) keep working.
- **VERIFY:** `python -c "from contextops_bench.clients import BenchResult, CompletionResponse; print('ok')"`.
- **TEST:** `pytest -q` green.

### 5.1 Create the Protocol
- **WHAT:** Define the implicit client contract explicitly.
- **FILES:** `contextops_bench/client_protocol.py` (new)
- **CONTENT:**
  ```python
  """The contract every bench client satisfies."""
  from __future__ import annotations
  from typing import Protocol, runtime_checkable

  from contextops_bench.types import CompletionResponse


  @runtime_checkable
  class BenchClient(Protocol):
      PROVIDER: str
      supports_split_messages: bool

      def list_models(self) -> list[str]: ...

      def complete(self, *, model: str, messages: list[dict],
                   temperature: float = 0.0, max_tokens: int = 64,
                   system: str | None = None) -> CompletionResponse: ...
  ```
- **VERIFY:** `python -c "from contextops_bench.client_protocol import BenchClient; print('ok')"`.

### 5.2 Fix `EchoClient.complete` LSP violation
- **WHAT:** `EchoClient.complete` (clients.py:493) omits `system`. Add it so it's substitutable.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** Signature becomes
  `def complete(self, *, model, messages, temperature=0.0, max_tokens=64, system=None)`.
  Body ignores `system`. Add class attr `supports_split_messages: bool = False`.
- **VERIFY:** read the signature.
- **TEST:** `pytest tests/test_bench_unit.py -q` green.

### 5.3 Update `get_client` return type
- **WHAT:** `get_client` returns `BaseHTTPClient | EchoClient`; change to `BenchClient`.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** Import `BenchClient`, annotate `def get_client(provider: str, **kwargs) -> BenchClient:`.
- **VERIFY:** `python -c "from contextops_bench.clients import get_client; print(get_client('echo'))"`.

### 5.4 Simplify `runner.run_one`
- **WHAT:** With LSP fixed, every client accepts `system=None`. Drop the `complete_kwargs` conditional (lines 125-132) and always pass `system=`.
- **FILES:** `contextops_bench/runner.py`
- **CHANGE:**
  ```python
  if use_optimized and getattr(client, "supports_split_messages", False) and system_str:
      messages = [{"role": "user", "content": user_str or "(empty)"}]
      system_arg = system_str
  else:
      messages = [{"role": "user", "content": prompt_str or "(empty)"}]
      system_arg = None
  resp = client.complete(model=target.model, messages=messages,
                         temperature=0.0, max_tokens=32, system=system_arg)
  ```
- **VERIFY:** read; the `**complete_kwargs` conditional is gone.
- **TEST:** `pytest tests/test_bench_unit.py -q` green; `python -m contextops_bench smoke` works.

---

## Phase 6 — Bench: provider registry (DRY)

> Replaces the two hand-maintained provider lists with one registry.

### 6.1 Add `CLIENTS` registry
- **WHAT:** Derive the provider→class map from each class's `PROVIDER` attribute.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** At module bottom (after all client classes defined):
  ```python
  CLIENTS: dict[str, type] = {
      cls.PROVIDER: cls for cls in
      (OllamaClient, LMStudioClient, OpenRouterClient, AnthropicDirectClient, EchoClient)
  }
  ```
- **VERIFY:** `python -c "from contextops_bench.clients import CLIENTS; print(sorted(CLIENTS))"`.

### 6.2 Rewrite `get_client` to use the registry
- **WHAT:** Replace the if/elif chain (clients.py:528-538) with a dict lookup.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:**
  ```python
  def get_client(provider: str, **kwargs) -> BenchClient:
      p = provider.lower()
      if p not in CLIENTS:
          raise ValueError(f"Unknown provider: {provider}. Use one of: {sorted(CLIENTS)}.")
      return CLIENTS[p](**kwargs)
  ```
- **VERIFY:** `pytest tests/test_bench_unit.py::test_get_client_factory -q` green.
- **TEST:** full `pytest -q` green.

### 6.3 `__main__` derives `choices` from registry
- **WHAT:** `__main__.py:28` hardcodes the choices list; derive it.
- **FILES:** `contextops_bench/__main__.py`
- **CHANGE:** Import `CLIENTS`; change line 28 to `choices=sorted(CLIENTS)`.
- **VERIFY:** `python -m contextops_bench smoke --provider bogus 2>&1 | grep -i "error"` shows argparse rejecting `bogus`.
- **TEST:** `python -m contextops_bench smoke` works.

---

## Phase 7 — Bench: decompose `OpenRouterClient.complete` (SRP)

> The 135-line god method becomes 5 small functions. Do these one at a time,
> running tests after each extraction.

### 7.1 Extract `_extract_usage(raw)` → small dataclass
- **WHAT:** Lines 416-441 (usage parsing + cost) become a helper returning a small dataclass.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** Define `@dataclass UsageMetrics:` with `prompt_tokens`, `completion_tokens`, `cached_tokens`, `cache_creation_tokens`. Method `_extract_usage(self, raw) -> UsageMetrics` encapsulates the fallback chain (clients.py:420-432). Cost now comes from `estimate_cost` (Phase 4.2 already done).
- **VERIFY:** read.
- **TEST:** `pytest -q` green.

### 7.2 Extract `_shape_messages(model, messages, system, cache_mode)`
- **WHAT:** Lines 347-378 (Anthropic vs non-Anthropic system shaping, the `split_at` heuristic) become a pure function.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** New method returning `list[dict]`. Extract `200` and `//4` into named module constants `INTRO_SPLIT_MAX = 200`, `INTRO_SPLIT_DIVISOR = 4` with comments.
- **VERIFY:** read.
- **TEST:** add a unit test calling `_shape_messages` with a known system string and asserting the two-block structure for an Anthropic model.

### 7.3 Extract `_apply_provider_pinning(payload, model, cache_mode)`
- **WHAT:** Lines 396-410 (provider pinning) become a method that mutates `payload` in place.
- **FILES:** `contextops_bench/clients.py`
- **VERIFY:** read.
- **TEST:** `pytest -q` green.

### 7.4 Isolate the debug-print side effect
- **WHAT:** Lines 447-452 (`print(...)` behind `OPENROUTER_DEBUG_PROVIDER`) become `_debug_log(provider, cached_tokens, prompt_tokens)` — still side-effectful, but isolated and named.
- **FILES:** `contextops_bench/clients.py`
- **VERIFY:** `grep -n "print(" contextops_bench/clients.py` only inside `_debug_log`.

### 7.5 Move env-var reads to constructor
- **WHAT:** `OPENROUTER_CACHE_MODE`, `OPENROUTER_PROVIDER_PIN`, `OPENROUTER_DEBUG_PROVIDER` are read on every `complete()` call. Read once in `__init__` and store as instance attrs.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** In `__init__` add `self.cache_mode = os.environ.get("OPENROUTER_CACHE_MODE", "per_block")` etc. Replace env reads in `complete` with instance attrs.
- **VERIFY:** `grep -n "os.environ" contextops_bench/clients.py` only in `__init__` methods.
- **TEST:** `pytest -q` green; `python -m contextops_bench smoke` works.
- **COMMIT Phase 7** — SRP win.

---

## Phase 8 — Bench: runner correctness fixes

### 8.1 Fix the fake p95 percentile
- **WHAT:** `runner._stats` (lines 372-375) computes `sorted(latencies)[int(len*0.95)]` (no interpolation, returns `0.0` for `n<20`). Replace with nearest-rank percentile, document small-N behavior.
- **FILES:** `contextops_bench/runner.py`
- **CHANGE:**
  ```python
  def _percentile(values: list[float], p: float) -> float:
      """Nearest-rank percentile. Returns 0.0 for empty input."""
      if not values:
          return 0.0
      s = sorted(values)
      rank = max(1, math.ceil(p / 100 * len(s)))   # 1-indexed
      return s[min(rank, len(s)) - 1]
  ```
  Then `latency_ms_p95 = round(_percentile(latencies, 95), 1)`. Remove the `len >= 20` gate.
- **VERIFY:** add `test_percentile_small_n` and `test_percentile_large_n` to `test_bench_unit.py`; verify p95 of `[1,2,3,4,5,6,7,8,9,10]` is `10` (rank=ceil(9.5)=10) and p95 of `[1]*100 + [999]` returns a sensible value.
- **TEST:** `pytest -q` green.

### 8.2 Dedup sign-formatting in `render_summary`
- **WHAT:** Lines 405, 407 both compute `sign = "+" if x >= 0 else ""`.
- **FILES:** `contextops_bench/runner.py`
- **CHANGE:** Use f-string signed format spec directly: `f"{d['cache_hit_rate_delta']:+.1%}"` and `f"{d['cost_per_call_delta_usd']:+.6f}"` — built-in sign handling, no local variable needed.
- **VERIFY:** read.
- **TEST:** `pytest -q` green; eyeball `python -m contextops_bench smoke` output for sane signs.

### 8.3 Fix `save_csv` empty-input behavior
- **WHAT:** Lines 270-271 silently return on empty `results` — no header written, indistinguishable from "no rows" downstream.
- **FILES:** `contextops_bench/runner.py`
- **CHANGE:** Raise `ValueError("no results to save")` on empty input — fail loud, since callers should never pass empty.
- **VERIFY:** add `test_save_csv_empty_raises`.
- **TEST:** `pytest -q` green.

### 8.4 Narrow `OllamaClient.list_models` bare except
- **WHAT:** Lines 103-107 catch all `Exception` and return `[]`, masking real bugs.
- **FILES:** `contextops_bench/clients.py`
- **CHANGE:** `except (urllib.error.URLError, ValueError, KeyError): return []` (`ValueError` covers JSON decode).
- **VERIFY:** read.
- **TEST:** `pytest -q` green.

### 8.5 Clarify `_make_fresh_client` semantics
- **WHAT:** Lines 169-177 mutate and return the same object; the docstring "Return a client with empty cache state" is misleading (implies a new instance).
- **FILES:** `contextops_bench/runner.py`
- **CHANGE:** Rename to `_reset_client_state(client) -> None` (returns nothing, mutates in place); update callers. Update the docstring to be honest about in-place mutation.
- **VERIFY:** `grep -rn "_make_fresh_client" --include="*.py" .` returns nothing.
- **TEST:** `pytest -q` green.

---

## Phase 9 — Bench: `__main__` fixes

### 9.1 Fix the `smoke`/`run_all` args-mutation bug
- **WHAT:** `smoke()` mutates the shared `args` namespace (`provider`, `parallel`, `n`), so `run_all → smoke → _execute` then `run_all → _execute` sees mutated `args.n=10` and `args.provider="echo"`.
- **FILES:** `contextops_bench/__main__.py`
- **CHANGE:** Make `smoke` operate on a shallow copy:
  ```python
  def smoke(args) -> int:
      args = argparse.Namespace(**vars(args))   # shallow copy — smoke must not leak mutations
      args.provider = "echo"
      args.parallel = 1
      args.n = 10
      return _execute(args, label="smoke", n=10, include_edge_cases=True)
  ```
- **VERIFY:** add a test that constructs a parser, parses `run_all --provider echo --n 50`, runs `smoke(args)`, and asserts `args.n == 50` afterward (unmutated).
- **TEST:** `pytest -q` green; `python -m contextops_bench run_all --provider echo --n 50` runs both phases.

### 9.2 Decompose `_execute`
- **WHAT:** The 70-line `_execute` (lines 65-132) does too much. Split into helpers.
- **FILES:** `contextops_bench/__main__.py`
- **CHANGE:**
  ```python
  def _resolve_preset(args) -> tuple[str | None, str | None]: ...
  def _build_prompt_list(n, fixed_system, fixed_tools, model, include_edge_cases) -> tuple[list, list[int]]: ...
  def _write_artifacts(out_dir, label, results, summary) -> None: ...
  def _execute(args, *, label, n, include_edge_cases=False) -> int:
      # ~20 lines orchestrating the above + run_batch + print
  ```
  Encapsulate the fragile edge-case ID arithmetic inside `_build_prompt_list` so the IDs are computed next to the place that defines what an edge-case ID is.
- **VERIFY:** read; `_execute` is now ~20 lines.
- **TEST:** `python -m contextops_bench smoke` works; CSV has expected columns.

### 9.3 Fix stale docstring
- **WHAT:** Lines 1-8 reference `python -m bench.smoke` (the module is `contextops_bench`).
- **FILES:** `contextops_bench/__main__.py`
- **CHANGE:** Update usage examples to `python -m contextops_bench smoke|local|cloud|direct|run_all`.
- **VERIFY:** read.

---

## Phase 10 — Bench: prompt_factory fixes

### 10.1 Replace global RNG reseed with local instance
- **WHAT:** `generate_one` (line 165) calls `random.seed(seed)` — mutates global RNG, perturbing any other `random` user in the process.
- **FILES:** `contextops_bench/prompt_factory.py`
- **CHANGE:** Use a local instance:
  ```python
  def generate_one(seed=None, *, fixed_system=None, fixed_tools=None, fixed_model=None) -> Prompt:
      rng = random.Random(seed) if seed is not None else random.Random()
      ... # replace all `random.xxx` with `rng.xxx`
  ```
  In `generate_many`, similarly use one `rng = random.Random(seed)` and pass `seed=None` to `generate_one` but seed via the shared rng — OR keep per-prompt seeds but use local instances (cleaner: keep per-prompt `random.Random(seed+i)`).
- **VERIFY:** `python -c "import random; from contextops_bench.prompt_factory import generate_one; generate_one(seed=42); print(random.random())"` returns the same value every run (proves global RNG untouched).
- **TEST:** `pytest tests/test_bench_unit.py -q` green; the diversity test still passes (reproducibility preserved because seeds are explicit).

### 10.2 Move preset content to data files
- **WHAT:** `REALISTIC_AGENT_SYSTEM` (~68 lines) and `REALISTIC_AGENT_TOOLS` (~98 lines of JSON-as-string) dominate the file. Move to `contextops_bench/data/`.
- **FILES:** new `contextops_bench/data/realistic_agent_system.md`, `contextops_bench/data/realistic_agent_tools.json`; modified `prompt_factory.py`.
- **CHANGE:** At module top:
  ```python
  from pathlib import Path
  _DATA = Path(__file__).parent / "data"
  REALISTIC_AGENT_SYSTEM = (_DATA / "realistic_agent_system.md").read_text()
  _tools_raw = (_DATA / "realistic_agent_tools.json").read_text()
  json.loads(_tools_raw)   # fail-fast on malformed JSON at import time
  REALISTIC_AGENT_TOOLS = _tools_raw
  ```
- **VERIFY:** `python -c "from contextops_bench.prompt_factory import REALISTIC_AGENT_TOOLS; import json; json.loads(REALISTIC_AGENT_TOOLS); print('valid json')"`.
- **TEST:** `pytest -q` green; update `MANIFEST.in` / `pyproject.toml` `package-data` if needed so data files ship with the wheel.

---

## Phase 11 — Tests backfill (gaps cluster where the bugs were)

### 11.1 `tests/test_pricing.py`
- Already added in Phase 3.4. Extend with: cross-check that `optimize()`'s savings estimate for `claude-haiku-4.5` is consistent with `pricing.PRICING` (regression test for the old $0.80/$1.00 drift).

### 11.2 `tests/test_bench_clients.py` (new)
- **WHAT:** Cover the extracted helpers without network.
- **CASES:**
  - `_shape_messages(...)` — Anthropic model + system → 2-block structure with `cache_control` on 2nd block; non-Anthropic → plain prepend; no system → unchanged.
  - `_extract_usage(raw)` — handles all 3 cached-token field names; unknown fields default to 0.
  - `EchoClient.complete(system="...")` — accepts the kwarg (LSP regression).
  - `get_client` — registry-based; unknown provider raises with helpful message listing choices.

### 11.3 Extend `tests/test_bench_unit.py`
- `_percentile` small-N and large-N.
- `save_csv` empty raises.
- `run_one` — `system=` always passed when supported (mock client that records kwargs).
- Add `test_render_order_field` — `Prompt.sections()` respects `render_order` (if not already in test_optimizer from Phase 2.2).

### 11.4 `tests/test_cli.py` (new)
- **WHAT:** `click.testing.CliRunner` smoke for the library CLI (currently 0 coverage).
- **CASES:** `contextops optimize --system S --query Q --model gpt-4o` exits 0 with valid JSON; `contextops --version` reports `0.3.0`; `compare` with the two `evals/*.json` files runs.

### 11.5 `tests/test_anthropic_integration.py` (new, network-gated)
- **WHAT:** The one durable artifact from the diag-script investigation. Skip unless `ANTHROPIC_API_KEY` set.
- **CASES:** 5 sequential calls through `AnthropicDirectClient` with the realistic-agent preset; assert call 2+ has `cache_read_input_tokens > 0`. This is exactly what `diag_pinned_v2.py` probed manually and what the whole in-flight feature is meant to guarantee.
- **VERIFY:** `pytest -q` skips it locally (no key); runs in CI when secret is added.

---

## Phase 12 — Docs

### 12.1 Fill `[Unreleased]` in CHANGELOG.md
- Group Phase 0–11 changes under `Added` / `Changed` / `Fixed` per Keep a Changelog. Notable entries: `Prompt.render_order` field (Added, public API); shared `contextops.pricing` module (Added); `BenchClient` Protocol (Added); `OpenRouterClient.complete` decomposed (Changed, internal); p95 percentile bug (Fixed); `smoke`/`run_all` args mutation (Fixed); version drift (Fixed).

### 12.2 README updates
- Document the `direct` subcommand and `--preset-agent` flag (both new in the working tree, not yet in README).
- Add a single "Environment variables" section listing all 5 (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OPENROUTER_CACHE_MODE`, `OPENROUTER_PROVIDER_PIN`, `OPENROUTER_DEBUG_PROVIDER`) in one place rather than scattered.

### 12.3 Note the diag-script relocation
- One line in CHANGELOG: "Diagnostic cache-control probes moved to `scripts/diag/`."

---

## Final verification

After Phase 12:

```bash
python -m pytest -v --tb=short          # all green
python -m contextops_bench smoke         # <30s
python -m contextops_bench smoke --n 100 --parallel 4
grep -rn "_bench_render_order\|_maybe\|EDGE_CASE_PROMPT_IDS\|__import__" --include="*.py" .   # empty
grep -rn "1_000_000" contextops_bench/   # empty (cost math centralized)
scripts/test_local.sh                    # full local CI
```

---

## Phase priority (if you want to do less than the full plan)

| Priority | Phases | Why |
|---|---|---|
| **Highest leverage** | 2, 3, 4, 9.1 | Kill the monkeypatch; kill the worst duplication + pricing drift; fix a real user-facing bug |
| **High value, low risk** | 0, 6, 8.1, 8.5 | Trivial cleanups, registry DRY, percentile correctness, honest naming |
| **Refactor polish** | 5, 7, 9.2, 10 | Protocol/ABC, god-method decomposition, RNG isolation, data-file move |
| **Hygiene** | 1, 11, 12 | Relocate scratch scripts, backfill tests, update docs |

Each phase is independently shippable as one commit. If something breaks,
`git revert` one phase — phases don't cross-depend (except 3→4 and 2→5, which
are explicitly noted).

---

## Cross-phase dependency graph

```
0 (cleanups) ──────────────────────────────────── standalone
1 (diag move) ─────────────────────────────────── standalone
2 (render_order field) ─────┐
                            ├─► 5.4 (run_one simplification needs the field)
3 (pricing module) ──► 4 (bench cost dedup)       3 also unblocks nothing else
6 (registry) ──────────────────────────────────── needs 5.3 (BenchClient type) ideally but not strictly
7 (OpenRouter decompose) ─── needs 4.2 done first (uses estimate_cost)
8 (runner fixes) ──────────────────────────────── standalone (8.1, 8.5 independent)
9 (__main__ fixes) ─── 9.2 benefits from 6.3 but works either way
10 (prompt_factory) ───────────────────────────── standalone
11 (tests) ──────────── backfills coverage for whatever phases landed
12 (docs) ──────────── written last to reflect what shipped
```
