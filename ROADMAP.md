# ContextOps Roadmap

This document explains **what we're building next and why**, in enough detail
that a junior engineer can pick up any section and start implementing without
needing a design meeting first. It expands on the short bullet list in the
[`README.md`](README.md#-roadmap) roadmap section.

If you're new to the codebase, read this first:
- `contextops/` — the core library (prompt model, optimizer, eval, CLI, local SQLite logger).
- `contextops_bench/` — the benchmark harness (drives real LLM providers, measures cache hit rate / cost, generates the public dashboard).
- `docs/dashboard/index.html` — auto-generated from `bench/results/*.summary.json` by `scripts/generate_dashboard.py`.

There are four tracks. They're independent — you can work on any one without
blocking the others.

| Track | Codename | Effort | Status |
|---|---|---|---|
| A | RAG Curator | Large (multi-week) | Core + eval integration shipped (v0.4) |
| B | Access-Aware Context + Audit Trail | Large (multi-week) | Not started — design only |
| C | Bench/Dashboard Maturity | Small (days) | Partially done — quality gates shipped, backfill pending |
| D | New Providers/Metrics | Small (hours-days each) | `vllm`/`tgi` + `safety`/`format_compliance` shipped — more welcome |

---

## Track A — RAG Curator (v0.4) ✅ done

**Implemented**: `contextops/curator.py` (`DocumentChunk`, `CuratorConfig`,
`CurationResult`, `curate()`, `build_documents()`, `context_precision()`),
`Prompt.from_chunks()` in `contextops/models.py`, the `contextops curate` CLI
command, and optional `curated_chunks`/`curated_chunks_baseline`/
`curated_chunks_optimized` params on `evaluate()`/`evaluate_ab()` in
`contextops/eval.py` (the stretch goal below — done via a deterministic
n-gram-overlap heuristic, not an LLM judge, to keep it free and fast). See
`tests/test_curator.py` and the new tests in `tests/test_v02_eval.py`.

**Also implemented (real-LLM validation, beyond the original stretch goal)**:
`contextops_bench/curator_bench.py` — a dedicated bench that runs a synthetic
noisy-retrieval dataset (`generate_curation_dataset()`, deterministic given a
seed) through a real LLM twice per query (raw/uncurated vs `curate()`-filtered
chunks), measuring token/cost deltas AND answer-quality impact (judge metrics
+ `context_precision`), via `python -m contextops_bench curator`. This is a
different independent variable from the existing cache-hit-rate bench in
`contextops_bench/runner.py` (content filtering vs section ordering), so it's
a separate module reusing only the low-level LLM client plumbing. Results
(`*.curator_summary.json`) feed a new "RAG Curator Bench" section in
`scripts/generate_dashboard.py`, rendered independently of the existing
cache/cost charts. See `tests/test_curator_bench.py` and the extended
`tests/test_dashboard_generator.py`.

One deviation from the original design below: recency uses a proper
half-life formula (`0.5 ** (age_days / half_life_days)`, so a chunk aged
exactly `half_life_days` scores 0.5), not the `exp(-age_days/half_life_days)`
sketch — the exp version decays to `1/e` (~0.37) at that age, not `0.5`,
which would have made `half_life_days` a misleading parameter name.

### The problem, in plain English

ContextOps today answers the question **"in what order should I send my
prompt sections?"** — it doesn't ask **"should this content be in the prompt
at all?"**.

If you're building a RAG (Retrieval-Augmented Generation) app, you typically
retrieve, say, the top-10 most similar document chunks from a vector database
and stuff all 10 into the `documents` section of your prompt
(`contextops/models.py::Prompt.documents`). But:

- Some of those 10 chunks are irrelevant (the retriever isn't perfect —
  similarity search is a proxy, not ground truth).
- More documents = more tokens = higher cost AND worse LLM answers
  (irrelevant context increases hallucination risk — this is a
  well-documented "lost in the middle" / distractor effect).
- Right now, ContextOps just reorders whatever you hand it. Garbage in,
  reordered garbage out.

**The RAG Curator's job: filter the retrieved documents down to only the
ones worth paying tokens for, using more than one signal, before they ever
reach the optimizer.**

### Why "multi-signal"?

Relying on a single signal (e.g. cosine similarity score from your vector
DB) is fragile — a chunk can have high similarity but be stale, or be from a
low-trust source, or be near-duplicate of another chunk already included.
The curator should combine several signals into one confidence score:

| Signal | What it measures | Where it comes from |
|---|---|---|
| **Semantic similarity** | How close the chunk's embedding is to the query | Passed in by the caller (we don't want to force a specific vector DB or embedding model) |
| **Recency** | How old the chunk's source document is | Optional metadata field on the chunk (e.g. `updated_at`) |
| **Redundancy** | Is this chunk saying the same thing as another chunk already selected? | Computed by the curator itself (e.g. simple n-gram overlap, or an optional embedding-based dedup) |
| **Source trust / metadata** | Is this from a pinned/verified doc vs. a random scrape? | Optional metadata field the caller provides (e.g. `trust_score`) |

Each signal is normalized to 0..1, combined into a weighted score, and
compared against a **strict threshold**. If a chunk's combined score is below
the threshold, it's dropped — even if it's better than doing nothing, "strict"
means we'd rather under-include than dilute the prompt with noise.

### Where this lives in the codebase

New module: `contextops/curator.py`. Suggested shape:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class DocumentChunk:
    text: str
    similarity: float              # 0..1, required — caller computes this
    updated_at: float | None = None    # unix timestamp, optional
    trust_score: float | None = None   # 0..1, optional, caller-provided
    metadata: dict | None = None

@dataclass
class CuratorConfig:
    weights: dict[str, float]      # e.g. {"similarity": 0.6, "recency": 0.2, "trust": 0.2}
    threshold: float = 0.6         # strict cutoff — chunks below this are dropped
    dedup_threshold: float = 0.9   # chunks more similar to each other than this are considered duplicates
    max_chunks: int | None = None  # optional hard cap even after filtering

@dataclass
class CurationResult:
    kept: list[DocumentChunk]
    dropped: list[tuple[DocumentChunk, str]]   # (chunk, reason) — for debugging/audit
    scores: list[float]                         # combined score per kept chunk, same order as `kept`

def curate(chunks: list[DocumentChunk], config: CuratorConfig) -> CurationResult:
    ...
```

Then in `contextops/models.py`, add a convenience method or a new
`Prompt.from_chunks(...)` constructor that takes `list[DocumentChunk]`,
runs `curate()`, and joins the kept chunks into the `documents` field —
so the existing `optimize()` pipeline in `contextops/optimizer.py` doesn't
need to change at all. The curator is a **pre-processing step**, not a
change to the reordering logic.

### Concrete plan of implementation steps

1. **Design the scoring function.** Start with a simple weighted sum:
   `score = w1*similarity + w2*recency_score + w3*trust_score`, where
   `recency_score = exp(-age_days / half_life_days)` (a standard decay
   curve — a doc from today scores ~1.0, a doc from a year ago scores much
   lower, controlled by `half_life_days`).
2. **Implement dedup.** Simplest version: pairwise token-set Jaccard
   similarity between kept chunks; if a candidate is >`dedup_threshold`
   similar to an already-kept chunk, drop it as a duplicate (keep the
   higher-scored one). Don't over-engineer this — no need for a real
   embedding index for v1, a simple bag-of-words overlap is a fine first
   pass and is dependency-free.
3. **Implement `curate()`** combining scoring + threshold + dedup + optional
   `max_chunks` cap. Every dropped chunk should carry a human-readable
   reason string (`"below threshold (0.42 < 0.6)"`, `"duplicate of chunk #3"`)
   — this is important for debugging bad RAG answers later.
4. **Wire into `Prompt`** via a new constructor/helper (don't change the
   existing `Prompt.documents: str` field type — keep backward compatibility).
5. **CLI support**: add `contextops curate` subcommand (see `contextops/cli.py`
   for the existing `optimize`/`compare`/`eval` command patterns) that takes
   a JSON file of chunks + a query, prints kept/dropped with scores.
6. **Tests**: unit tests for scoring math (deterministic — no LLM calls
   needed, this is pure functions), dedup logic, threshold edge cases
   (all above threshold, all below, exactly at threshold).
7. **Eval integration (stretch goal)**: extend `contextops/eval.py`'s
   `compare()`/`evaluate_ab()` to report a "context precision" delta —
   how many kept chunks were actually cited/used in the LLM's response
   (requires a judge-style check, can reuse `contextops/judge.py`'s pattern).

### Acceptance criteria (how you know it's done)

- `curate()` is a pure function with no network calls, fully unit-testable. ✅
- Given a synthetic set of chunks with known similarity/recency/trust
  values, `curate()` produces the expected kept/dropped split. ✅
- Dedup correctly drops near-identical chunks. ✅
- `contextops curate` CLI command works end-to-end on a sample JSON file. ✅
- README gets a new "RAG Curator" section under "What's in the box" with
  a runnable example, following the same style as the existing sections. ✅

---

## Track B — Access-Aware Context + Audit Trail (v1.0)

### The problem, in plain English

This is an **enterprise/on-prem** feature. Imagine a company chatbot that
different employees use — a support agent, a manager, and an external
contractor might all ask the same question, but the context (documents,
history) available to each of them should differ based on what they're
allowed to see. Today, `contextops/models.py::Prompt` has no concept of
"who is asking" — any content you put in `documents`/`context`/`history`
goes to the LLM, full stop.

**Two related but separable pieces:**

1. **Access-aware context**: redact/filter prompt sections based on the
   caller's role/permissions before the prompt is optimized and sent.
2. **Audit trail**: log (locally — this project is local-first, see
   `contextops/logger.py`) exactly what context was included vs. excluded
   for every call, and why, so a security/compliance review can answer
   "did user X's chatbot session leak content they shouldn't have seen?"

### Why this is separate from Track A

The RAG Curator (Track A) filters based on **relevance** (is this chunk
useful?). Access control (Track B) filters based on **permission** (is this
chunk *allowed* to be shown to this specific user?), which is a hard
security boundary — a chunk can be perfectly relevant and still must be
redacted if the user lacks access. These need different failure modes:
Track A failing just makes answers worse; Track B failing is a security
incident. They will likely compose (redact first, then curate what's left),
but should be built and tested independently.

### Where this lives in the codebase

New module: `contextops/access.py`. Suggested shape:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Principal:
    """Who is making this request."""
    id: str
    roles: set[str]
    attributes: dict[str, str] | None = None   # e.g. {"department": "finance"}

@dataclass
class AccessPolicy:
    """A single rule: does `principal` get to see content tagged `required_roles`?"""
    required_roles: set[str]              # empty set = no restriction (public)
    section: str | None = None            # None = applies to all sections

class PolicyEngine(Protocol):
    def is_allowed(self, principal: Principal, tag: str) -> bool: ...

@dataclass
class TaggedContent:
    """A piece of content (a document chunk, a history message) with access tags."""
    text: str
    section: str            # matches contextops.models.Section
    required_roles: set[str] = None   # who can see this

@dataclass
class RedactionResult:
    prompt: "Prompt"                          # the filtered Prompt, ready for optimize()
    redacted: list[tuple[TaggedContent, str]]  # (content, reason) for everything removed

def apply_access_policy(
    contents: list[TaggedContent],
    principal: Principal,
    engine: PolicyEngine,
) -> RedactionResult:
    ...
```

For the audit trail, **extend the existing SQLite schema** in
`contextops/logger.py` rather than building a new logging system —
this project is intentionally local-first / SQLite-only, don't add a new
storage backend. Add a new table:

```sql
CREATE TABLE IF NOT EXISTS access_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    call_id INTEGER,                 -- foreign key to `calls.id`, nullable (redaction can happen before a call is even logged)
    section TEXT NOT NULL,
    action TEXT NOT NULL,            -- "included" | "redacted"
    reason TEXT,
    content_hash TEXT NOT NULL       -- hash of the content, NOT the content itself (don't duplicate sensitive data into the audit log)
);
```

**Important design constraint**: the audit log must NOT store the actual
redacted content — only a hash (for correlation) and metadata (why it was
redacted). Otherwise the audit trail itself becomes a data leak.

### Concrete plan of implementation steps

1. **Define the data model** (`Principal`, `AccessPolicy`/`PolicyEngine`,
   `TaggedContent`) in `contextops/access.py`. Start with a simple built-in
   `PolicyEngine` implementation: `RoleBasedPolicyEngine` that just checks
   `principal.roles & content.required_roles` is non-empty (or
   `required_roles` is empty = public).
2. **Implement `apply_access_policy()`** — pure function, no I/O. Takes
   tagged content + a principal, returns what's allowed plus a redaction
   list with reasons.
3. **Extend `contextops/logger.py`** with the `access_audit` table and a
   `Logger.log_access(...)` method following the existing `Logger.log()`
   pattern (see lines 61-86 of `contextops/logger.py` for the pattern to
   copy).
4. **Wire into the CLI** (optional for v1, nice-to-have): a `--principal-role`
   flag on `contextops optimize` that, if content is tagged, applies
   redaction before optimizing.
5. **Tests**: unit tests for policy matching logic (role overlap, public
   content, multiple roles), redaction reason strings, and the new SQLite
   table (follow the pattern in existing logger tests — check
   `tests/` for `test_logger*.py` if it exists, otherwise model off
   `tests/test_bench_quality.py`'s style of pure-function testing).
6. **Security review checklist** (write this as a comment or a short doc):
   confirm the audit table never stores raw content, confirm redaction
   happens BEFORE `optimize()` (never redact after — the optimizer must
   never see content the principal isn't allowed to see, so redaction is
   not just cosmetic).

### Acceptance criteria (how you know it's done)

- `apply_access_policy()` is a pure function, fully unit-tested with no
  network/LLM calls.
- The audit table stores content **hashes**, never raw content — write a
  test that explicitly asserts this (e.g. assert the redacted text never
  appears verbatim in any audit row).
- A redacted section never reaches `optimize()` — write an integration
  test proving the full pipeline (tag → redact → optimize) drops
  disallowed content.
- README gets a new section under "What's in the box".

---

## Track C — Bench/Dashboard Maturity

### Where we are

We just shipped (this session):
- Deterministic quality gates (`contextops_bench/quality.py`):
  `min_n=20`, `max_error_rate=15%`, cost-delta 95% CI must exclude zero,
  known cache-marker-drop paths auto-disqualify.
- Ground-truth `provider`/`model` + `generated_at` timestamp persisted
  into every `summary.json` (`contextops_bench/runner.py::summarize()`).
- Error classification (`classify_error()`) and confidence scoring
  (`confidence_level()`) — auth errors (401/403) force `confidence="invalid"`.
- Dashboard (`scripts/generate_dashboard.py`) shows Verified / Confidence /
  Run date columns.

### What's left (smaller, well-scoped tasks — good for a junior engineer)

1. **Backfill stale runs.** Of the 8 committed runs in `bench/results/`,
   only 2 pass the quality gate (`openai/gpt-4o-mini`, `zen/claude-sonnet-4-6`).
   The rest fail for fixable reasons:
   - `zen_n5.summary.json` — only n=5, needs `min_n=20`. Re-run with
     `python -m contextops_bench cloud --provider direct_zen --model claude-sonnet-4-6 --preset-agent realistic --n 30`.
   - `cloud_anthropic_*.summary.json` — these went through OpenRouter,
     which drops the Anthropic cache marker (`cache_marker_dropped=True`,
     see `contextops_bench/quality.py::CACHE_MARKER_DROP_PATHS`). Re-run
     via `--provider direct_anthropic` instead of OpenRouter to get a
     measurement path that can actually pass the gate.
   - `local_ollama.summary.json`, `mac_zen_n5.summary.json` — missing the
     cost-delta confidence interval entirely (pre-date the CI computation
     in `runner.py`). Just need a fresh run with the current code.

   **Task**: re-run each of these, replace the old `.summary.json`/`.csv`
   pair in `bench/results/`, regenerate the dashboard
   (`python scripts/generate_dashboard.py`), commit.

2. **CI gate on bench data quality.** `.github/workflows/bench-regression.yml`
   currently only checks `cache_hit_rate_p50` against a threshold. Add a
   check that fails the workflow if a *newly added* `summary.json` in a PR
   has `quality.verified == False` — this stops unverified data from ever
   reaching the public dashboard again. Look at `scripts/ci_bench_gate.py`
   for the existing gate pattern to extend.

3. **Historical trend charts.** Now that every summary has `generated_at`,
   the dashboard could show a time series of cache-hit-rate / cost-delta
   per provider+model over multiple runs (right now each provider+model
   only shows its *latest* run — there's no history). This requires:
   - Deciding whether to keep multiple `summary.json` files per
     provider+model (e.g. timestamped filenames) instead of overwriting.
   - A new chart in `scripts/generate_dashboard.py` (reuse the existing
     Chart.js setup — look at the `hitChart`/`costChart` canvas blocks).
   - This is the most open-ended item in this track — scope it down before
     starting; a simple version is "show the last 5 runs per provider+model
     as a sparkline" rather than a full time-series database.

4. **Confidence-based sorting.** Currently the "All runs" table is sorted
   by filename. Consider sorting by `verified` status then `confidence`
   so the most trustworthy rows are at the top — small UX win, one-line
   change in `scripts/generate_dashboard.py::_load_runs()` or `main()`.

### Acceptance criteria

- All (or a documented subset) of `bench/results/*.summary.json` show
  `verified: true` OR have a clear, permanent reason logged for why they
  can't (e.g. cache-marker-drop paths will never pass, and that's fine —
  document it in the dashboard).
- CI fails on new unverified bench data being merged.
- `pytest` still green, `ruff`/`mypy` clean (same bar as every other change
  in this repo — see `CONTRIBUTING.md`).

---

## Track D — New Providers / Metrics

This is the **lowest-effort, highest-parallelism** track — good for
onboarding a new contributor or picking up between bigger tasks. Two
independent sub-tracks:

### D1. New bench providers ✅ done (`vllm`, `tgi`)

Currently supported (`contextops_bench/clients.py`): `ollama`, `lmstudio`,
`vllm`, `tgi`, `openrouter`, `direct_anthropic`, `direct_openai`,
`direct_google`, `direct_zen`, plus `EchoClient` for offline tests.

`VLLMClient` subclasses `OllamaClient` (OpenAI-compatible
`/v1/chat/completions`, `cost_usd=0.0`). `TGIClient` targets TGI's native
`/generate` endpoint, flattens `messages` into a single `inputs` string, and
estimates token counts with `tiktoken` since TGI reports no usage. Both are
wired into `get_client()` and the `--provider` CLI choice; neither is in
`CACHE_BEARING_PROVIDERS` (self-hosted, no cache economics to measure).

**To add another new provider** (same template as above):

1. Look at `OllamaClient` in `contextops_bench/clients.py` (line ~93) as
   the template — vLLM and TGI both expose OpenAI-compatible
   `/v1/chat/completions` endpoints, so a new client is often just a thin
   subclass of `BaseHTTPClient` with a different default `base_url` and
   possibly different cache-token field names in the response JSON.
2. Register it in `contextops_bench/clients.py::get_client()` (the
   provider-name → class dispatch function).
3. Add a CLI alias in `contextops_bench/__main__.py` if the provider needs
   special handling (e.g. does it support cache_control markers? does it
   need an API key?).
4. Add unit tests in `tests/test_bench_unit.py` following the pattern used
   for the existing `direct_openai`/`direct_google` clients (mock the HTTP
   response, assert the parsed `CompletionResponse` fields are correct —
   no real network calls in unit tests).
5. Update the README's "Bench harness" section and the provider comparison
   table.

**Note**: since vLLM/TGI serve self-hosted models, there's no cache
markers/billing to measure the way there is for Anthropic/OpenAI/Google —
the main value is token-count/latency measurement, not cache-hit-rate
economics. Set expectations accordingly in the PR description.

### D2. New eval judge metrics ✅ done (`safety`, `format_compliance`)

Currently supported (`contextops/judge.py::_METRICS`): `faithfulness`,
`relevance`, `completeness`, `conciseness`, `safety`, `format_compliance`.

`safety` follows the `conciseness` shape (`{response}` only) but **fails
closed**: `default_score_if_missing=0.0`, not `0.5` — an unparseable safety
verdict should read as "unsafe until proven otherwise", not blend into the
middle of the score distribution. `format_compliance` reuses the existing
`expected` field (already threaded through `score_one`/`score_many`/
`eval.py`) to carry the **required format spec** instead of an expected
answer (e.g. `expected="valid JSON with keys: name, age"`) — this avoided
extending `contextops/dataset.py`'s `DatasetItem` schema. Document this
`expected`-as-format-spec convention clearly wherever `format_compliance` is
used, since `expected` normally means something else for other metrics.

**To add another new metric** (same template as above):

1. Add a new entry to the `_METRICS` dict in `contextops/judge.py` (see
   lines 25-113 for the existing pattern, including `safety` and
   `format_compliance`): a `description`, a `system` prompt describing what
   "good" means for this metric, a `user` prompt template with the fields
   needed (`{response}`, and optionally `{context}`/`{query}`/`{expected}`
   — reuse `{expected}` for spec-like inputs rather than adding new
   `DatasetItem` fields, following `format_compliance`'s precedent), and a
   `default_score_if_missing` fallback (default to `0.5` for quality
   metrics; consider failing closed to `0.0`/`1.0` for safety-critical
   metrics, as `safety` does).
2. Check `contextops/eval.py::evaluate_ab()` to see how metrics are wired
   into the A/B report — a new metric name in `_METRICS` should "just work"
   once added, since the eval loop iterates over whatever metrics you pass
   in via `--metrics` (CLI) or the `metrics=[...]` parameter (Python API).
3. Add unit tests in `tests/test_v02_eval.py` following the pattern used
   for `safety`/`format_compliance` (offline `CallableJudge`/`EchoJudge`,
   no real LLM calls).
4. Update the README's "LLM-as-judge eval" feature row and CLI example
   (`--metrics relevance,completeness,...`) to mention the new metric.

### Acceptance criteria

- New provider: unit tests pass with mocked HTTP responses, `smoke` test
  (`python -m contextops_bench smoke`) still passes (uses `EchoClient`,
  unaffected by new providers, but confirms nothing broke the dispatch
  logic). ✅ verified for `vllm`/`tgi`.
- New metric: unit tests pass with a mocked judge client, CLI
  `--metrics <new_metric>` works end-to-end against the echo judge
  (`contextops eval --echo --run-fn echo ...`). ✅ verified for
  `safety`/`format_compliance`.

---

## How to pick up any of these tracks

1. Read the relevant section above fully before touching code.
2. Check `CONTRIBUTING.md` for the PR workflow, commit conventions, and
   how releases/`CHANGELOG.md` entries work.
3. Run the full test suite before and after your change:
   `pytest` (should stay green — currently 171 tests passing).
4. Run `ruff check .` and `mypy contextops contextops_bench` before
   committing — this repo has zero tolerance for lint/type errors on
   touched files.
5. Add tests for anything new — see `tests/test_bench_quality.py` for a
   good example of testing pure functions without any network/LLM
   dependency, which is the style this project prefers wherever possible.
