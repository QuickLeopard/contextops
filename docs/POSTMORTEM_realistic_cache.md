# v0.3.1 — realistic-preset cache key regression fix

> **Release notes / PR description for `v0.3.1`. This file is the source
> of truth — paste the body below into GitHub Release / PR description
> forms as needed. The CHANGELOG entry is the canonical compressed
> version.**

## TL;DR

The `realistic` agent preset that ships with the bench harness had a
**cache key regression**: `role` was randomized per call, silently
invalidating Anthropic prompt cache every time. The optimized arm was
paying the 1.25× cache-write surcharge and getting zero cache-reads
back — directly contradicting the purpose of the bench. After fixing
(3-line patch in `prompt_factory.py`), the optimized arm is **90%
cheaper per call** at n=30 with 89.2% mean cache hit rate.

## The bug

The bench harness sends the "stable" prefix (system prompt + tool schema
+ agent role) as an Anthropic `system[]` content block with
`cache_control: {type: "ephemeral"}`. Anthropic's prompt cache keys on
exact-match hashing of that prefix.

The `realistic` agent preset pinned `system` and `tools` to constants,
but `generate_one` did this for the `role` section:

```python
role=random.choice(ROLE_PROMPTS) if random.random() < 0.6 else "",
```

Six candidate role strings (`weather-agent`, `code-assistant`,
`data-analyst`, `support-bot`, `translator`, `""`...) rotating across
calls meant the cache key rotated too. Anthropic correctly reported
`cache_creation_input_tokens > 0` and `cache_read_input_tokens = 0`
every single call — the standard "I have never seen this prefix before"
signal.

**Cost impact** (per Anthropic Sonnet 4.6 pricing):
- A normal input token: $3.00 / M
- A cache-read token: $0.30 / M (90% discount)
- A cache-write token: $3.75 / M (25% surcharge)

So every "optimized" call was paying +$0.0026 more than the baseline
call (the 0.25× surcharge on 3,476 system tokens) and getting nothing
back. Optimized was *more expensive* than baseline per call — exactly
the opposite of what the bench was supposed to demonstrate.

## The fix

```diff
# contextops_bench/prompt_factory.py
AGENT_PRESETS: dict[str, dict[str, str]] = {
    "realistic": {
        "system": REALISTIC_AGENT_SYSTEM,
        "tools": REALISTIC_AGENT_TOOLS,
+       # Identity is part of the agent definition — must be constant for
+       # cache to be hit across calls. Randomizing role rotates the cache
+       # key and silently turns every call into a cold cache_write.
+       "role": "code-assistant",
    },
}
```

Plus 3 lines in `generate_one` / `generate_many` to thread a
`fixed_role` parameter through (mirroring `fixed_system`, `fixed_tools`,
`fixed_model`). One line in `__main__._execute` to read `role` from
the preset and log it at startup — so any future regression in
preset-pinning is immediately visible in the bench output.

## Why this wasn't caught earlier

Three layers of "it works on my machine" compounded:

1. **The bench harness itself wasn't running against Anthropic until
   `direct_zen`/`direct_anthropic` providers landed.** Tests ran against
   `EchoClient` (offline stub that simulates cache behavior with a
   prefix-match algorithm that *did* tolerate per-call role changes
   in degraded mode). So `pytest` was happy while `claude-sonnet-4-6`
   was unhappy.

2. **OpenRouter's OpenAI-compatible adapter drops the `cache_control`
   marker during OpenAI → Anthropic translation,** so cache reads
   always show 0 even with correct setup. The bench going through
   OpenRouter first gave a false signal that "cache control doesn't
   work". Switching to OpenCode-ZEN (which passes Anthropic-native
   `cache_control` through unchanged) was what made the
   `cache_creation` signal visible.

3. **The 6/30 vs 0/5 ratio across two n runs.** A first n=30 run with
   the bug still in place got 6/30 cache hits — looking vaguely
   plausible, masking the systematic failure. The randomness of the
   `role` rotation meant *some* prompts shared role with a previous
   call, leading to a "bursty, account-size-limited cache pool"
   hypothesis (suggesting cache TTL constraints, account size, etc.)
   that was actually a red herring.

## Verification (OpenCode-ZEN, `--preset-agent realistic`, --n 30)

| | Optimized | Baseline | Δ |
| --- | --- | --- | --- |
| Mean prompt tokens | 398 | 3,750 | **−3,352** |
| Mean cached tokens | 3,364 | 0 | **+3,364** |
| Cache hit rate | **89.2%** | 0.0% | **+89.2pp** |
| Cost / call | **$0.00107** | $0.01062 | **−$0.00955** |
| Total run cost | $0.032 | $0.319 | **−$0.287** |

Across a single 60-call A/B run, the optimized arm ran for $0.032 vs
$0.319 baseline — **saved $0.287 per run**. Project that to an agent
making 10K similar calls/day: ~$95/day, ~$2,860/month saved on a single
Sonnet-4.6 instance.

(A few transport errors during the run — 1 optimized, 2 baseline —
unrelated to caching. The comparison stands.)

## What users need to do

Nothing. The fix is fully internal. If you were running the bench
harness against `direct_zen` or `direct_anthropic` with the `realistic`
preset, your next `pip install --upgrade contextops-tool` will give you
correct cache measurements.

If you were running with `--fixed-system` and `--fixed-tools` only (no
`--preset-agent`), the bench was always using randomized role — so your
prior measurements would have been symptomatic too. Re-run after
upgrading to confirm.

## Migration notes

- The pinned role value is `"code-assistant"` — chosen to be
  semantically consistent with the `REALISTIC_AGENT_SYSTEM` text
  ("You are Atlas, a senior software engineering assistant").
- If you need a different role for your real agent, pass
  `--fixed-role "..."` to override (available since this release).
- The bench startup log now includes `role=...` alongside the existing
  `system~N chars` and `tools~N chars` lines, so any future regression
  in preset-pinning is visible at first glance.

## Related work (bundled in the same release)

The fix alone wouldn't have been measurable without the bench
infrastructure to actually surface cache reads/writes. Also included:

- **`direct_zen` provider** — OpenCode-ZEN gateway (Anthropic-native
  API format, `x-api-key` auth, `opencode-cli/0.5.0` User-Agent to
  avoid Cloudflare WAF 1010).
- **`direct_anthropic` provider** — direct Anthropic API path for users
  with `ANTHROPIC_API_KEY`.
- **Cache-control marker wiring in `run_one`** — sends the stable
  prefix as a separate `system[]` content block with
  `cache_control: {type: "ephemeral"}` when the provider supports it.
- **Pricing tables with cache_read/cache_write multipliers** —
  0.10× / 1.25× for Anthropic, 0.10× / 0.28× for Gemini (Gemini's
  cache write is cheaper than input — opposite of Anthropic).
- **`--preset-agent` CLI flag** — loads named presets of
  `system + tools + role` so the bench can simulate real production
  agent workloads.

## Commits

- `fc0e9fa` — `fix(bench): realistic preset pins role so cache key
  stays constant`
- `280b82a` — `chore(repo): preserve bench diagnostic scripts under
  scripts/diagnostics/`

## Lesson (one line)

> When you say "this should be cached", check that *every* sub-token of
> the cacheable prefix is actually constant across calls. Sub-section
> randomization is invisible at the prompt level but lethal at the
> cache-key level.
