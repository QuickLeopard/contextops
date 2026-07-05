# Diagnostic scripts

One-off scripts used to debug specific issues with the ContextOps bench
harness. Each script targets a single hypothesis and is meant to be run
manually, not as part of CI.

These were used to chase down the "realistic preset cache key regression"
behind commit `fc0e9fa`. They are preserved as-is for future reference —
they demonstrate the systematic debug process we used to nail the bug
(given the cache showed zero hits, work backwards through API auth, User-
Agent WAF gates, request body shape, prefix stability, response field
semantics, etc.).

## Scripts by goal

| Script | Goal |
| --- | --- |
| `diag_anthropic.py` | Verify Anthropic direct API surface area for cache fields |
| `diag_cache.py` | Pin down whether OpenRouter was the cache issue |
| `diag_pinned.py` | Test OpenRouter provider pinning effect on cache |
| `diag_pinned_v2.py` | Re-test pinning with explicit Anthropic-direct routing |
| `diag_zen.py` | Sanity-check OpenCode-ZEN endpoint and auth |
| `diag_zen_auth.py` | Diagnose ZEN auth header variants (Bearer vs x-api-key) |
| `diag_zen_ua.py` | Diagnose Cloudflare WAF 1010 on default Python UA |
| `diag_zen_usage.py` | Print Anthropic-native usage block across 3 sequential calls |
| `diag_zen_n5.py` | Reproduce n=5 bench scenario to confirm cache behavior at small N |

## When to add a new entry here

If you find yourself writing yet another ad-hoc debug script in the repo
root to answer a one-off hypothesis — keep it. Add it here with a one-line
goal comment so the next person debugging the same area doesn't have to
re-derive the diagnostic.
