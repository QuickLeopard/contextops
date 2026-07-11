# Diagnostic cache-control probes

These are **scratch investigation scripts** — not part of the `contextops`
package and not imported by anything. They exist to probe live provider
behaviour that the cache-control feature in `contextops_bench` depends on.
Each hits the OpenRouter chat-completions endpoint directly with
`urllib.request` and prints the raw `usage` block from a few sequential calls.

All of them require `OPENROUTER_API_KEY` in the environment and make real
(billed) API calls. Run them from the repo root so the `sys.path` insertion
resolves:

```bash
export OPENROUTER_API_KEY=sk-...
python scripts/diag/diag_cache.py
```

| Script | What it probes |
|---|---|
| `diag_cache.py` | Which cache-related fields OpenRouter surfaces for an OpenAI model (`gpt-4o-mini`) — `cached_tokens`? `cache_read_input_tokens`? `prompt_tokens_details.cached_tokens`? |
| `diag_anthropic.py` | Whether Anthropic prompt caching activates through OpenRouter when a `cache_control: {type: "ephemeral"}` marker is placed on the system message. |
| `diag_pinned.py` | Whether provider pinning (`provider.order: ["anthropic"]`, `allow_fallbacks: false`) actually routes to Anthropic direct (not Bedrock/Vertex), and whether the cache activates under that routing. |
| `diag_pinned_v2.py` | Same as `diag_pinned.py` but uses the full realistic-agent preset (system + tools ≈ 1182 tokens) so the request clears Anthropic's 1024-token cache minimum. 5 sequential calls. |

## Why they still exist

The durable behaviour they probe should eventually be captured as the
network-gated `tests/test_anthropic_integration.py` (planned). Until that test
exists, these scripts are the fastest way to answer "is the cache actually
activating?" when something looks wrong in bench results.
