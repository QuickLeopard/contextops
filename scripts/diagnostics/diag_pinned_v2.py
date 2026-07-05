"""Diagnose v2: use the FULL preset content (system + tools ≈ 1182 tokens) and 5 sequential calls.

The previous diagnostic only used the 498-token system content — under the
1024-token cache minimum. This one uses the full preset to match what the bench
sends. If the cache activates here but not in the bench, there's a bench bug.
"""
import os
import sys
import json
import urllib.request

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

# Import the preset from the bench package
sys.path.insert(0, "/Volumes/My Data/Work/Minimax Code/contextops")
from contextops_bench.prompt_factory import AGENT_PRESETS  # noqa: E402

SYSTEM = AGENT_PRESETS["realistic"]["system"] + "\n\n" + AGENT_PRESETS["realistic"]["tools"]

url = "https://openrouter.ai/api/v1/chat/completions"

print(f"SYSTEM: {len(SYSTEM)} chars (rough token estimate: {len(SYSTEM)//4})")
print(f"Calling 5 sequential requests, provider pinned to anthropic, cache_control on system...\n")

for i in range(1, 6):
    payload = {
        "model": "anthropic/claude-haiku-4.5",
        "provider": {"order": ["anthropic"], "allow_fallbacks": False},
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": f"What is {i} + {i}? Just the number."},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://contextops-bench.local",
        },
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    usage = resp.get("usage", {})
    ptd = usage.get("prompt_tokens_details", {})
    print(f"call {i}: prompt={usage.get('prompt_tokens'):>5}  "
          f"cached={ptd.get('cached_tokens', 0):>5}  "
          f"cache_write={ptd.get('cache_write_tokens', 0):>5}  "
          f"provider={resp.get('provider')}")
