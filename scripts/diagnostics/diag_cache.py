"""Diagnose: what cache-related fields does OpenRouter surface for gpt-4o-mini?

Run 3 sequential calls with the same long prefix and print the full usage
block + any cache-related fields from each response. Tells us:
- Whether OpenRouter surfaces cache_read_input_tokens at all
- What the field is called (cached_tokens? cache_read_input_tokens? prompt_tokens_details.cached_tokens?)
- Whether the cache is actually hitting (compare call 1 vs call 2 vs call 3)
"""
import os
import sys
import json
import urllib.request

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

# ~600-token stable prefix. Combined with the user message, total > 1024 tokens.
LONG_PREFIX = "You are a careful, thorough assistant who double-checks every claim. " * 50

url = "https://openrouter.ai/api/v1/chat/completions"

print(f"Calling 3 sequential requests with {len(LONG_PREFIX)}-char prefix...\n")

for i in range(1, 4):
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": LONG_PREFIX},
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
        },
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())

    print(f"=== call {i} ===")
    print(f"  model: {resp.get('model')}")
    print(f"  id:    {resp.get('id')}")
    print(f"  usage: {json.dumps(resp.get('usage', {}), indent=4)}")
    # Print all top-level keys in case there's a cache field outside 'usage'
    print(f"  all top-level keys: {sorted(resp.keys())}")
    print()
