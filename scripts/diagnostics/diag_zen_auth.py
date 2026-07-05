"""Diagnose: try the Anthropic-native auth style (x-api-key + anthropic-version)."""
import os
import sys
import json
import urllib.request

key = os.environ.get("ZEN_API_KEY", "")
if not key:
    print("ERROR: no Zen key")
    sys.exit(1)

SYSTEM = "You are a helpful assistant. " * 200
url = "https://opencode.ai/zen/v1/messages"
payload = {
    "model": "claude-sonnet-4-6",
    "system": SYSTEM,
    "messages": [{"role": "user", "content": "Say hi in 5 words."}],
    "max_tokens": 20,
    "temperature": 0.0,
}

# Try every plausible auth header combination.
combos = [
    ("x-api-key (lowercase) only", {"x-api-key": key}),
    ("x-api-key + anthropic-version", {"x-api-key": key, "anthropic-version": "2023-06-01"}),
    ("X-API-Key (capitalized)", {"X-API-Key": key}),
    ("anthropic-api-key", {"anthropic-api-key": key}),
    ("X-Api-Key + anthropic-version", {"X-Api-Key": key, "anthropic-version": "2023-06-01"}),
    ("anthropic_auth (snake)", {"anthropic_auth": key}),
]

for name, headers_extras in combos:
    print(f"=== Trying: {name} ===", flush=True)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "opencode-cli/0.5.0",  # the one that bypassed Cloudflare
            **headers_extras,
        },
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        print(f"  SUCCESS with {name}")
        print(f"  model: {resp.get('model')}")
        print(f"  usage: {json.dumps(resp.get('usage', {}), indent=4)}")
        break
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        print(f"  HTTP {e.code}: {body}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    print()
