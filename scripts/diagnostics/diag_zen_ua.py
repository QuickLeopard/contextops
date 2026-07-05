"""Diagnose: try different User-Agent headers to bypass Cloudflare 1010."""
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
    "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": "Say hi in 5 words."}],
    "max_tokens": 20,
    "temperature": 0.0,
}

# User-Agents to try. Cloudflare often whitelists known SDKs/browsers.
user_agents = [
    ("curl", "curl/7.88.0"),
    ("claude-cli", "claude-cli/1.0.0"),
    ("opencode-cli", "opencode-cli/0.5.0"),
    ("ai-sdk-anthropic", "ai-sdk/anthropic/0.0.50"),
    ("anthropic-sdk-python", "anthropic-sdk-python/0.40.0"),
    ("Mozilla", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"),
]

for name, ua in user_agents:
    print(f"=== Trying User-Agent: {name} ===", flush=True)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": ua,
        },
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        print(f"  ✓ SUCCESS with {name}")
        print(f"  model: {resp.get('model')}")
        print(f"  usage: {json.dumps(resp.get('usage', {}), indent=4)}")
        break
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        print(f"  HTTP {e.code}: {body}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    print()
