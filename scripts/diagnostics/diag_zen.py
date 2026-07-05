"""Diagnose: capture the FULL Zen error body to see why 403 is happening."""
import os
import sys
import json
import urllib.request

key = os.environ.get("ZEN_API_KEY") or os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY", "")
if not key:
    print("ERROR: no Zen key in env")
    sys.exit(1)

print(f"Using key starting with: {key[:8]}... (len={len(key)})")
print(f"Key env var: ZEN_API_KEY={'set' if os.environ.get('ZEN_API_KEY') else 'unset'}, "
      f"OPENCODE_API_KEY={'set' if os.environ.get('OPENCODE_API_KEY') else 'unset'}, "
      f"OPENCODE_ZEN_API_KEY={'set' if os.environ.get('OPENCODE_ZEN_API_KEY') else 'unset'}")
print()

SYSTEM = "You are a helpful assistant. " * 200  # ~2000 chars

url = "https://opencode.ai/zen/v1/messages"
payload = {
    "model": "claude-sonnet-4-6",
    "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
    "messages": [{"role": "user", "content": "Say hi in 5 words."}],
    "max_tokens": 20,
    "temperature": 0.0,
}

for auth_style in ("bearer", "x-api-key"):
    print(f"=== Trying auth style: {auth_style} ===")
    headers = {"Content-Type": "application/json"}
    if auth_style == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    else:
        headers["x-api-key"] = key
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST", headers=headers,
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        print(f"  SUCCESS! model={resp.get('model')}")
        print(f"  usage: {json.dumps(resp.get('usage', {}), indent=4)}")
        break
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"  HTTP {e.code}: {body[:500]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    print()
