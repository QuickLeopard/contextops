"""Diagnose: what does the raw response usage look like for cache hit vs miss?"""
import os
import sys
import json
import urllib.request

key = os.environ.get("ZEN_API_KEY", "")
if not key:
    print("ERROR: no Zen key")
    sys.exit(1)

SYSTEM = "You are a helpful assistant. " * 200  # ~1500 tokens, below 1024 minimum
url = "https://opencode.ai/zen/v1/messages"

# Make 3 sequential calls with the same system, see how usage changes
for i in range(1, 4):
    payload = {
        "model": "claude-sonnet-4-6",
        "system": [
            {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": f"Call {i}: just say ok."}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "User-Agent": "opencode-cli/0.5.0",
        },
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print(f"=== call {i} ===")
    print(f"  usage: {json.dumps(resp.get('usage', {}), indent=4)}")
    print(f"  raw keys: {sorted(resp.keys())}")
    print()
