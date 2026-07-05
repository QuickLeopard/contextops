"""Mimic the bench's request format exactly — same payload shape the ZenDirectClient
sends in clients.py. Use the same system size as the realistic preset (~3476 tokens)
and call 5 times sequentially. Print cache_read vs cache_creation each call.
"""
import json
import os
import urllib.request
import hashlib

key = os.environ.get("ZEN_API_KEY", "")
if not key:
    raise SystemExit("Set ZEN_API_KEY first")

# Roughly 3476 tokens of system content (matches realistic preset)
SYSTEM = ("You are a helpful assistant. " * 50 + "\n\n" +
          "Describe the user's situation and provide actionable advice. " * 60 + "\n\n" +
          "Use markdown headings, bullet lists, and code blocks where appropriate. " * 40 + "\n\n" +
          "Tool usage policy: prefer minimal tool calls, ask before destructive ops. " * 30)
sys_tokens = len(SYSTEM) // 4
print(f"system chars={len(SYSTEM)}  est_tokens={sys_tokens}\n", flush=True)

url = "https://opencode.ai/zen/v1/messages"

for i in range(1, 6):
    payload = {
        "model": "claude-sonnet-4-6",
        "system": [{"type": "text", "text": SYSTEM,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user",
                      "content": f"Call {i}: summarize what you are in one sentence."}],
        "max_tokens": 32,
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
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
    u = resp.get("usage", {})
    cr = u.get("cache_read_input_tokens", 0)
    cc = u.get("cache_creation_input_tokens", 0)
    inp = u.get("input_tokens", 0)
    out = u.get("output_tokens", 0)
    verdict = ("READ ✅" if cr > 0 else
               "WRITE ➕" if cc > 0 else "MISS ❌")
    print(f"call {i}  in={inp:>5}  out={out:>4}  cache_read={cr:>5}  cache_creation={cc:>5}  {verdict}",
          flush=True)
