"""Diagnose Anthropic cache_control via OpenRouter.

Mirrors what the bench does for the OPTIMIZED arm:
  - system message with cache_control: {type: "ephemeral"}
  - user message with variable content

Sends 3 sequential requests with the same system content and different
user content, prints the full usage block each time.
"""
import os
import sys
import json
import urllib.request

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

# ~1100 token stable prefix (system prompt + tools combined as a single string)
SYSTEM = (
    "You are Atlas, a senior software engineering assistant. "
    "You specialize in diagnosing production issues. "
    "Be concise. Lead with the answer, then the reasoning. "
    "Use markdown with code blocks. "
    "When you don't know, say so. Don't invent API names or version numbers. "
) * 30  # ~1500 chars * 30 = ~45,000 chars ≈ ~11,000 tokens, way above 1024

url = "https://openrouter.ai/api/v1/chat/completions"

print(f"SYSTEM prefix: {len(SYSTEM)} chars (~{len(SYSTEM)//4} tokens)")
print(f"Calling 3 sequential requests with cache_control marker on system message...\n")

for i in range(1, 4):
    payload = {
        "model": "anthropic/claude-haiku-4.5",
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
            # Force OpenRouter to use Anthropic (not a fallback)
            "HTTP-Referer": "https://contextops-bench.local",
        },
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())

    print(f"=== call {i} ===")
    print(f"  model:    {resp.get('model')}")
    print(f"  id:       {resp.get('id')}")
    print(f"  provider: {resp.get('provider', '-')}")
    usage = resp.get("usage", {})
    print(f"  usage:    {json.dumps(usage, indent=4)}")
    print()
