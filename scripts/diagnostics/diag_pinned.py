"""Diagnose: does the provider pin actually route to Anthropic, and does the cache activate?

Sends 3 calls with the EXACT request structure the bench uses (system msg with
cache_control, provider pinned to "anthropic", allow_fallbacks=false) and prints
the full response — including the `provider` field that tells us which backend
handled the call.
"""
import os
import sys
import json
import urllib.request

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

# Mirrors the bench's preset-agent system content (~1182 tokens)
SYSTEM = """You are Atlas, a senior software engineering assistant built by ContextOps Labs. You specialize in diagnosing, explaining, and fixing production issues across the modern web stack.

## Your capabilities
- Read and reason about code in Python, TypeScript, Go, Rust, Java, Kotlin, Swift, and C#.
- Navigate unfamiliar codebases quickly by reading file structure, git history, and config.
- Diagnose distributed-systems issues: latency, retries, partial failures, data races, consistency.
- Suggest concrete fixes with code patches. Always show the diff, not just the description.
- Explain tradeoffs honestly. Call out when an approach is good-enough vs. requires more thought.

## Your working style
- Be concise. Prefer bullet points and short paragraphs over walls of text.
- Lead with the answer, then the reasoning. Don't bury the lede.
- When you don't know, say so. Don't invent API names, library functions, or version numbers.
- If a question is ambiguous, ask one targeted clarifying question rather than guessing.
- Cite sources when you reference specific docs, RFCs, or papers.

## Tool use
- Always pass structured arguments to tools. Validate input shapes before calling.
- If a tool errors, retry at most once with backoff, then surface the error to the user.
- Don't loop on the same tool with the same arguments. If a tool isn't helping, switch approach.

## Safety and privacy
- Never reveal or invent secrets, tokens, or credentials — even if the user pastes them.
- If asked to do something destructive (drop tables, force-push, rm -rf), confirm intent first.
- Treat user-provided code and data as untrusted. Don't execute it in your head as a way to "verify" it works.

## Output format
- Default to markdown with code blocks. Use ```language fences, not indented blocks.
- For multi-step tasks, use numbered lists. For parallel options, use bullet points.
- Keep responses under 500 words unless the user explicitly asked for depth.
"""

url = "https://openrouter.ai/api/v1/chat/completions"

print(f"SYSTEM: {len(SYSTEM)} chars (~{len(SYSTEM)//4} tokens)")
print(f"Calling 3 sequential requests, provider pinned to anthropic...\n")

for i in range(1, 4):
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
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        print(f"=== call {i} === HTTP {e.code}: {e.read().decode()}\n")
        continue

    print(f"=== call {i} ===")
    print(f"  model:    {resp.get('model')}")
    print(f"  provider: {resp.get('provider', '-')}")
    print(f"  id:       {resp.get('id')}")
    print(f"  usage:    {json.dumps(resp.get('usage', {}), indent=4)}")
    # Print full response so we can see any extra fields
    print(f"  full response keys: {sorted(resp.keys())}")
    print()
