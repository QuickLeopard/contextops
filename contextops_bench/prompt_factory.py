"""Generate >1000 diverse prompts for stress testing.

The goal is to exercise every code path in ContextOps:
- Tiny, small, medium, large, huge prompts
- Every section present / absent in different combinations
- Multiple languages (en, ru, zh, code)
- Edge cases: empty sections, very long sections, unicode, emojis
- Different model targets
- Different goals (cache_friendly, balanced, quality)
"""

from __future__ import annotations

import random
from typing import Iterator

from contextops.models import Prompt, HistoryMessage

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4.6",
    "claude-haiku-4.5",
    "qwen3-30b",
    "gigachat",
    "yandexgpt",
    "mistral-large-2",
]

GOALS = ["cache_friendly", "balanced", "quality"]

SYSTEM_PROMPTS_EN = [
    "You are a helpful assistant.",
    "You are an expert Python developer. Write clean, idiomatic code.",
    "You are a senior DevOps engineer. Be concise and direct.",
    "You are a careful QA reviewer. Find bugs before they ship.",
    "You are a friendly customer support agent. Be polite and empathetic.",
    "You are a financial analyst. Cite sources when possible.",
    "You are a security researcher. Think adversarially.",
    "You are a translator. Preserve meaning, not literal words.",
    "You are a research assistant. Always cite the underlying facts.",
    "You are a code reviewer. Focus on correctness, readability, and security.",
]

SYSTEM_PROMPTS_RU = [
    "Ты — полезный ассистент.",
    "Ты — старший разработчик на Python. Пиши чистый, идиоматичный код.",
    "Ты — DevOps инженер. Будь кратким и прямым.",
    "Ты — внимательный тестировщик. Находи баги до релиза.",
]

SYSTEM_PROMPTS_ZH = [
    "你是一个有帮助的助手。",
    "你是一个高级 Python 开发人员。",
]

TOOL_DEFS = [
    '[{"name": "search", "parameters": {"q": "string"}}, {"name": "calc", "parameters": {"expr": "string"}}]',
    '[{"name": "get_weather", "parameters": {"city": "string"}}, {"name": "send_email", "parameters": {"to": "string", "body": "string"}}]',
    '[{"name": "db_query", "parameters": {"sql": "string"}}, {"name": "db_schema", "parameters": {"table": "string"}}]',
    '[{"name": "code_exec", "parameters": {"language": "string", "code": "string"}}]',
    "[]",  # no tools
]

ROLE_PROMPTS = [
    "weather-agent",
    "code-assistant",
    "data-analyst",
    "support-bot",
    "translator",
    "",
]

QUERIES_EN = [
    "What's the weather in Berlin?",
    "Refactor this Python function to use list comprehension.",
    "Why is my Kafka consumer lag spiking?",
    "Translate this sentence to French: 'Good morning, how are you?'",
    "Summarize the attached document in 3 bullets.",
    "Find the SQL query that joins customers and orders.",
    "What is the boiling point of water at sea level?",
    "Explain the CAP theorem in one paragraph.",
    "List the top 5 vulnerabilities in this OWASP top 10 list.",
    "Write a unit test for this function.",
]

QUERIES_RU = [
    "Какая погода в Москве?",
    "Переведи на английский: 'Доброе утро, как дела?'",
    "Что такое CAP-теорема?",
    "Напиши функцию сортировки пузырьком на Python.",
]

QUERIES_ZH = [
    "北京今天天气怎么样？",
    "用 Python 写一个快速排序。",
]

QUERIES_CODE = [
    "def fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)",
    "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id;",
    "kubectl get pods -n kube-system --field-selector=status.phase!=Running",
]


def _gen_random_text(min_words: int = 5, max_words: int = 30) -> str:
    """Generate a random sentence of plausible-looking words."""
    n = random.randint(min_words, max_words)
    words = ["alpha", "beta", "gamma", "delta", "vector", "matrix", "tensor",
             "graph", "node", "edge", "cache", "queue", "stream", "batch",
             "kernel", "module", "schema", "index", "query", "result",
             "lorem", "ipsum", "dolor", "amet", "consectetur", "adipiscing"]
    return " ".join(random.choices(words, k=n)) + "."


def _gen_documents(min_chars: int = 100, max_chars: int = 2000) -> str:
    """Generate a long document blob."""
    n_chars = random.randint(min_chars, max_chars)
    blob = []
    while sum(len(s) for s in blob) < n_chars:
        blob.append(_gen_random_text(20, 60))
    return "\n\n".join(blob)[:n_chars]


def _gen_history(min_turns: int = 1, max_turns: int = 5) -> list[dict]:
    """Generate a chat history."""
    n = random.randint(min_turns, max_turns)
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append({"role": "user", "content": _gen_random_text(3, 12)})
        else:
            out.append({"role": "assistant", "content": _gen_random_text(5, 20)})
    return out


def _gen_context() -> str:
    return _gen_random_text(50, 200)


def _maybe(generator_fn, p: float = 0.7):
    """50/50: include this section or not."""
    return generator_fn() if random.random() < p else type(generator_fn())()


def generate_one(
    seed: int | None = None,
    *,
    fixed_system: str | None = None,
    fixed_tools: str | None = None,
    fixed_model: str | None = None,
    fixed_role: str | None = None,
) -> Prompt:
    """Generate a single random prompt.

    `fixed_system` / `fixed_tools` / `fixed_role` lock those sections across all
    generated prompts — this simulates a real deployment where many requests share
    the same agent definition (and therefore benefit from cache). All three are
    part of the cacheable prefix; leaving any of them random rotates the cache key
    and silently turns every call into a cold write.

    `fixed_model` overrides the per-prompt model. Without it, generate_one() picks
    a random model from `MODELS` (mostly cloud-only names) — which is the right
    behavior for the offline `smoke`/echo bench, but wrong for `local` runs that
    need every prompt to use a model that actually exists on Ollama/LM Studio.
    """
    if seed is not None:
        random.seed(seed)

    lang_roll = random.random()
    if lang_roll < 0.7:
        system_default = random.choice(SYSTEM_PROMPTS_EN)
        query = random.choice(QUERIES_EN)
    elif lang_roll < 0.9:
        system_default = random.choice(SYSTEM_PROMPTS_RU)
        query = random.choice(QUERIES_RU)
    else:
        system_default = random.choice(SYSTEM_PROMPTS_ZH)
        query = random.choice(QUERIES_ZH)

    if random.random() < 0.3:
        query = random.choice(QUERIES_CODE)

    return Prompt(
        system=fixed_system if fixed_system is not None else system_default,
        tools=fixed_tools if fixed_tools is not None else random.choice(TOOL_DEFS),
        role=fixed_role if fixed_role is not None else (random.choice(ROLE_PROMPTS) if random.random() < 0.6 else ""),
        context=_gen_context() if random.random() < 0.5 else "",
        documents=_gen_documents() if random.random() < 0.6 else "",
        history=_gen_history() if random.random() < 0.5 else [],
        query=query,
        model=fixed_model if fixed_model is not None else random.choice(MODELS),
        goal=random.choice(GOALS),
    )


def generate_many(
    n: int = 1000,
    seed: int = 42,
    *,
    fixed_system: str | None = None,
    fixed_tools: str | None = None,
    fixed_model: str | None = None,
    fixed_role: str | None = None,
) -> Iterator[Prompt]:
    """Yield `n` random prompts. Reproducible via `seed`.

    Pass `fixed_system` / `fixed_tools` / `fixed_role` to simulate a real workload
    where many requests share the same agent definition (cache hit rate should
    grow across the batch).

    Pass `fixed_model` to lock the model name across all generated prompts
    (required for `local` runs against Ollama/LM Studio — the random default
    pool is mostly cloud-only names).
    """
    random.seed(seed)
    for i in range(n):
        yield generate_one(
            seed=seed + i,
            fixed_system=fixed_system,
            fixed_tools=fixed_tools,
            fixed_model=fixed_model,
            fixed_role=fixed_role,
        )


# Pre-built deterministic edge cases — these MUST be exercised in any smoke test.
EDGE_CASES: list[Prompt] = [
    Prompt(),  # completely empty
    Prompt(query="just a query"),  # only variable part
    Prompt(system="only system"),  # only stable part
    Prompt(system="S", tools="T", role="R", context="C", documents="D",
           history=[{"role": "user", "content": "hi"}], query="Q"),
    Prompt(system="русский: 你好 🌍", query="emoji ❓"),
    Prompt(query="x" * 50_000),  # huge query
    Prompt(documents="x" * 100_000),  # huge documents
    Prompt(system="S", goal="quality"),  # quality mode
    Prompt(history=[HistoryMessage(role="user", content="str"), "bare string"]),
    Prompt(system="A\nB\nC", tools="D\nE\nF", documents="\n\n\n"),  # weird whitespace
]


# Realistic agent preset — long enough to exceed:
# - OpenAI's 1024-token auto-cache threshold
# - Anthropic Sonnet/Opus 1024-token explicit cache minimum
# - Anthropic Haiku 2048-token explicit cache minimum (HIGHER than Sonnet)
#
# Total target: ~6000+ tokens of stable system+tools content, simulating
# a production agent whose system prompt + tool definitions stay fixed
# across many calls. This is the realistic scenario ContextOps exists to
# optimize — agents with big tool schemas and detailed system prompts.
REALISTIC_AGENT_SYSTEM = """You are Atlas, a senior software engineering assistant built by ContextOps Labs. You specialize in diagnosing, explaining, and fixing production issues across the modern web stack. You have been deployed in production at hundreds of teams and have a reputation for being both correct and concise.

## Your capabilities
- Read and reason about code in Python, TypeScript, Go, Rust, Java, Kotlin, Swift, and C# — including their respective package ecosystems (pip/npm, cargo, maven/gradle, swiftpm, etc).
- Navigate unfamiliar codebases quickly by reading file structure, git history, configuration, and deployment manifests. You can orient in a 500k-line monorepo in under five tool calls.
- Diagnose distributed-systems issues: latency spikes, retry storms, partial failures, data races, consistency anomalies, cache stampedes, thundering herds, leader election flapping, queue backpressure, slow consumers.
- Suggest concrete fixes with code patches. Always show the diff, not just the description. A patch is worth a thousand paragraphs of explanation.
- Explain tradeoffs honestly. Call out when an approach is good-enough vs. requires more thought. Distinguish between "this works in the happy path" and "this is robust."

## Your working style
- Be concise. Prefer bullet points and short paragraphs over walls of text. If a response exceeds 500 words and the user didn't ask for depth, you are being too verbose — tighten it.
- Lead with the answer, then the reasoning. Don't bury the lede. The first sentence should be the conclusion if there is one.
- When you don't know, say so. Don't invent API names, library functions, version numbers, or RFC section numbers. If you're uncertain, hedge with "I think" or "verify against the official docs" — never make up specifics.
- If a question is ambiguous, ask ONE targeted clarifying question rather than guessing wrong on five dimensions at once.
- Cite sources when you reference specific docs, RFCs, papers, or known CVEs. Drop a URL or doc title; don't leave the user to hunt.
- Never say "as an AI" or "as a language model" — those phrases are noise. Just answer the question.

## Tool use discipline
- Always pass structured arguments to tools. Validate input shapes before calling. Don't pass null where a string is required, don't pass strings where enums are expected.
- If a tool errors, retry at most ONCE with backoff. After that, surface the error to the user — don't loop.
- Don't call the same tool with the same arguments more than twice. If a tool isn't helping, switch approach or ask the user.
- Read before you write. Use `search_code` or `list_directory` before `read_file` to orient. Use `read_file` before `edit_file` to understand the surrounding context.
- Prefer small, focused patches over large rewrites. A 5-line patch is reviewable; a 500-line patch is not.
- When in doubt, don't. If the user pastes destructive-looking code, ask before executing.

## Safety and privacy
- Never reveal or invent secrets, tokens, or credentials — even if the user pastes them. Don't echo them back, don't repeat them, don't transform them. Treat them as toxic.
- If asked to do something destructive (drop tables, force-push, rm -rf, kill processes, format disks), confirm intent first. State exactly what will happen and ask for explicit go-ahead.
- Treat user-provided code and data as untrusted. Don't execute it in your head as a way to "verify" it works — actually run it in a sandbox if needed.
- Don't help with exploits, malware, credential stuffing, prompt injection payloads, or other dual-use attacks. If the user is clearly doing red-team work in a sanctioned context, help them; if not, decline.

## Output format
- Default to markdown with code blocks. Use triple-backtick fences with a language tag, not indented blocks.
- For multi-step tasks, use numbered lists. For parallel options, use bullet points. For sequential dependencies, use arrows (->).
- For code changes, lead with a short summary, then the diff, then any caveats.
- For diagnoses, lead with the root cause (one sentence), then the evidence (2-5 bullets), then the fix (numbered steps).
- Keep responses under 500 words unless the user explicitly asked for depth. If you need to go longer, say so and offer to split.

## Project context
- This is a production codebase with real users. Stability matters more than cleverness. Choose boring technology over novel technology when both work.
- Tests are not optional. If you suggest a change, suggest a test for it. If the change is hard to test, say why and propose the closest practical test.
- Performance work should be measured, not guessed. If you suggest an optimization, suggest a benchmark for it. Don't make claims like "this will be 10x faster" without numbers.
- Backwards compatibility matters. If you change a public API or schema, call out the migration path.

## Common patterns you should recognize
- "Fix this bug" → diagnose root cause → propose minimal patch → write test → verify
- "Add a feature" → check existing patterns → propose API shape → implement → test → docs
- "Refactor this" → identify the smell → propose target shape → mechanical transformation → verify no behavior change
- "Why is this slow?" → form a hypothesis → measure → confirm or revise → propose fix → measure again
- "How does X work?" → trace the code path → explain in 1-2 paragraphs → point to the key lines

## Common anti-patterns to flag
- Catching broad `Exception` and swallowing it
- Mutable default arguments in Python (`def f(x=[])`)
- `SELECT *` in production queries
- N+1 query patterns in ORM code
- Unbounded retries without backoff
- Secrets in source control
- Missing or overly-broad CSP/CORS headers
- Synchronous I/O in async paths
- `ThreadPoolExecutor` without a bounded queue in hot paths

## Response style examples
GOOD: "The bug is a missing `await` on line 42. The function returns a coroutine, not the result, so the test asserts on a coroutine object that always passes. Fix: add `await`. Add a test that calls `await foo()` and checks the return type."

BAD: "I think there might be an issue with the async/await pattern in the function. You may want to consider whether the function is properly awaited. It's hard to say without more context but you could try adding an await keyword and see if that helps. Let me know if you have other questions!"

The good version is concrete, identifies the line, explains the mechanism, and proposes a fix. The bad version is vague, hedged, and adds no information. Always be the good version.
"""

REALISTIC_AGENT_TOOLS = """[
  {
    "name": "search_code",
    "description": "Search a code repository for files matching a query. Returns file paths with line numbers and a short snippet of the match. Use this before reading a file directly — it's cheaper and gives you orientation. Supports regex. If a search returns too many results, narrow with `path` or use `max_results` to cap. Search is case-insensitive by default; pass `case_sensitive: true` to disable. Results are returned in git-relevance order (most-touched files first), not lexical order.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query. Supports regex (RE2 syntax)."},
        "path": {"type": "string", "description": "Subdirectory to scope the search, relative to repo root. Default: repo root."},
        "max_results": {"type": "integer", "description": "Max files to return. Default 20, max 100."},
        "case_sensitive": {"type": "boolean", "description": "Disable case-insensitive matching. Default false."},
        "include_globs": {"type": "array", "items": {"type": "string"}, "description": "Only search files matching these globs (e.g. ['*.py', '*.ts'])."},
        "exclude_globs": {"type": "array", "items": {"type": "string"}, "description": "Skip files matching these globs (e.g. ['node_modules/*', 'dist/*'])."}
      },
      "required": ["query"]
    }
  },
  {
    "name": "read_file",
    "description": "Read a file from the repository. Returns the file contents with line numbers (cat -n style). For large files, prefer reading specific line ranges via start_line / end_line to save context. Reading more than 500 lines in one call is usually a sign you should narrow your scope. Files are returned as UTF-8; binary files are not supported via this tool — use a different mechanism for binaries.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Path relative to repo root."},
        "start_line": {"type": "integer", "description": "1-indexed start line. Default 1."},
        "end_line": {"type": "integer", "description": "1-indexed end line (inclusive). Default: end of file."},
        "max_lines": {"type": "integer", "description": "Cap on lines returned, regardless of end_line. Default 500."}
      },
      "required": ["path"]
    }
  },
  {
    "name": "run_command",
    "description": "Execute a shell command in the repository root. Returns stdout, stderr, and exit code as a single response. 30-second default timeout. Use for tests, builds, linters, type checkers, package install — not for arbitrary code execution or interactive commands. The command must be non-interactive (no TTY prompts, no `vim`, no `less`, no `ssh`). Commands that take longer than 30s should be split or use a longer timeout_seconds override.",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {"type": "string", "description": "The command to run. Must be non-interactive."},
        "timeout_seconds": {"type": "integer", "description": "Override default 30s timeout. Max 300s."},
        "working_dir": {"type": "string", "description": "Override the working directory. Default: repo root."},
        "env": {"type": "object", "description": "Additional environment variables, e.g. {\"NODE_ENV\": \"test\"}. Merged with the default env."}
      },
      "required": ["command"]
    }
  },
  {
    "name": "edit_file",
    "description": "Apply a unified-diff patch to a file. The patch must apply cleanly against the current file contents. Returns success or a unified-diff error showing the mismatch (hunk failed, line numbers off, etc). Prefer small, focused patches over large rewrites. A patch that touches more than 50 lines is probably too big — split it. Always include 3 lines of context above and below each hunk to make application robust. If the file doesn't exist, the patch must include `new file mode` and the full content.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Path relative to repo root."},
        "patch": {"type": "string", "description": "Unified diff format. Must be a complete diff (with `--- a/path` and `+++ b/path` headers)."},
        "explanation": {"type": "string", "description": "One-sentence summary of the change. Shown in the change log."},
        "dry_run": {"type": "boolean", "description": "If true, validate the patch but don't write. Default false."}
      },
      "required": ["path", "patch"]
    }
  },
  {
    "name": "list_directory",
    "description": "List the contents of a directory. Returns a tree-like view of files and subdirectories, two levels deep. Use to orient before reading specific files. Hidden files (starting with .) are excluded by default. Directories are listed first, then files, both sorted alphabetically. Sizes are not shown — use run_command with `du -sh` if you need sizes.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Directory path, relative to repo root. Default: repo root."},
        "max_depth": {"type": "integer", "description": "How deep to recurse. Default 2, max 5."},
        "include_hidden": {"type": "boolean", "description": "Include files starting with `.`. Default false."}
      },
      "required": ["path"]
    }
  },
  {
    "name": "git_log",
    "description": "Show recent git commits. Returns commit hash, author, timestamp, and message. Use to understand who touched what recently and why. The default format is one-line. For deeper inspection, use `git_show` (separate tool). Useful for diagnosing regressions: 'when did this file last change' and 'what's the history of this function'.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "Filter to commits touching this path. Default: all commits."},
        "max_commits": {"type": "integer", "description": "How many recent commits to show. Default 20, max 100."},
        "author": {"type": "string", "description": "Filter by author (name or email substring)."},
        "since": {"type": "string", "description": "ISO date or relative (e.g. '2 weeks ago'). Default: last 90 days."}
      }
    }
  },
  {
    "name": "git_show",
    "description": "Show a specific commit's diff. Returns the full patch and metadata for one commit. Use after git_log to inspect a suspicious commit. Pair with the commit hash from git_log.",
    "parameters": {
      "type": "object",
      "properties": {
        "commit": {"type": "string", "description": "Commit hash (full or short). Required."},
        "path": {"type": "string", "description": "Restrict the diff to this path. Default: all paths in the commit."}
      },
      "required": ["commit"]
    }
  }
]"""


AGENT_PRESETS: dict[str, dict[str, str]] = {
    "realistic": {
        "system": REALISTIC_AGENT_SYSTEM,
        "tools": REALISTIC_AGENT_TOOLS,
        # Identity is part of the agent definition — must be constant for cache
        # to be hit across calls. Randomizing role rotates the cache key and
        # silently turns every call into a cold cache_write.
        "role": "code-assistant",
    },
}