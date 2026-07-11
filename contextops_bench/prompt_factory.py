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
from pathlib import Path
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


def _gen_random_text(rng: random.Random, min_words: int = 5, max_words: int = 30) -> str:
    """Generate a random sentence of plausible-looking words."""
    n = rng.randint(min_words, max_words)
    words = ["alpha", "beta", "gamma", "delta", "vector", "matrix", "tensor",
             "graph", "node", "edge", "cache", "queue", "stream", "batch",
             "kernel", "module", "schema", "index", "query", "result",
             "lorem", "ipsum", "dolor", "amet", "consectetur", "adipiscing"]
    return " ".join(rng.choices(words, k=n)) + "."


def _gen_documents(rng: random.Random, min_chars: int = 100, max_chars: int = 2000) -> str:
    """Generate a long document blob."""
    n_chars = rng.randint(min_chars, max_chars)
    blob = []
    while sum(len(s) for s in blob) < n_chars:
        blob.append(_gen_random_text(rng, 20, 60))
    return "\n\n".join(blob)[:n_chars]


def _gen_history(rng: random.Random, min_turns: int = 1, max_turns: int = 5) -> list[dict]:
    """Generate a chat history."""
    n = rng.randint(min_turns, max_turns)
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append({"role": "user", "content": _gen_random_text(rng, 3, 12)})
        else:
            out.append({"role": "assistant", "content": _gen_random_text(rng, 5, 20)})
    return out


def _gen_context(rng: random.Random) -> str:
    return _gen_random_text(rng, 50, 200)


def generate_one(
    seed: int | None = None,
    *,
    fixed_system: str | None = None,
    fixed_tools: str | None = None,
    fixed_model: str | None = None,
) -> Prompt:
    """Generate a single random prompt.

    `fixed_system` / `fixed_tools` lock the system/tools section across all generated
    prompts — this simulates a real deployment where many requests share the same
    agent definition (and therefore benefit from cache).

    `fixed_model` overrides the per-prompt model. Without it, generate_one() picks
    a random model from `MODELS` (mostly cloud-only names) — which is the right
    behavior for the offline `smoke`/echo bench, but wrong for `local` runs that
    need every prompt to use a model that actually exists on Ollama/LM Studio.
    """
    # Use a LOCAL RNG so we never perturb the global `random` state (which other
    # code in the process may depend on). Explicit seed → reproducible output.
    rng = random.Random(seed) if seed is not None else random.Random()

    lang_roll = rng.random()
    if lang_roll < 0.7:
        system_default = rng.choice(SYSTEM_PROMPTS_EN)
        query = rng.choice(QUERIES_EN)
    elif lang_roll < 0.9:
        system_default = rng.choice(SYSTEM_PROMPTS_RU)
        query = rng.choice(QUERIES_RU)
    else:
        system_default = rng.choice(SYSTEM_PROMPTS_ZH)
        query = rng.choice(QUERIES_ZH)

    if rng.random() < 0.3:
        query = rng.choice(QUERIES_CODE)

    return Prompt(
        system=fixed_system if fixed_system is not None else system_default,
        tools=fixed_tools if fixed_tools is not None else rng.choice(TOOL_DEFS),
        role=rng.choice(ROLE_PROMPTS) if rng.random() < 0.6 else "",
        context=_gen_context(rng) if rng.random() < 0.5 else "",
        documents=_gen_documents(rng) if rng.random() < 0.6 else "",
        history=_gen_history(rng) if rng.random() < 0.5 else [],
        query=query,
        model=fixed_model if fixed_model is not None else rng.choice(MODELS),
        goal=rng.choice(GOALS),
    )


def generate_many(
    n: int = 1000,
    seed: int = 42,
    *,
    fixed_system: str | None = None,
    fixed_tools: str | None = None,
    fixed_model: str | None = None,
) -> Iterator[Prompt]:
    """Yield `n` random prompts. Reproducible via `seed`.

    Pass `fixed_system` / `fixed_tools` to simulate a real workload where many
    requests share the same agent definition (cache hit rate should grow across
    the batch).

    Pass `fixed_model` to lock the model name across all generated prompts
    (required for `local` runs against Ollama/LM Studio — the random default
    pool is mostly cloud-only names).
    """
    for i in range(n):
        yield generate_one(
            seed=seed + i,
            fixed_system=fixed_system,
            fixed_tools=fixed_tools,
            fixed_model=fixed_model,
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


# Realistic agent preset content lives in `data/` so it doesn't dominate this
# module. Long enough to exceed OpenAI's 1024-token auto-cache threshold and
# Anthropic's 1024/2048-token explicit cache minimums (Sonnet/Opus vs Haiku).
# Total: ~6000+ tokens of stable system+tools content — a production agent whose
# system prompt + tool definitions stay fixed across many calls (the realistic
# scenario ContextOps exists to optimize).
_DATA = Path(__file__).parent / "data"
REALISTIC_AGENT_SYSTEM = (_DATA / "realistic_agent_system.md").read_text()
REALISTIC_AGENT_TOOLS = (_DATA / "realistic_agent_tools.json").read_text()


AGENT_PRESETS: dict[str, dict[str, str]] = {
    "realistic": {
        "system": REALISTIC_AGENT_SYSTEM,
        "tools": REALISTIC_AGENT_TOOLS,
    },
}