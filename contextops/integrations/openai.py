"""One-line OpenAI SDK integration.

Wraps an `openai.OpenAI` client so every `chat.completions.create` call is
reordered for cache friendliness and optionally logged to the local SQLite
database.

Usage:
    import openai
    from contextops.integrations.openai import patch

    client = openai.OpenAI()
    patch(client)

    # Subsequent calls are automatically reordered and logged:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello!"},
            {"role": "system", "content": "You are helpful."},
        ],
    )

To unpatch:
    from contextops.integrations.openai import unpatch
    unpatch(client)
"""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextops.logger import Logger
from contextops.models import CallLog, HistoryMessage, Prompt
from contextops.optimizer import reorder

try:
    from openai.types.chat import ChatCompletion
except ImportError:  # pragma: no cover - optional dep
    ChatCompletion = Any  # type: ignore[misc,assignment]

try:
    import openai
except ImportError:  # pragma: no cover - optional dep
    openai = None  # type: ignore[assignment]

_patched: dict[int, tuple[Any, Any]] = {}


def _messages_to_prompt(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o",
) -> Prompt:
    """Convert an OpenAI messages list to a ContextOps Prompt."""
    system_parts: list[str] = []
    other_messages: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            system_parts.append(str(content))
        else:
            other_messages.append(m)

    system = "\n\n".join(system_parts)
    tools_str = json.dumps(tools) if tools else ""

    query = ""
    history: list[Any] = []
    if other_messages and other_messages[-1].get("role") == "user":
        query_content = other_messages[-1].get("content") or ""
        if isinstance(query_content, list):
            query_content = "\n".join(
                part.get("text", "") for part in query_content if isinstance(part, dict)
            )
        query = str(query_content)
        history = other_messages[:-1]
    else:
        history = other_messages

    return Prompt(
        system=system,
        tools=tools_str,
        history=history,
        query=query,
        model=model,
    )


def _prompt_to_messages(
    prompt: Prompt, original_kwargs: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert a ContextOps Prompt back to OpenAI messages + kwargs."""
    messages: list[dict[str, Any]] = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})

    for item in prompt.history:
        if isinstance(item, HistoryMessage):
            messages.append({"role": item.role, "content": item.content})
        elif isinstance(item, dict):
            messages.append(
                {"role": item.get("role", "user"), "content": item.get("content", "")}
            )
        else:
            messages.append({"role": "user", "content": str(item)})

    if prompt.query:
        messages.append({"role": "user", "content": prompt.query})
    elif not messages:
        messages.append({"role": "user", "content": "(empty)"})

    kwargs = dict(original_kwargs)
    if "tools" in kwargs and prompt.tools:
        kwargs["tools"] = json.loads(prompt.tools)
    return messages, kwargs


def _extract_cached_tokens(usage: Any) -> int:
    """Best-effort extraction of cached prompt tokens from OpenAI usage."""
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        return getattr(details, "cached_tokens", 0) or 0
    return 0


def _estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
) -> float:
    """Rough cost estimate using contextops pricing when response has no cost."""
    from contextops.pricing import CACHE_READ_DISCOUNT, price_per_million

    price_per_m = price_per_million(model)
    # Very rough: assume output costs ~4× input (common for many models).
    output_price_per_m = price_per_m * 4.0
    cached_cost = (cached_tokens / 1_000_000) * price_per_m * CACHE_READ_DISCOUNT
    uncached_cost = ((prompt_tokens - cached_tokens) / 1_000_000) * price_per_m
    output_cost = (completion_tokens / 1_000_000) * output_price_per_m
    return cached_cost + uncached_cost + output_cost


def patch(
    client: Any,
    *,
    optimize: bool = True,
    log: bool = True,
    db_path: str | None = None,
) -> Any:
    """Patch an OpenAI client so completions are cache-optimized and logged.

    Args:
        client: An `openai.OpenAI` instance (or compatible object with
            `chat.completions.create`).
        optimize: Whether to reorder messages for cache friendliness.
        log: Whether to persist a `CallLog` to the local SQLite logger.
        db_path: Optional path to the SQLite database.

    Returns:
        The wrapper function for introspection/testing.
    """
    if openai is None:
        raise RuntimeError(
            "openai not installed. Run: pip install 'openai'"
        )

    orig = client.chat.completions.create
    if id(orig) in _patched:
        return _patched[id(orig)][1]

    logger = Logger(Path(db_path) if db_path else None) if log else None

    @functools.wraps(orig)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        messages = list(kwargs.get("messages", []))
        tools = kwargs.get("tools")
        model = kwargs.get("model", "gpt-4o")

        if optimize and messages:
            prompt = _messages_to_prompt(messages, tools=tools, model=model)
            optimized = reorder(prompt)
            new_messages, new_kwargs = _prompt_to_messages(optimized, kwargs)
            kwargs = new_kwargs
            kwargs["messages"] = new_messages

        start = time.time()
        response = orig(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000

        if logger is not None:
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            cached_tokens = _extract_cached_tokens(usage)
            cost = getattr(response, "cost_usd", None)
            if cost is None:
                cost = _estimate_cost(model, prompt_tokens, completion_tokens, cached_tokens)

            entry = CallLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost,
                latency_ms=elapsed_ms,
                prompt_hash="",
                section_order=[],
                metadata={"via": "contextops-openai-integration"},
            )
            logger.log(entry)

        return response

    client.chat.completions.create = wrapped
    _patched[id(wrapped)] = (client, orig)
    return wrapped


def unpatch(client: Any) -> bool:
    """Restore the original `chat.completions.create` method on a client."""
    for wrapper_id, (patched_client, orig) in list(_patched.items()):
        if patched_client is client:
            client.chat.completions.create = orig
            del _patched[wrapper_id]
            return True
    return False
