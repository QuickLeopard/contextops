"""LLM-as-judge evaluation for v0.2.

Uses any chat-completion API to score responses on multiple dimensions.
Each metric returns 0..1. Higher is better (except 'hallucination', which is inverted).
"""

from __future__ import annotations

import json
import re
from typing import Optional, Protocol


class JudgeClient(Protocol):
    """Minimal interface for a chat-completion client."""

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str: ...


# Metric definitions. Each is a prompt template + a JSON parser.
# Judges return {"score": float, "reason": str}.

_METRICS: dict[str, dict] = {
    "faithfulness": {
        "description": "Does the response stick to facts in the provided context?",
        "system": (
            "You are a strict evaluator. Score how faithful the RESPONSE is to the CONTEXT. "
            "Faithful means: every claim in the response must be supported by the context. "
            "If the response invents facts not present in the context, lower the score."
        ),
        "user": (
            "CONTEXT:\n{context}\n\n"
            "RESPONSE:\n{response}\n\n"
            "Reply with JSON only: {{\"score\": <0..1>, \"reason\": \"<one sentence>\"}}"
        ),
        "default_score_if_missing": 0.5,
    },
    "relevance": {
        "description": "Does the response address the user's query?",
        "system": (
            "You are a strict evaluator. Score how RELEVANT the response is to the QUERY. "
            "A relevant response directly addresses what the user asked, on-topic and on-target."
        ),
        "user": (
            "QUERY:\n{query}\n\n"
            "RESPONSE:\n{response}\n\n"
            "Reply with JSON only: {{\"score\": <0..1>, \"reason\": \"<one sentence>\"}}"
        ),
        "default_score_if_missing": 0.5,
    },
    "completeness": {
        "description": "Does the response cover the expected answer?",
        "system": (
            "You are a strict evaluator. Score how COMPLETE the response is compared to EXPECTED. "
            "Complete means: all key facts from EXPECTED are present in RESPONSE. "
            "Partial coverage = lower score."
        ),
        "user": (
            "EXPECTED ANSWER:\n{expected}\n\n"
            "RESPONSE:\n{response}\n\n"
            "Reply with JSON only: {{\"score\": <0..1>, \"reason\": \"<one sentence>\"}}"
        ),
        "default_score_if_missing": 0.5,
    },
    "conciseness": {
        "description": "Is the response free of unnecessary fluff?",
        "system": (
            "You are a strict evaluator. Score how CONCISE the response is. "
            "Concise = no preamble, no hedging, no repeated info. Direct and tight."
        ),
        "user": (
            "RESPONSE:\n{response}\n\n"
            "Reply with JSON only: {{\"score\": <0..1>, \"reason\": \"<one sentence>\"}}"
        ),
        "default_score_if_missing": 0.5,
    },
}


def list_metrics() -> list[str]:
    """All available metric names."""
    return list(_METRICS.keys())


def _extract_json(text: str) -> Optional[dict]:
    """Robust JSON extraction. Handles ```json fences and trailing prose."""
    # Try direct parse first.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip markdown code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Greedy brace match.
    brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _build_messages(metric: str, response: str, **fields) -> list[dict]:
    """Build chat messages for a given metric."""
    cfg = _METRICS[metric]
    user_content = cfg["user"].format(response=response, **fields)
    return [
        {"role": "system", "content": cfg["system"]},
        {"role": "user", "content": user_content},
    ]


def score_one(
    metric: str,
    response: str,
    *,
    judge: JudgeClient,
    model: str = "gpt-4o-mini",
    context: str = "",
    query: str = "",
    expected: str = "",
) -> dict:
    """Score one response on one metric. Returns {score, reason, raw}."""
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric: {metric}. Available: {list_metrics()}")

    messages = _build_messages(
        metric, response, context=context, query=query, expected=expected
    )
    raw = judge.complete(model=model, messages=messages, temperature=0.0)
    parsed = _extract_json(raw)

    if parsed is None or "score" not in parsed:
        return {
            "metric": metric,
            "score": _METRICS[metric]["default_score_if_missing"],
            "reason": f"judge returned non-JSON: {raw[:120]}",
            "raw": raw,
        }

    score = float(parsed["score"])
    score = max(0.0, min(1.0, score))  # clamp
    return {
        "metric": metric,
        "score": round(score, 3),
        "reason": str(parsed.get("reason", ""))[:300],
        "raw": raw,
    }


def score_many(
    responses: list[str],
    *,
    metrics: list[str],
    judge: JudgeClient,
    model: str = "gpt-4o-mini",
    contexts: Optional[list[str]] = None,
    queries: Optional[list[str]] = None,
    expecteds: Optional[list[str]] = None,
    on_progress=None,
) -> list[dict]:
    """Score many responses on many metrics.

    Returns a list of dicts:
        [{"index": 0, "metric": "faithfulness", "score": 0.9, "reason": "..."}, ...]
    """
    contexts = contexts or [""] * len(responses)
    queries = queries or [""] * len(responses)
    expecteds = expecteds or [""] * len(responses)
    on_progress = on_progress or (lambda i, n, m: None)

    out: list[dict] = []
    for i, response in enumerate(responses):
        for metric in metrics:
            result = score_one(
                metric,
                response,
                judge=judge,
                model=model,
                context=contexts[i] if i < len(contexts) else "",
                query=queries[i] if i < len(queries) else "",
                expected=expecteds[i] if i < len(expecteds) else "",
            )
            result["index"] = i
            out.append(result)
            on_progress(i + 1, len(responses), metric)
    return out