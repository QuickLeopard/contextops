"""Built-in judge clients for v0.2.

- `EchoJudge` — returns a fixed response. For tests and offline demos.
- `LiteLLMJudge` — wraps any litellm-supported provider (optional dep).
- `CallableJudge` — wraps any user-provided callable.

The Protocol lives in judge.py.
"""

from __future__ import annotations

import time
from typing import Callable

from contextops.judge import JudgeClient


class EchoJudge:
    """Always returns the same JSON. Useful for tests and offline demos."""

    def __init__(self, score: float = 0.85, reason: str = "echo"):
        self.score = score
        self.reason = reason

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str:
        return f'{{"score": {self.score}, "reason": "{self.reason}"}}'


class CallableJudge:
    """Wrap any user-provided function as a judge."""

    def __init__(self, fn: Callable[..., str]):
        self.fn = fn

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str:
        return self.fn(model=model, messages=messages, temperature=temperature)


class LiteLLMJudge:
    """Real judge using litellm. Optional — pip install litellm.

    Retries transient failures (timeouts, rate limits, provider hiccups)
    with exponential backoff so a single bad call doesn't abort a whole eval.
    """

    def __init__(self, *, max_retries: int = 3, timeout: float = 30.0, backoff_base: float = 1.0):
        try:
            import litellm  # type: ignore

            self._litellm = litellm
        except ImportError as e:
            raise RuntimeError(
                "litellm not installed. Run: pip install 'contextops[integrations]'"
            ) from e
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=self.timeout,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - any provider/network error is retryable
                last_exc = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2**attempt))
        raise RuntimeError(
            f"LiteLLMJudge.complete failed after {self.max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc


def default_judge() -> JudgeClient:
    """Pick the best available judge. Falls back to EchoJudge."""
    try:
        import litellm  # noqa: F401

        return LiteLLMJudge()
    except ImportError:
        return EchoJudge()