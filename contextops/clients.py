"""Built-in judge clients for v0.2.

- `EchoJudge` — returns a fixed response. For tests and offline demos.
- `LiteLLMJudge` — wraps any litellm-supported provider (optional dep).
- `CallableJudge` — wraps any user-provided callable.

The Protocol lives in judge.py.
"""

from __future__ import annotations

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
    """Real judge using litellm. Optional — pip install litellm."""

    def __init__(self):
        try:
            import litellm  # type: ignore

            self._litellm = litellm
        except ImportError as e:
            raise RuntimeError(
                "litellm not installed. Run: pip install 'contextops[integrations]'"
            ) from e

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str:
        resp = self._litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def default_judge() -> JudgeClient:
    """Pick the best available judge. Falls back to EchoJudge."""
    try:
        import litellm  # noqa: F401

        return LiteLLMJudge()
    except ImportError:
        return EchoJudge()