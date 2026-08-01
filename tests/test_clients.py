"""Tests for judge clients — retry/backoff behavior for LiteLLMJudge."""

from __future__ import annotations

import pytest

from contextops.clients import LiteLLMJudge


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def test_litellm_judge_retries_then_succeeds(monkeypatch):
    """A transient failure followed by success should not raise, and should
    not require more than `max_retries` attempts."""
    judge = LiteLLMJudge(max_retries=3, backoff_base=0.0)
    calls = {"n": 0}

    def flaky_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient provider error")
        return _FakeResponse('{"score": 0.9, "reason": "ok"}')

    monkeypatch.setattr(judge._litellm, "completion", flaky_completion)

    result = judge.complete(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert result == '{"score": 0.9, "reason": "ok"}'
    assert calls["n"] == 3


def test_litellm_judge_raises_after_exhausting_retries(monkeypatch):
    """A permanent failure should raise once retries are exhausted, not hang
    or silently swallow the error."""
    judge = LiteLLMJudge(max_retries=2, backoff_base=0.0)
    calls = {"n": 0}

    def always_fails(**kwargs):
        calls["n"] += 1
        raise RuntimeError("permanent failure")

    monkeypatch.setattr(judge._litellm, "completion", always_fails)

    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        judge.complete(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_litellm_judge_passes_timeout_to_completion(monkeypatch):
    """The configured timeout must be forwarded to litellm.completion."""
    judge = LiteLLMJudge(max_retries=0, timeout=12.5)
    seen_kwargs = {}

    def capture_completion(**kwargs):
        seen_kwargs.update(kwargs)
        return _FakeResponse("{}")

    monkeypatch.setattr(judge._litellm, "completion", capture_completion)

    judge.complete(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert seen_kwargs["timeout"] == 12.5
