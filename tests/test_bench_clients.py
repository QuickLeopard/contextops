"""Offline tests for bench client helpers (no network).

Covers the helpers extracted from OpenRouterClient.complete in Phase 7, plus
the registry and the LSP-fixed EchoClient. None of these hit a real API.
"""

import os

import pytest

# OpenRouterClient construction requires an API key; set a dummy one before import.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-for-offline-tests")

from contextops_bench.clients import (  # noqa: E402
    CLIENTS,
    EchoClient,
    OpenRouterClient,
    get_client,
)


# --- _shape_messages ---

def test_shape_messages_anthropic_per_block_two_blocks():
    """Anthropic model + per_block (default) → system split into intro + cache_control body."""
    client = OpenRouterClient()
    out = client._shape_messages(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": "hi"}],
        system="S" * 1000,
    )
    # First message is the system block with two content parts.
    assert out[0]["role"] == "system"
    content = out[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    # Second part carries the cache_control marker.
    assert content[1]["cache_control"] == {"type": "ephemeral"}
    # The user message follows.
    assert out[1] == {"role": "user", "content": "hi"}


def test_shape_messages_non_anthropic_plain_prepend():
    """Non-Anthropic model → plain system message prepend, no cache_control."""
    client = OpenRouterClient()
    out = client._shape_messages(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        system="a system prompt",
    )
    assert out[0] == {"role": "system", "content": "a system prompt"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_shape_messages_no_system_unchanged():
    """No system content → messages returned unchanged."""
    client = OpenRouterClient()
    msgs = [{"role": "user", "content": "hi"}]
    out = client._shape_messages(model="anthropic/claude-haiku-4.5", messages=msgs, system=None)
    assert out == msgs


# --- _apply_provider_pinning ---

def test_apply_provider_pinning_anthropic_defaults_to_pin():
    """Anthropic model + per_block mode (no explicit pin) → auto-pins to anthropic."""
    client = OpenRouterClient()
    payload: dict = {}
    client._apply_provider_pinning(payload, model="anthropic/claude-haiku-4.5", pin_provider=None)
    assert payload["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}


def test_apply_provider_pinning_non_anthropic_no_pin():
    """Non-Anthropic model → no pinning applied."""
    client = OpenRouterClient()
    payload: dict = {}
    client._apply_provider_pinning(payload, model="openai/gpt-4o-mini", pin_provider=None)
    assert "provider" not in payload


def test_apply_provider_pinning_explicit_override():
    """Explicit pin_provider takes precedence over the default."""
    client = OpenRouterClient()
    payload: dict = {}
    client._apply_provider_pinning(
        payload, model="anthropic/claude-haiku-4.5", pin_provider="bedrock",
    )
    assert payload["provider"] == {"order": ["bedrock"], "allow_fallbacks": False}


# --- _maybe_debug ---

def test_maybe_debug_silent_by_default(capsys):
    client = OpenRouterClient()
    client._maybe_debug(raw={"provider": "anthropic"}, cached_tokens=100, prompt_tokens=200)
    assert capsys.readouterr().out == ""


def test_maybe_debug_prints_when_enabled(capsys, monkeypatch):
    client = OpenRouterClient()
    monkeypatch.setattr(client, "debug_provider", True)
    client._maybe_debug(raw={"provider": "anthropic"}, cached_tokens=100, prompt_tokens=200)
    out = capsys.readouterr().out
    assert "provider=anthropic" in out
    assert "cached" in out


# --- registry ---

def test_registry_contains_all_providers():
    assert set(CLIENTS) == {"ollama", "lmstudio", "openrouter", "direct_anthropic", "echo"}


def test_get_client_unknown_raises_with_choices():
    with pytest.raises(ValueError, match="direct_anthropic"):
        get_client("does-not-exist")


def test_get_client_alias_anthropic():
    """The 'anthropic' alias resolves to 'direct_anthropic'."""
    # Can't construct without a key, so just check the alias map resolves.
    from contextops_bench.clients import _PROVIDER_ALIASES
    assert _PROVIDER_ALIASES["anthropic"] == "direct_anthropic"


def test_echo_client_conforms_to_protocol():
    """EchoClient satisfies the BenchClient Protocol (runtime-checkable)."""
    from contextops_bench.client_protocol import BenchClient
    assert isinstance(EchoClient(), BenchClient)
