"""Tests for the one-line OpenAI SDK integration."""

from __future__ import annotations

import json


from contextops.integrations.openai import (
    _messages_to_prompt,
    _prompt_to_messages,
    patch,
    unpatch,
)
from contextops.logger import Logger
from contextops.models import Prompt


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    prompt_tokens_details = None


class _FakeResponse:
    def __init__(self):
        self.usage = _FakeUsage()
        self.model = "gpt-4o"


class _FakeCompletions:
    def __init__(self, response=None):
        self.response = response or _FakeResponse()
        self.calls: list[dict] = []

    def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response=None):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(response)})()


def test_messages_to_prompt_extracts_system_and_query():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    prompt = _messages_to_prompt(messages, model="gpt-4o")
    assert prompt.system == "You are helpful."
    assert prompt.query == "Hello!"
    assert prompt.history == []


def test_messages_to_prompt_treats_interleaved_user_as_history():
    messages = [
        {"role": "user", "content": "First."},
        {"role": "assistant", "content": "Reply."},
        {"role": "user", "content": "Second."},
    ]
    prompt = _messages_to_prompt(messages, model="gpt-4o")
    assert prompt.query == "Second."
    assert len(prompt.history) == 2
    assert prompt.history[0].content == "First."
    assert prompt.history[1].content == "Reply."


def test_messages_to_prompt_serializes_tools():
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    messages = [{"role": "user", "content": "Hi"}]
    prompt = _messages_to_prompt(messages, tools=tools, model="gpt-4o")
    assert json.loads(prompt.tools) == tools


def test_prompt_to_messages_preserves_system_first():
    prompt = Prompt(system="Sys", query="Q", model="gpt-4o")
    messages, kwargs = _prompt_to_messages(prompt, {})
    assert messages[0] == {"role": "system", "content": "Sys"}
    assert messages[-1] == {"role": "user", "content": "Q"}


def test_patch_wraps_create_method():
    client = _FakeClient()
    patch(client, optimize=False, log=False)
    client.chat.completions.create(model="gpt-4o", messages=[])
    assert len(client.chat.completions.calls) == 1


def test_patch_reorders_messages_so_system_is_first():
    client = _FakeClient()
    patch(client, log=False)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Query first."},
            {"role": "system", "content": "System second."},
        ],
    )
    assert client.chat.completions.calls[0]["messages"][0]["role"] == "system"
    assert client.chat.completions.calls[0]["messages"][-1]["role"] == "user"


def test_patch_logs_to_custom_db(tmp_path):
    db_path = tmp_path / "calls.db"
    client = _FakeClient()
    patch(client, db_path=str(db_path))
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )

    logger = Logger(db_path)
    rows = logger.recent(limit=1)
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-4o"
    assert rows[0]["metadata"] == {"via": "contextops-openai-integration"}


def test_unpatch_restores_original_create():
    client = _FakeClient()
    original_func = client.chat.completions.create.__func__
    patch(client, optimize=False, log=False)
    assert client.chat.completions.create is not original_func
    unpatch(client)
    assert client.chat.completions.create.__func__ is original_func


def test_patch_is_idempotent():
    client = _FakeClient()
    patch(client, optimize=False, log=False)
    first_wrapped = client.chat.completions.create
    patch(client, optimize=False, log=False)
    assert client.chat.completions.create is first_wrapped
