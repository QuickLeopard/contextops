"""Tests for Prompt.from_openai_messages / from_anthropic_messages importers."""

from contextops.models import Prompt


# --- from_openai_messages ---

def test_from_openai_messages_basic():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    p = Prompt.from_openai_messages(msgs)
    assert p.system == "You are helpful."
    assert p.query == "What is 2+2?"
    # The first user turn + assistant reply become history.
    assert len(p.history) == 2
    assert p.history[0].role == "user" and p.history[0].content == "Hi"
    assert p.history[1].role == "assistant" and p.history[1].content == "Hello!"


def test_from_openai_messages_multiple_system_merged():
    msgs = [
        {"role": "system", "content": "Rule 1."},
        {"role": "system", "content": "Rule 2."},
        {"role": "user", "content": "Q"},
    ]
    p = Prompt.from_openai_messages(msgs)
    assert p.system == "Rule 1.\n\nRule 2."
    assert p.query == "Q"


def test_from_openai_messages_single_user():
    msgs = [{"role": "user", "content": "just a question"}]
    p = Prompt.from_openai_messages(msgs)
    assert p.system == ""
    assert p.query == "just a question"
    assert p.history == []


def test_from_openai_messages_list_content():
    """OpenAI vision-style content (list of parts) is coerced to text."""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]},
    ]
    p = Prompt.from_openai_messages(msgs)
    assert "What's in this image?" in p.query


def test_from_openai_messages_tool_messages_dropped():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "call the tool"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "42", "tool_call_id": "x"},
        {"role": "user", "content": "thanks"},
    ]
    p = Prompt.from_openai_messages(msgs)
    # First user turn demoted to history; tool message dropped; final user = query.
    assert p.query == "thanks"
    assert any(h.role == "user" and h.content == "call the tool" for h in p.history)


# --- from_anthropic_messages ---

def test_from_anthropic_messages_string_system():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ]
    p = Prompt.from_anthropic_messages(msgs, system="You are a bot.")
    assert "You are a bot." in p.system
    assert p.query == "Bye"


def test_from_anthropic_messages_block_system():
    """Anthropic system as a list of content blocks — text blocks are extracted."""
    msgs = [{"role": "user", "content": "Q"}]
    p = Prompt.from_anthropic_messages(msgs, system=[
        {"type": "text", "text": "Rule A."},
        {"type": "text", "text": "Rule B."},
    ])
    assert "Rule A." in p.system and "Rule B." in p.system
    assert p.query == "Q"


def test_from_anthropic_messages_no_system():
    msgs = [{"role": "user", "content": "Q"}]
    p = Prompt.from_anthropic_messages(msgs, system=None)
    assert p.system == ""
    assert p.query == "Q"


# --- round-trip: import then split ---

def test_imported_prompt_splits_correctly():
    """An imported prompt should split into a non-empty system prefix."""
    from contextops.render import split_prompt
    msgs = [
        {"role": "system", "content": "You are a coding assistant. " * 200},
        {"role": "user", "content": "Fix this bug"},
    ]
    p = Prompt.from_openai_messages(msgs)
    split = split_prompt(p)
    assert "coding assistant" in split.system
    assert split.user == "Fix this bug"
