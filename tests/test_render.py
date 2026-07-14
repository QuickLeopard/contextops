"""Tests for the public render / split_prompt helpers."""

from contextops.models import Prompt
from contextops.optimizer import reorder
from contextops.render import PromptSplit, STABLE_SECTIONS, render_prompt, split_prompt


def test_render_prompt_joins_sections():
    p = Prompt(system="S", query="Q")
    text, order = render_prompt(p)
    assert text == "S\n\nQ"
    assert order == ["system", "query"]


def test_render_prompt_respects_render_order():
    p = Prompt(system="S", query="Q")
    p.render_order = ["query", "system"]
    text, order = render_prompt(p)
    assert order == ["query", "system"]
    assert text == "Q\n\nS"


def test_split_prompt_basic():
    """system/tools/role → system side; context/documents/history/query → user side."""
    p = Prompt(system="SYS", tools="TOOLS", context="CTX", query="Q?")
    split = split_prompt(p)
    assert isinstance(split, PromptSplit)
    assert "SYS" in split.system and "TOOLS" in split.system
    assert "CTX" in split.user and "Q?" in split.user
    # System side should NOT contain variable content.
    assert "CTX" not in split.system
    assert "Q?" not in split.system


def test_split_prompt_after_reorder():
    """split_prompt reflects the ordering produced by reorder()."""
    p = Prompt(query="Q", system="S", tools="T")
    optimized = reorder(p)
    split = split_prompt(optimized)
    # Stable sections land in system, variable in user.
    assert "S" in split.system and "T" in split.system
    assert "Q" in split.user


def test_split_prompt_empty_sections_omitted():
    """Empty sections shouldn't produce stray double-newlines."""
    p = Prompt(system="S", query="Q")  # no tools/role/context/documents/history
    split = split_prompt(p)
    assert split.system == "S"
    assert split.user == "Q"


def test_split_prompt_all_stable():
    """A prompt with only stable sections → empty user side."""
    p = Prompt(system="S", tools="T", role="R")
    split = split_prompt(p)
    assert split.system == "S\n\nT\n\nR"
    assert split.user == ""


def test_split_prompt_all_variable():
    """A prompt with only variable sections → empty system side."""
    p = Prompt(query="Q", context="C")
    split = split_prompt(p)
    assert split.system == ""
    assert "Q" in split.user and "C" in split.user


def test_stable_sections_constant():
    assert STABLE_SECTIONS == frozenset({"system", "tools", "role"})
