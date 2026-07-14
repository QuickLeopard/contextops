"""Render a `Prompt` into the shapes providers expect.

This is the bridge between ContextOps's structured prompt model and the
`(system, user)` message split that providers want for prompt caching:

    prompt = Prompt(system=..., tools=..., query=...)
    optimized = reorder(prompt)
    system_str, user_str, _ = split_prompt(optimized)

    # Now hand (system_str, user_str) to your provider — Anthropic, OpenAI,
    # OpenRouter, anyone. The stable prefix lives in `system_str` so the
    # provider's cache_control (or auto prefix cache) can engage.

`split_prompt` respects `Prompt.render_order`, so it always reflects the
ordering produced by `reorder()` (or any custom `render_order` you set).
"""

from __future__ import annotations

from dataclasses import dataclass

from contextops.models import Prompt, Section

# Sections treated as "stable" (cacheable) content — sent as the system message.
# Anything not in this set is "variable" content sent as the user message.
STABLE_SECTIONS: frozenset[str] = frozenset({"system", "tools", "role"})


@dataclass(frozen=True)
class PromptSplit:
    """Result of splitting a `Prompt` into cacheable system + variable user content."""

    system: str       # stable sections joined with double newlines (empty if none)
    user: str         # variable sections joined in render order (empty if none)
    section_order: list[Section]  # full ordered list of section names rendered


def render_prompt(p: Prompt) -> tuple[str, list[Section]]:
    """Render a prompt to a single string + return its section order.

    Respects `Prompt.render_order` (set by `reorder()`), so the output reflects
    whatever ordering is currently active on the prompt.
    """
    sections = p.sections()
    parts = [content for _, content in sections]
    order = [name for name, _ in sections]
    return "\n\n".join(parts), order


def split_prompt(p: Prompt) -> PromptSplit:
    """Split a prompt into (system_content, user_content, section_order).

    - `system_content` = stable sections (`system`, `tools`, `role`) joined with
      double newlines. Send this as the provider's `system` field with
      `cache_control: {type: "ephemeral"}` (Anthropic) or rely on the provider's
      auto prefix cache (OpenAI).
    - `user_content` = variable sections (`context`, `documents`, `history`,
      `query`) in render order. Send this as the user message.
    - `section_order` = the full ordered list of section names, useful for
      logging/telemetry.

    Respects `Prompt.render_order`, so call this *after* `reorder()` to get the
    cache-optimized layout.
    """
    _, order = render_prompt(p)
    section_map: dict[str, str] = {name: content for name, content in p.sections()}

    sys_parts: list[str] = []
    usr_parts: list[str] = []
    for name in order:
        if name in STABLE_SECTIONS:
            sys_parts.append(section_map.get(name, ""))
        else:
            usr_parts.append(section_map.get(name, ""))
    return PromptSplit(
        system="\n\n".join(s for s in sys_parts if s),
        user="\n\n".join(u for u in usr_parts if u),
        section_order=list(order),
    )
