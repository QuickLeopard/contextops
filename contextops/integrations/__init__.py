"""Optional integrations for logging and patching LLM clients."""

from __future__ import annotations

from contextops.integrations.litellm import install_callback

__all__ = ["install_callback"]
