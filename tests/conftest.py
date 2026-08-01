"""Shared pytest fixtures/config.

Avoid litellm's import-time network fetch of its remote model-cost map —
it's slow and flaky in sandboxed/offline test environments, and irrelevant
to the judge-scoring logic being tested here. Must be set before litellm
is first imported (any test importing `contextops.clients` triggers it).
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
