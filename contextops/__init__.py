"""ContextOps — cache-aware prompt optimizer + local cost logger."""

from importlib import metadata as _metadata

from contextops.optimizer import optimize, reorder, count_tokens, estimate_cache_hit
from contextops.logger import Logger, CallLog
from contextops.eval import compare, evaluate, evaluate_ab
from contextops.judge import list_metrics
from contextops.dataset import DatasetItem, load as load_dataset
from contextops.models import Prompt, OptimizationResult, HistoryMessage
from contextops.clients import EchoJudge, CallableJudge, LiteLLMJudge, default_judge

try:
    # Single source of truth: the version declared in pyproject.toml.
    __version__ = _metadata.version("contextops-tool")
except _metadata.PackageNotFoundError:
    # Package not installed (e.g. running from a raw checkout) — keep a
    # reasonable fallback that should be bumped alongside pyproject.toml.
    __version__ = "0.3.2"

__all__ = [
    "optimize",
    "reorder",
    "count_tokens",
    "estimate_cache_hit",
    "Logger",
    "CallLog",
    "compare",
    "evaluate",
    "evaluate_ab",
    "list_metrics",
    "DatasetItem",
    "load_dataset",
    "Prompt",
    "OptimizationResult",
    "HistoryMessage",
    "EchoJudge",
    "CallableJudge",
    "LiteLLMJudge",
    "default_judge",
    "__version__",
]