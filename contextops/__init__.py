"""ContextOps — cache-aware prompt optimizer + local cost logger."""

from contextops.optimizer import optimize, reorder, count_tokens, estimate_cache_hit
from contextops.logger import Logger, CallLog
from contextops.eval import compare, evaluate, evaluate_ab
from contextops.judge import list_metrics
from contextops.dataset import DatasetItem, load as load_dataset
from contextops.models import Prompt, OptimizationResult, HistoryMessage
from contextops.clients import EchoJudge, CallableJudge, LiteLLMJudge, default_judge

__version__ = "0.2.0"

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