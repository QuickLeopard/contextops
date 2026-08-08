"""Deterministic quality gates for benchmark results.

A bench run's headline numbers (cache hit rate delta, cost delta) can look
impressive while being statistically meaningless — too few samples, too many
errors, or a provider path that structurally can't measure what we're trying
to measure (e.g. OpenRouter drops Anthropic's `cache_control` marker; see
`contextops_bench.clients.AnthropicDirectClient` docstring).

`evaluate_quality_gate` turns those pitfalls into explicit, reproducible
pass/fail rules so every summary.json (and the public dashboard built from
them) carries an honest "is this real?" verdict instead of just raw numbers.

For paid cloud providers verification requires a statistically significant
cost delta. For local/self-hosted providers (ollama, vllm, tgi, lmstudio)
cost is always zero, so verification falls back to a significant token-count
or latency delta instead.
"""

from __future__ import annotations

import re

# A run needs at least this many paired (optimized, baseline) observations
# after error/edge-case exclusion to be considered adequately powered.
MIN_N = 20

# If either arm's error rate exceeds this, the underlying numbers are too
# contaminated by failed calls (bad model IDs, network flakiness, rate
# limits) to trust.
MAX_ERROR_RATE = 0.15

# (provider, model_prefix) pairs where the cache marker is known to be
# dropped or unsupported before it reaches the underlying model, making
# cache_hit_rate structurally ~0% regardless of prompt ordering. This is a
# routing/adapter limitation, not a ContextOps bug — see the
# AnthropicDirectClient / ZenDirectClient docstrings in clients.py.
CACHE_MARKER_DROP_PATHS: tuple[tuple[str, str], ...] = (
    ("openrouter", "anthropic/"),
)

# Providers that report zero cost because the model is self-hosted. For these
# paths the quality gate cannot use USD deltas; it verifies on statistically
# significant token or latency savings instead.
LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "tgi", "lmstudio"})


def cache_marker_dropped(provider: str, model: str) -> bool:
    """True if `provider` is known to strip the cache marker for `model`."""
    return any(
        provider == p and model.startswith(prefix)
        for p, prefix in CACHE_MARKER_DROP_PATHS
    )


def is_local_provider(provider: str) -> bool:
    """True if `provider` is a self-hosted path with no measurable USD cost."""
    return provider.lower() in LOCAL_PROVIDERS


# Error categories, ordered from most to least severe for confidence scoring.
# "auth" errors (401/403) mean the request never reached the model at all —
# they invalidate the run rather than just adding noise, since they usually
# indicate an expired/rotated/invalid API key affecting some or all calls.
ERROR_CATEGORIES = ("auth", "rate_limit", "server_error", "client_error", "network", "unknown")

_HTTP_ERROR_RE = re.compile(r"HTTP Error (\d{3})")
_STATUS_CODE_RE = re.compile(r"\b(\d{3})\b")
_NETWORK_PATTERNS = (
    "ssl", "eof", "remote end closed", "connection reset", "connection refused",
    "timed out", "timeout", "urlopen error", "broken pipe", "name or service not known",
)


def classify_error(message: str) -> str:
    """Categorize a raw error string into one of `ERROR_CATEGORIES`.

    Looks for an HTTP status code first (`HTTP Error 403: Forbidden` style,
    as raised by `urllib.error.HTTPError`), then falls back to matching
    common network-failure substrings (SSL resets, timeouts, connection
    drops — the actual errors observed in real bench runs so far).
    """
    if not message:
        return "unknown"
    msg = message.lower()

    code_match = _HTTP_ERROR_RE.search(message) or _STATUS_CODE_RE.search(message)
    if code_match:
        code = int(code_match.group(1))
        if code in (401, 403):
            return "auth"
        if code == 429:
            return "rate_limit"
        if 500 <= code < 600:
            return "server_error"
        if 400 <= code < 500:
            return "client_error"

    if any(pattern in msg for pattern in _NETWORK_PATTERNS):
        return "network"

    return "unknown"


def summarize_errors(error_messages: list[str]) -> dict[str, int]:
    """Count error messages by category. Returns e.g. {"network": 3, "auth": 1}."""
    breakdown: dict[str, int] = {}
    for msg in error_messages:
        category = classify_error(msg)
        breakdown[category] = breakdown.get(category, 0) + 1
    return breakdown


def confidence_level(error_rate: float, error_breakdown: dict[str, int] | None = None) -> str:
    """Map error rate + error severity to a human-readable confidence label.

    Any `auth` errors force "invalid" regardless of rate: an auth failure
    means the API key was rejected for at least one call, which usually
    means it was rejected for a batch of adjacent calls too (rotated/rate
    -limited key), making the whole run's numbers suspect rather than just
    "a bit noisy". Otherwise, confidence degrades smoothly with error rate.
    """
    breakdown = error_breakdown or {}
    if breakdown.get("auth", 0) > 0:
        return "invalid"
    if error_rate <= 0.05:
        return "high"
    if error_rate <= MAX_ERROR_RATE:
        return "medium"
    return "low"


def evaluate_quality_gate(
    summary: dict,
    *,
    provider: str = "",
    model: str = "",
    min_n: int = MIN_N,
    max_error_rate: float = MAX_ERROR_RATE,
) -> dict:
    """Compute a deterministic pass/fail quality gate for a bench summary.

    `verified=True` means: adequately powered (n >= min_n), low error rate,
    no detected auth failures, and a statistically significant delta on the
    metric the provider can actually measure.

    For most providers this is a cost-delta 95% CI that excludes zero. For
    self-hosted/local providers (ollama, vllm, tgi, lmstudio) cost is always
    zero, so verification falls back to a significant token-count or latency
    delta. A significant *increase* is just as "verified" as a decrease; it
    simply isn't a "win".

    Returns a dict with individual boolean flags plus human-readable
    `reasons` for any failure, so callers (CLI output, dashboard) can
    explain *why* a run isn't trustworthy instead of just hiding it.
    """
    opt = summary.get("optimized", {}) or {}
    base = summary.get("baseline", {}) or {}
    delta = summary.get("delta", {}) or {}

    n = min(opt.get("n", 0), base.get("n", 0))
    opt_err_rate = (opt.get("errors", 0) / opt["n"]) if opt.get("n") else 1.0
    base_err_rate = (base.get("errors", 0) / base["n"]) if base.get("n") else 1.0
    error_rate = max(opt_err_rate, base_err_rate)

    # Combine both arms' error breakdowns (if present) for confidence scoring.
    error_breakdown: dict[str, int] = {}
    for arm in (opt, base):
        for category, count in (arm.get("error_breakdown") or {}).items():
            error_breakdown[category] = error_breakdown.get(category, 0) + count
    confidence = confidence_level(error_rate, error_breakdown)
    has_auth_errors = error_breakdown.get("auth", 0) > 0

    low_n = n < min_n
    high_error_rate = error_rate > max_error_rate

    # Cost significance: the default metric for paid/cloud providers.
    ci_low = delta.get("cost_delta_ci_low_usd")
    ci_high = delta.get("cost_delta_ci_high_usd")
    if ci_low is not None and ci_high is not None:
        has_cost_ci = True
        cost_significant = not (ci_low <= 0 <= ci_high)
    else:
        has_cost_ci = False
        cost_significant = False

    # Token and latency significance: used for local/self-hosted providers
    # where cost is identically zero. Significant *increase* is as "verified"
    # as a decrease; it simply isn't a win.
    tk_low = delta.get("prompt_tokens_delta_ci_low")
    tk_high = delta.get("prompt_tokens_delta_ci_high")
    if tk_low is not None and tk_high is not None:
        token_significant = not (tk_low <= 0 <= tk_high)
    else:
        token_significant = False

    lat_low = delta.get("latency_ms_p50_delta_ci_low")
    lat_high = delta.get("latency_ms_p50_delta_ci_high")
    if lat_low is not None and lat_high is not None:
        latency_significant = not (lat_low <= 0 <= lat_high)
    else:
        latency_significant = False

    marker_dropped = cache_marker_dropped(provider, model)
    local_provider = is_local_provider(provider)

    reasons: list[str] = []
    if low_n:
        reasons.append(f"n={n} < min_n={min_n}")
    if has_auth_errors:
        reasons.append(
            f"{error_breakdown['auth']} auth error(s) detected (401/403) — "
            f"API key was rejected for at least one call, results are unreliable"
        )
    if high_error_rate:
        reasons.append(f"error_rate={error_rate:.0%} > max_error_rate={max_error_rate:.0%}")

    if local_provider:
        if not (token_significant or latency_significant):
            if tk_low is not None and lat_low is not None:
                reasons.append(
                    f"local provider: token Δ 95% CI [{tk_low:+.1f}, {tk_high:+.1f}] "
                    f"and latency Δ 95% CI [{lat_low:+.1f}, {lat_high:+.1f}] both include zero"
                )
            elif tk_low is not None:
                reasons.append(
                    f"local provider: token Δ 95% CI [{tk_low:+.1f}, {tk_high:+.1f}] includes zero; "
                    f"no latency CI available"
                )
            elif lat_low is not None:
                reasons.append(
                    f"local provider: latency Δ 95% CI [{lat_low:+.1f}, {lat_high:+.1f}] includes zero; "
                    f"no token CI available"
                )
            else:
                reasons.append(
                    "local provider: no token or latency confidence interval available"
                )
        else:
            if token_significant:
                reasons.append(
                    f"local provider verified on token Δ 95% CI "
                    f"[{tk_low:+.1f}, {tk_high:+.1f}]"
                )
            if latency_significant:
                reasons.append(
                    f"local provider verified on latency Δ 95% CI "
                    f"[{lat_low:+.1f}, {lat_high:+.1f}]"
                )
    else:
        if not has_cost_ci:
            reasons.append("no confidence interval available (too few paired samples or legacy summary format)")
        elif not cost_significant:
            reasons.append(f"cost delta 95% CI [{ci_low:+.6f}, {ci_high:+.6f}] includes zero")

    if marker_dropped:
        reasons.append(
            f"provider {provider!r} is known to drop the cache marker for "
            f"{model!r} — cache_hit_rate is not measurable on this path"
        )

    significant = (
        (token_significant or latency_significant)
        if local_provider
        else cost_significant
    )

    # `marker_dropped` and detected auth errors both block `verified`: a
    # cache marker that structurally can't survive means any cost delta
    # isn't attributable evidence, and auth errors mean some calls never
    # reached the model at all, regardless of statistical significance.
    verified = (
        not low_n
        and not high_error_rate
        and not has_auth_errors
        and significant
        and not marker_dropped
    )

    return {
        "n": n,
        "error_rate": round(error_rate, 3),
        "error_breakdown": error_breakdown,
        "confidence": confidence,
        "low_n": low_n,
        "high_error_rate": high_error_rate,
        "has_auth_errors": has_auth_errors,
        "has_ci": has_cost_ci,
        "cost_significant": cost_significant,
        "token_significant": token_significant,
        "latency_significant": latency_significant,
        "local_provider": local_provider,
        "significant": significant,
        "cache_marker_dropped": marker_dropped,
        "verified": verified,
        "reasons": reasons,
    }

