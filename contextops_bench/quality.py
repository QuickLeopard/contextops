"""Deterministic quality gates for benchmark results.

A bench run's headline numbers (cache hit rate delta, cost delta) can look
impressive while being statistically meaningless — too few samples, too many
errors, or a provider path that structurally can't measure what we're trying
to measure (e.g. OpenRouter drops Anthropic's `cache_control` marker; see
`contextops_bench.clients.AnthropicDirectClient` docstring).

`evaluate_quality_gate` turns those pitfalls into explicit, reproducible
pass/fail rules so every summary.json (and the public dashboard built from
them) carries an honest "is this real?" verdict instead of just raw numbers.
"""

from __future__ import annotations

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


def cache_marker_dropped(provider: str, model: str) -> bool:
    """True if `provider` is known to strip the cache marker for `model`."""
    return any(
        provider == p and model.startswith(prefix)
        for p, prefix in CACHE_MARKER_DROP_PATHS
    )


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
    and a cost-delta 95% CI that excludes zero (statistically significant)
    — regardless of the direction of that delta. A significant *increase*
    in cost is just as "verified" as a decrease; it simply isn't a "win".

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

    low_n = n < min_n
    high_error_rate = error_rate > max_error_rate

    ci_low = delta.get("cost_delta_ci_low_usd")
    ci_high = delta.get("cost_delta_ci_high_usd")
    if ci_low is not None and ci_high is not None:
        has_ci = True
        significant = not (ci_low <= 0 <= ci_high)
    else:
        has_ci = False
        significant = False

    marker_dropped = cache_marker_dropped(provider, model)

    reasons: list[str] = []
    if low_n:
        reasons.append(f"n={n} < min_n={min_n}")
    if high_error_rate:
        reasons.append(f"error_rate={error_rate:.0%} > max_error_rate={max_error_rate:.0%}")
    if not has_ci:
        reasons.append("no confidence interval available (too few paired samples or legacy summary format)")
    elif not significant:
        reasons.append(f"cost delta 95% CI [{ci_low:+.6f}, {ci_high:+.6f}] includes zero")
    if marker_dropped:
        reasons.append(
            f"provider {provider!r} is known to drop the cache marker for "
            f"{model!r} — cache_hit_rate is not measurable on this path"
        )

    # `marker_dropped` blocks `verified` too: if the cache marker is known to
    # be stripped on this path, any cost delta we observe isn't attributable
    # evidence of ContextOps' reordering effect, even if it's statistically
    # significant — the mechanism this tool relies on structurally can't
    # operate here.
    verified = not low_n and not high_error_rate and significant and not marker_dropped

    return {
        "n": n,
        "error_rate": round(error_rate, 3),
        "low_n": low_n,
        "high_error_rate": high_error_rate,
        "has_ci": has_ci,
        "significant": significant,
        "cache_marker_dropped": marker_dropped,
        "verified": verified,
        "reasons": reasons,
    }
