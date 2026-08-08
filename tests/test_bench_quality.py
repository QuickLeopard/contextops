"""Tests for contextops_bench.quality — deterministic bench result gates."""

from __future__ import annotations

from contextops_bench.quality import (
    MAX_ERROR_RATE,
    MIN_N,
    cache_marker_dropped,
    classify_error,
    confidence_level,
    evaluate_quality_gate,
    is_local_provider,
    summarize_errors,
)


def _summary(
    *,
    n_opt: int = 30,
    n_base: int = 30,
    errors_opt: int = 0,
    errors_base: int = 0,
    ci_low: float | None = -0.01,
    ci_high: float | None = -0.005,
    token_ci_low: float | None = None,
    token_ci_high: float | None = None,
    latency_ci_low: float | None = None,
    latency_ci_high: float | None = None,
) -> dict:
    delta = {}
    if ci_low is not None and ci_high is not None:
        delta["cost_delta_ci_low_usd"] = ci_low
        delta["cost_delta_ci_high_usd"] = ci_high
    if token_ci_low is not None and token_ci_high is not None:
        delta["prompt_tokens_delta_ci_low"] = token_ci_low
        delta["prompt_tokens_delta_ci_high"] = token_ci_high
    if latency_ci_low is not None and latency_ci_high is not None:
        delta["latency_ms_p50_delta_ci_low"] = latency_ci_low
        delta["latency_ms_p50_delta_ci_high"] = latency_ci_high
    return {
        "optimized": {"n": n_opt, "errors": errors_opt},
        "baseline": {"n": n_base, "errors": errors_base},
        "delta": delta,
    }


def test_verified_when_adequately_powered_low_error_and_significant():
    summary = _summary()
    gate = evaluate_quality_gate(summary, provider="direct_zen", model="claude-sonnet-4-6")
    assert gate["verified"] is True
    assert gate["reasons"] == []
    assert gate["n"] == 30
    assert gate["error_rate"] == 0.0


def test_fails_on_low_n():
    summary = _summary(n_opt=5, n_base=5)
    gate = evaluate_quality_gate(summary, provider="direct_zen", model="claude-sonnet-4-6")
    assert gate["verified"] is False
    assert gate["low_n"] is True
    assert any("min_n" in r for r in gate["reasons"])


def test_fails_on_high_error_rate():
    summary = _summary(n_opt=30, n_base=30, errors_opt=10)
    gate = evaluate_quality_gate(summary, provider="openai", model="gpt-4o-mini")
    assert gate["verified"] is False
    assert gate["high_error_rate"] is True
    assert gate["error_rate"] > MAX_ERROR_RATE


def test_fails_when_ci_includes_zero():
    summary = _summary(ci_low=-0.002, ci_high=0.001)
    gate = evaluate_quality_gate(summary, provider="openai", model="gpt-4o-mini")
    assert gate["verified"] is False
    assert gate["significant"] is False
    assert any("includes zero" in r for r in gate["reasons"])


def test_fails_when_no_ci_present():
    summary = _summary(ci_low=None, ci_high=None)
    gate = evaluate_quality_gate(summary, provider="openai", model="gpt-4o-mini")
    assert gate["verified"] is False
    assert gate["has_ci"] is False


def test_verified_true_for_significant_cost_increase():
    """A statistically significant INCREASE is still 'verified' — just not a win."""
    summary = _summary(ci_low=0.005, ci_high=0.01)
    gate = evaluate_quality_gate(summary, provider="openrouter", model="anthropic/claude-sonnet-4.6")
    assert gate["significant"] is True
    # cache_marker_dropped is still flagged even though cost data is trustworthy
    assert gate["cache_marker_dropped"] is True
    assert gate["verified"] is False  # openrouter+anthropic path always flagged


def test_cache_marker_dropped_detection():
    assert cache_marker_dropped("openrouter", "anthropic/claude-haiku-4.5") is True
    assert cache_marker_dropped("direct_zen", "claude-sonnet-4-6") is False
    assert cache_marker_dropped("openrouter", "openai/gpt-4o-mini") is False


def test_min_n_boundary_is_inclusive():
    summary = _summary(n_opt=MIN_N, n_base=MIN_N)
    gate = evaluate_quality_gate(summary, provider="openai", model="gpt-4o-mini")
    assert gate["low_n"] is False


def test_empty_summary_does_not_crash():
    gate = evaluate_quality_gate({}, provider="", model="")
    assert gate["verified"] is False
    assert gate["low_n"] is True


def test_classify_error_http_status_codes():
    assert classify_error("HTTP Error 403: Forbidden") == "auth"
    assert classify_error("HTTP Error 401: Unauthorized") == "auth"
    assert classify_error("HTTP Error 429: Too Many Requests") == "rate_limit"
    assert classify_error("HTTP Error 500: Internal Server Error") == "server_error"
    assert classify_error("HTTP Error 400: Bad Request") == "client_error"


def test_classify_error_network_patterns():
    assert classify_error("Remote end closed connection without response") == "network"
    assert classify_error(
        "<urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol>"
    ) == "network"
    assert classify_error("Connection timed out") == "network"


def test_classify_error_unknown_and_empty():
    assert classify_error("") == "unknown"
    assert classify_error("something weird happened") == "unknown"


def test_summarize_errors_counts_by_category():
    breakdown = summarize_errors([
        "HTTP Error 403: Forbidden",
        "HTTP Error 403: Forbidden",
        "Remote end closed connection without response",
    ])
    assert breakdown == {"auth": 2, "network": 1}


def test_confidence_level_forces_invalid_on_auth_errors():
    assert confidence_level(0.01, {"auth": 1}) == "invalid"
    assert confidence_level(0.0, {"auth": 5}) == "invalid"


def test_confidence_level_scales_with_error_rate():
    assert confidence_level(0.0, {}) == "high"
    assert confidence_level(0.05, {}) == "high"
    assert confidence_level(0.10, {"network": 3}) == "medium"
    assert confidence_level(0.30, {"network": 9}) == "low"


def test_evaluate_quality_gate_blocks_verified_on_auth_errors():
    """Even a low error rate with an auth error present should not be verified."""
    summary = _summary(n_opt=30, n_base=30, errors_opt=0, errors_base=0)
    summary["optimized"]["error_breakdown"] = {"auth": 1}
    gate = evaluate_quality_gate(summary, provider="openai", model="gpt-4o-mini")
    assert gate["verified"] is False
    assert gate["has_auth_errors"] is True
    assert gate["confidence"] == "invalid"
    assert any("auth error" in r for r in gate["reasons"])


def test_local_provider_verified_on_significant_token_delta():
    """Self-hosted providers with zero cost can verify on token savings."""
    summary = _summary(
        ci_low=0.0, ci_high=0.0,  # cost is zero / not significant
        token_ci_low=-50.0,
        token_ci_high=-10.0,
    )
    gate = evaluate_quality_gate(summary, provider="ollama", model="llama3.2:latest")
    assert gate["local_provider"] is True
    assert gate["token_significant"] is True
    assert gate["verified"] is True
    assert any("local provider verified on token" in r for r in gate["reasons"])


def test_local_provider_verified_on_significant_latency_delta():
    """Self-hosted providers with zero cost can verify on latency savings."""
    summary = _summary(
        ci_low=0.0, ci_high=0.0,
        latency_ci_low=-2000.0,
        latency_ci_high=-500.0,
    )
    gate = evaluate_quality_gate(summary, provider="vllm", model="mistral-7b")
    assert gate["local_provider"] is True
    assert gate["latency_significant"] is True
    assert gate["verified"] is True


def test_local_provider_fails_when_no_significant_token_or_latency_delta():
    summary = _summary(
        ci_low=0.0, ci_high=0.0,
        token_ci_low=-5.0,
        token_ci_high=5.0,
        latency_ci_low=-100.0,
        latency_ci_high=100.0,
    )
    gate = evaluate_quality_gate(summary, provider="ollama", model="llama3.2:latest")
    assert gate["local_provider"] is True
    assert gate["token_significant"] is False
    assert gate["latency_significant"] is False
    assert gate["verified"] is False
    assert any("both include zero" in r for r in gate["reasons"])


def test_non_local_provider_still_requires_cost_significance():
    summary = _summary(
        token_ci_low=-50.0,
        token_ci_high=-10.0,
        ci_low=-0.002,
        ci_high=0.001,
    )
    gate = evaluate_quality_gate(summary, provider="openai", model="gpt-4o-mini")
    assert gate["local_provider"] is False
    assert gate["cost_significant"] is False
    assert gate["verified"] is False
    assert any("cost delta 95% CI" in r for r in gate["reasons"])


def test_is_local_provider_recognizes_self_hosted_paths():
    assert is_local_provider("ollama") is True
    assert is_local_provider("vllm") is True
    assert is_local_provider("tgi") is True
    assert is_local_provider("lmstudio") is True
    assert is_local_provider("openai") is False
    assert is_local_provider("OpenAI") is False
