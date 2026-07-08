"""Tests for ``contextops_bench.stats`` — bootstrap CI + effect size.

These tests are intentionally value-driven (not implementation-driven) so they
remain valid if the bootstrap algorithm gets swapped for a faster one later.
"""

from __future__ import annotations

import pytest

from contextops_bench.stats import bootstrap_ci, effect_size_pct


class TestBootstrapCI:
    def test_empty_returns_zeros(self):
        """Empty input collapses to a neutral default."""
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_identical_values_collapse_to_point(self):
        """All paired deltas equal → resampled mean is always that value → CI is a point."""
        vals = [0.001] * 30
        low, high = bootstrap_ci(vals, n_boot=2_000, seed=42)
        assert low == high == pytest.approx(0.001, abs=1e-6)

    def test_clear_difference_excludes_zero(self):
        """Optimized much cheaper than baseline → CI on (opt − base) is all-negative.

        This is the regression-style test that the bench-quality v0.3.3 plan
        promised: at n=50 with a 10× cost gap, the bootstrap CI must not
        straddle zero. Otherwise the headline "X% cheaper" claim is unreliable.
        """
        deltas = [-0.009] * 50  # 50 paired observations, optimized is $0.009 cheaper each
        low, high = bootstrap_ci(deltas, n_boot=5_000, seed=42)
        assert high < 0.0
        assert low < 0.0

    def test_symmetric_around_zero_straddles_zero(self):
        """Paired noise around zero → CI should bracket 0 (no detectable effect)."""
        # Alternating ±0.001 averages to ~0 under any resample → CI close to zero.
        vals = [0.001, -0.001] * 25
        low, high = bootstrap_ci(vals, n_boot=2_000, seed=42)
        assert low <= 0.0 <= high

    def test_deterministic_via_seed(self):
        """Same seed + same input → identical output."""
        vals = [0.01, 0.02, 0.03, 0.04, 0.05] * 10
        assert bootstrap_ci(vals, seed=42) == bootstrap_ci(vals, seed=42)

    def test_caller_can_set_ci(self):
        """CI parameter actually changes the bound width."""
        # n=200 with no spread → CI bounds at any level should be near mean.
        # Wider CI requested → bounds should be further from the center.
        vals = [1.0] * 100 + [0.0] * 100  # mean ≈ 0.5
        ci_50 = bootstrap_ci(vals, ci=0.50, n_boot=2_000, seed=42)
        ci_99 = bootstrap_ci(vals, ci=0.99, n_boot=2_000, seed=42)
        assert (ci_99[1] - ci_99[0]) > (ci_50[1] - ci_50[0])


class TestEffectSizePct:
    def test_negative_when_optimized_cheaper(self):
        """optimized = 0.001, baseline = 0.01 → med_diff = −0.009 → −90.0%."""
        opt = [0.001] * 20
        base = [0.01] * 20
        # median(opt − base) = -0.009; median(base) = 0.01; ratio = -0.9 = -90.0%
        assert effect_size_pct(opt, base) == pytest.approx(-90.0, abs=0.01)

    def test_zero_when_equivalent(self):
        """Same values → median diff = 0 → 0.0% effect."""
        assert effect_size_pct([0.005] * 20, [0.005] * 20) == 0.0

    def test_zero_when_either_empty(self):
        """Defensive: empty inputs do not blow up."""
        assert effect_size_pct([], []) == 0.0
        assert effect_size_pct([0.1], []) == 0.0
        assert effect_size_pct([], [0.1]) == 0.0

    def test_zero_when_baseline_median_is_zero(self):
        """% comparison is undefined when baseline median is 0 → return 0.0 instead of inf/NaN."""
        opt = [0.001, 0.002, 0.003]
        base = [0.0, 0.0, 0.0]
        assert effect_size_pct(opt, base) == 0.0

    def test_positive_when_optimized_more_expensive(self):
        """Sanity-check the sign: opt > base → positive effect size (bad for cost)."""
        opt = [0.01] * 10
        base = [0.001] * 10
        # median(opt − base) = +0.009; median(base) = 0.001; ratio = +9.0 = +900.0%
        assert effect_size_pct(opt, base) == pytest.approx(900.0, abs=0.01)
