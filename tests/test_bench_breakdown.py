"""Tests for ``contextops_bench.breakdown`` — per-prompt A/B summary."""

from __future__ import annotations

import csv

from contextops_bench.breakdown import (
    COLUMNS,
    per_prompt_breakdown,
    render_breakdown_table,
    save_breakdown_csv,
)
from contextops_bench.clients import BenchResult


def make_result(
    prompt_id: int,
    *,
    model: str = "m",
    use_optimized: bool = True,
    prompt_tokens: int = 100,
    cached_tokens: int = 0,
    cost_usd: float = 0.0,
    error: str = "",
) -> BenchResult:
    """Shorthand constructor for tests."""
    return BenchResult(
        prompt_id=prompt_id,
        model=model,
        provider="echo",
        use_optimized=use_optimized,
        prompt_tokens=prompt_tokens,
        completion_tokens=10,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        error=error,
    )


class TestPerPromptBreakdown:
    def test_empty_inputs_returns_empty_list(self):
        assert per_prompt_breakdown([], []) == []

    def test_basic_pair_extraction(self):
        """One paired prompt → one row with all expected fields populated."""
        opt = [make_result(0, cost_usd=0.001, cached_tokens=80)]
        base = [make_result(0, cost_usd=0.002, cached_tokens=10)]
        rows = per_prompt_breakdown(opt, base)
        assert len(rows) == 1
        r = rows[0]
        assert r["prompt_id"] == 0
        assert r["model"] == "m"
        assert r["prompt_tokens"] == 100
        assert r["baseline_cost"] == 0.002
        assert r["optimized_cost"] == 0.001
        assert r["delta_cost"] == -0.001  # optimized cheaper
        # delta_pct = -0.001 / 0.002 * 100 = -50.0%
        assert r["delta_pct"] == -50.0
        # cache_hit = cached / (cached + prompt). denom != 0 → use formula.
        assert r["baseline_cache_hit"] == round(10 / (10 + 100), 3)
        assert r["optimized_cache_hit"] == round(80 / (80 + 100), 3)

    def test_sorts_by_abs_delta_cost_descending(self):
        """Top row has the largest |Δ cost| regardless of sign."""
        # pid 0: small delta; pid 1: huge negative; pid 2: medium positive
        opt = [
            make_result(0, cost_usd=0.0010),  # base 0.0011 → delta = -0.0001
            make_result(1, cost_usd=0.0001),  # base 0.0100 → delta = -0.0099
            make_result(2, cost_usd=0.0100),  # base 0.0050 → delta = +0.0050
        ]
        base = [
            make_result(0, cost_usd=0.0011),
            make_result(1, cost_usd=0.0100),
            make_result(2, cost_usd=0.0050),
        ]
        rows = per_prompt_breakdown(opt, base)
        ids = [r["prompt_id"] for r in rows]
        # |0.0099| > |0.0050| > |0.0001| → 1, 2, 0
        assert ids == [1, 2, 0]

    def test_top_n_truncates(self):
        """Output has at most top_n rows even when more prompts are available."""
        opt = [make_result(i, cost_usd=0.001 * i) for i in range(20)]
        base = [make_result(i, cost_usd=0.002 * i) for i in range(20)]
        rows = per_prompt_breakdown(opt, base, top_n=3)
        assert len(rows) == 3

    def test_skips_prompt_ids_present_in_only_one_arm(self):
        """A prompt present only in optimized (or only in baseline) is dropped."""
        opt = [make_result(0), make_result(1), make_result(99)]   # 99 missing from base
        base = [make_result(0), make_result(1)]
        rows = per_prompt_breakdown(opt, base)
        ids = {r["prompt_id"] for r in rows}
        assert ids == {0, 1}
        assert 99 not in ids

    def test_skips_rows_with_errors_in_either_arm(self):
        """Either arm errored → drop the pair (cost/cache not meaningful)."""
        opt = [make_result(0), make_result(1, error="boom")]
        base = [make_result(0, error="explode"), make_result(1)]
        rows = per_prompt_breakdown(opt, base)
        # Both pairs have at least one error → empty
        assert rows == []

    def test_zero_baseline_cost_yields_zero_delta_pct(self):
        """% comparison undefined when baseline_cost == 0 → return 0.0 not inf."""
        opt = [make_result(0, cost_usd=0.001)]
        base = [make_result(0, cost_usd=0.0)]
        rows = per_prompt_breakdown(opt, base)
        assert rows[0]["delta_cost"] == 0.001
        assert rows[0]["delta_pct"] == 0.0


class TestBreakdownCSV:
    def test_save_writes_header_and_rows(self, tmp_path):
        rows = [
            {
                "prompt_id": 0,
                "model": "echo-model",
                "prompt_tokens": 100,
                "baseline_cost": 0.001,
                "optimized_cost": 0.0005,
                "delta_cost": -0.0005,
                "delta_pct": -50.0,
                "baseline_cache_hit": 0.0,
                "optimized_cache_hit": 0.5,
            }
        ]
        path = tmp_path / "out" / "test.breakdown.csv"
        save_breakdown_csv(rows, path)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == list(COLUMNS)
            data = list(reader)
            assert len(data) == 1
            assert data[0]["prompt_id"] == "0"   # CSV writes ints as strings

    def test_save_empty_rows_writes_header_only(self, tmp_path):
        path = tmp_path / "test.breakdown.csv"
        save_breakdown_csv([], path)
        with path.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == list(COLUMNS)
            assert list(reader) == []


class TestRender:
    def test_returns_empty_string_for_empty_rows(self):
        assert render_breakdown_table([]) == ""

    def test_includes_header_and_rows(self):
        rows = [
            {
                "prompt_id": 0,
                "model": "echo-model",
                "prompt_tokens": 100,
                "delta_cost": -0.001,
                "delta_pct": -50.0,
                "baseline_cache_hit": 0.1,
                "optimized_cache_hit": 0.5,
            }
        ]
        out = render_breakdown_table(rows)
        assert "[BREAKDOWN]" in out
        assert "delta cost" in out
        assert "echo-model" in out
        # Format-string spot checks: cost padded, sign-pct formatted, percent cols.
        assert "-0.001000" in out
        assert "-50.0" in out
        assert "50%" in out  # optimized_cache_hit at 0.5
