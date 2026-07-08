"""Statistical helpers for A/B bench measurements.

Stdlib only — no scipy, no numpy. Keeps the bench self-contained and the
install footprint flat. Functions are pure: no globals, deterministic when
seeded, easy to test.

These helpers back the `cost_delta_ci_low_usd` / `cost_delta_ci_high_usd` /
`effect_size_pct` fields in bench summaries (`contextops_bench.runner.summarize`).
"""

from __future__ import annotations

import random
import statistics


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Return ``(low, high)`` percentile bounds of the bootstrap distribution of the mean.

    Algorithm:
      1. Resample ``values`` with replacement ``n_boot`` times.
      2. Compute the mean of each resample.
      3. Sort the bootstrap means.
      4. Return the ``(1 - ci) / 2`` and ``1 - (1 - ci) / 2`` percentile cuts.

    Stdlib only (``random``, no third-party stats libs). Deterministic given
    the same ``seed``. Returned values are rounded to 6 decimal places to
    match the precision of the rest of the bench's USD figures.

    Edge cases:
      - Empty input → ``(0.0, 0.0)``
      - Single value or all-identical values → collapses to that point
      - When ``len(values) < 20``, ``n_boot`` auto-scales down to ``max(1_000,
        n_boot // 10)`` — running 10k resamples against N<20 is wasteful work.
    """
    if not values:
        return (0.0, 0.0)

    n = len(values)
    if n == 1:
        return (float(values[0]), float(values[0]))

    # All-identical short-circuit: every resample mean equals that value.
    first = values[0]
    if all(v == first for v in values):
        return (float(first), float(first))

    effective_n_boot = n_boot if n >= 20 else max(1_000, n_boot // 10)

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(effective_n_boot):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()

    alpha = (1.0 - ci) / 2.0
    # Clamp indices defensively against float fuzz near 0 / 1.
    lo_idx = max(0, min(int(alpha * effective_n_boot), effective_n_boot - 1))
    hi_idx = max(0, min(int((1.0 - alpha) * effective_n_boot), effective_n_boot - 1))
    return (round(boot_means[lo_idx], 6), round(boot_means[hi_idx], 6))


def effect_size_pct(
    optimized: list[float],
    baseline: list[float],
) -> float:
    """Return ``median(optimized − baseline) / median(baseline) × 100``.

    Robust to skewed cost distributions (median, not mean). Negative means
    the optimized arm is cheaper — the desired outcome for this bench.

    Pairs ``optimized[i]`` with ``baseline[i]`` by index; the two lists must
    be the same length and correspond to the same prompts in the same order.
    The runner guarantees this by zipping on ``prompt_id``.

    Edge cases:
      - Either list empty → ``0.0`` (no comparison possible)
      - ``median(baseline) == 0`` → ``0.0`` (percentage undefined)
    """
    if not optimized or not baseline:
        return 0.0
    paired = list(zip(optimized, baseline))
    if not paired:
        return 0.0
    paired_diffs = [o - b for o, b in paired]
    med_diff = statistics.median(paired_diffs)
    med_base = statistics.median(baseline)
    if med_base == 0.0:
        return 0.0
    return round((med_diff / med_base) * 100.0, 2)
