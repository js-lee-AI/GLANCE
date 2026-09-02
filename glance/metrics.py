"""Reading a run: acceptance, speedup, and whether a gap is real.

Three conventions are worth stating, because they decide numbers and are easy
to get silently wrong.

*Warm-up.* The first sample of a run pays for allocator growth and, where the
engine captures CUDA graphs, for the capture itself. Every timing here drops
it, which is why a hundred reported samples are measured as a hundred and one.

*Pooling.* A task's accepted length pools rounds, not prompts. Prompts differ
in how many rounds they take, and averaging per prompt would quietly weight a
short prompt's rounds more heavily than a long one's.

*Aggregation.* A speedup is a ratio, so tasks combine by geometric mean, and
always against each system's own autoregressive baseline on the same card.
Ratios taken against someone else's baseline do not compare.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["drop_warmup", "acceptance_length", "geomean", "speedup",
           "paired_bootstrap"]


def drop_warmup(samples, n=1):
    """Drop the first ``n`` samples of a run."""
    return list(samples)[n:]


def acceptance_length(rounds):
    """Mean accepted length per round, pooled over rounds.

    Args:
        rounds: accepted length of each round, counting the committed token,
            so a round that accepts nothing scores 1 rather than 0.

    This is the metric that survives a change of engine. Wall-clock does not:
    it moves with the kernel, the card and the batch, which is why the paper
    reports both and compares systems on this one.
    """
    rounds = list(rounds)
    if not rounds:
        return float("nan")
    return float(sum(rounds)) / len(rounds)


def geomean(values):
    """Geometric mean, the right average for a set of ratios."""
    values = [float(v) for v in values]
    if not values or any(v <= 0 for v in values):
        return float("nan")
    return float(math.exp(sum(math.log(v) for v in values) / len(values)))


def speedup(baseline_ms_per_token, ms_per_token):
    """Ratio against the baseline measured on the same card, in the same run."""
    return float(baseline_ms_per_token) / float(ms_per_token)


def paired_bootstrap(a, b, n=10000, alpha=0.05, seed=0):
    """Confidence interval on the paired mean difference ``a - b``.

    Prompts are resampled, not measurements, and each resample keeps both arms
    of a prompt together, so the interval carries the pairing that the protocol
    already bought by running the arms back to back on one card.

    Returns:
        ``(mean, low, high)``. The gap is worth reporting when the interval
        excludes zero.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms must have equal length, got {a.shape} and {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    draws = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return (float(diff.mean()),
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))
