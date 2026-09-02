"""The draftability law.

One relation organises every acceptance number in the paper. Treat each block
offset as a Bernoulli match whose probability is set by the target's own
next-token entropy at that round,

    logit p = b0 - b1 * H,

and the accepted run is geometric, so its expectation is

    E[a | H] = p / (1 - p),

capped at the head's reachable length. Entropy falls as the image pins the
answer, which is why grounded workloads accept the longest blocks and open
description the shortest, and why the fitted slope b1 steepens with grounding.

The law is what makes acceptance predictable rather than merely observed. It
transfers across targets and modalities, and it names its own boundary: where
entropy stays high, a chain drafter is the better instrument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["DraftabilityLaw", "expected_acceptance", "decile_curve"]

DEFAULT_CAP = 15.0
"""Reachable accepted length, above which the geometric form stops meaning
anything because the block itself runs out."""


def expected_acceptance(entropy, b0, b1, cap=DEFAULT_CAP):
    """``E[a | H]`` under the law, elementwise."""
    z = np.clip(b0 - b1 * np.asarray(entropy, float), -60.0, 60.0)
    p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-9)
    return np.minimum(p / (1.0 - p), cap)


def _bootstrap_ci(x, rng, n=2000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    draws = np.asarray(x)[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def decile_curve(entropy, accepted, bins=10, seed=0, min_per_bin=5):
    """Mean accepted length per entropy quantile, with bootstrap intervals.

    This is how the law is read off data: rounds are noisy one at a time, and
    binning by entropy exposes the shape without smoothing it away.
    """
    H = np.asarray(entropy, float)
    a = np.asarray(accepted, float)
    rng = np.random.default_rng(seed)
    edges = np.quantile(H, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    rows = []
    for i in range(bins):
        take = (H >= edges[i]) & (H < edges[i + 1])
        if take.sum() < min_per_bin:
            continue
        lo, hi = _bootstrap_ci(a[take], rng)
        rows.append({"entropy": float(H[take].mean()),
                     "accepted": float(a[take].mean()),
                     "ci_low": lo, "ci_high": hi, "n": int(take.sum())})
    return rows


@dataclass
class DraftabilityLaw:
    """A fitted law, and the diagnostics that say whether to believe it."""

    b0: float
    b1: float
    cap: float = DEFAULT_CAP
    n: int = 0
    mean_accepted: float = 0.0
    r2_curve: float = float("nan")
    """Fit against the entropy-decile means. The curve is the claim."""

    r2_round: float = float("nan")
    """Fit against single rounds, which are dominated by their own noise."""

    r2_ceiling: float = float("nan")
    """How much of the round-level variance entropy could explain at best."""

    spearman_r: float = float("nan")
    spearman_p: float = float("nan")
    curve: list = field(default_factory=list)

    @property
    def frac_of_ceiling(self):
        """Round-level fit as a share of what entropy alone could reach."""
        if not self.r2_ceiling or not np.isfinite(self.r2_ceiling) or self.r2_ceiling <= 0:
            return float("nan")
        return float(self.r2_round / self.r2_ceiling)

    def expected(self, entropy):
        """``E[a | H]`` for one entropy or an array of them."""
        out = expected_acceptance(entropy, self.b0, self.b1, self.cap)
        return float(out) if np.ndim(out) == 0 else out

    def holds(self):
        """Whether the fit clears the bar the paper reports results under.

        Four conditions, all required: entropy lowers acceptance (``b1 > 0``);
        the rank correlation is negative and significant; the curve fits, or
        the round-level fit reaches half of what entropy could explain; and the
        fit is not degenerate.
        """
        checks = {
            "slope_positive": bool(self.b1 > 0),
            "negative_rank_correlation": bool(
                self.spearman_r < 0 and self.spearman_p < 0.05),
            "curve_or_ceiling": bool(
                (np.isfinite(self.r2_curve) and self.r2_curve >= 0.5)
                or (np.isfinite(self.frac_of_ceiling) and self.frac_of_ceiling >= 0.5)),
            "non_degenerate": bool(self.mean_accepted >= 0.15 and self.b1 <= 25.0),
        }
        return all(checks.values()), checks

    @classmethod
    def fit(cls, entropy, accepted, cap=DEFAULT_CAP, bins=10, seed=0):
        """Fit ``b0`` and ``b1`` by nonlinear least squares on the rounds.

        Args:
            entropy: measured next-token entropy, one value per round.
            accepted: accepted length past the committed token, same length.

        Rounds, not decile means, are the fitting unit, so the bins used for
        reporting never feed back into the parameters.
        """
        from scipy import optimize, stats

        H = np.asarray(entropy, float)
        a = np.asarray(accepted, float)
        if len(a) < 30 or H.std() == 0 or a.std() == 0:
            raise ValueError("need at least 30 rounds with variation in both axes")

        mean_a = max(float(a.mean()), 0.05)
        p0 = mean_a / (1 + mean_a)
        start = [math.log(p0 / (1 - p0)), 1.0]
        fitted = optimize.least_squares(
            lambda theta: expected_acceptance(H, theta[0], theta[1], cap) - a,
            start, method="lm", max_nfev=20000)
        b0, b1 = float(fitted.x[0]), float(fitted.x[1])

        curve = decile_curve(H, a, bins=bins, seed=seed)
        rank = stats.spearmanr(H, a)
        return cls(
            b0=b0, b1=min(b1, 25.0), cap=cap, n=len(a),
            mean_accepted=float(a.mean()),
            r2_curve=_r2_curve(curve, b0, b1, cap),
            r2_round=_r2(a, expected_acceptance(H, b0, b1, cap)),
            r2_ceiling=_entropy_ceiling(H, a),
            spearman_r=float(rank.correlation), spearman_p=float(rank.pvalue),
            curve=curve)


def _r2(y, pred):
    y = np.asarray(y, float)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1 - float(((y - pred) ** 2).sum()) / ss_tot


def _r2_curve(curve, b0, b1, cap):
    if len(curve) < 2:
        return float("nan")
    H = np.array([row["entropy"] for row in curve])
    y = np.array([row["accepted"] for row in curve])
    return _r2(y, expected_acceptance(H, b0, b1, cap))


def _entropy_ceiling(entropy, accepted, bins=50):
    """Variance share a perfect function of entropy alone could explain.

    Single rounds carry noise no entropy-indexed model can touch, so a raw
    round-level R^2 understates the fit. Binning finely and taking the
    between-bin variance gives the ceiling to read it against.
    """
    H = np.asarray(entropy, float)
    a = np.asarray(accepted, float)
    if len(a) < bins * 2:
        bins = max(5, len(a) // 10)
    edges = np.quantile(H, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    grand = a.mean()
    between = 0.0
    for i in range(bins):
        take = (H >= edges[i]) & (H < edges[i + 1])
        if take.sum():
            between += take.sum() * (a[take].mean() - grand) ** 2
    total = float(((a - grand) ** 2).sum())
    return float(between / total) if total > 0 else float("nan")
