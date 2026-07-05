"""Small statistics helpers.

A run of 8 (or even 50) cases is a *sample*. Reporting a bare point estimate
for coverage or failure rate invites over-reading noise as signal, so the
harness reports a Wilson score interval alongside every proportion.

Wilson is preferred over the normal (Wald) interval because it behaves at the
extremes that matter most here — a 0/137 never-event count still yields a
non-trivial upper bound, whereas Wald collapses to [0, 0]. (Newcombe RG,
Stat Med 1998;17:857-872.)
"""
from __future__ import annotations

from collections import Counter
from math import sqrt


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: number of positive events.
        n: number of trials.
        z: standard-normal quantile (1.96 -> 95%).

    Returns:
        (low, high), each clamped to [0, 1]. Returns (0.0, 0.0) for n == 0.
    """
    if n <= 0:
        return (0.0, 0.0)

    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def gwet_ac1(pairs: list[tuple[str, str]], categories: list[str] | None = None) -> float:
    """Gwet's AC1 agreement coefficient for two raters over categorical labels.

    Preferred over Cohen/Fleiss κ when the categories are skewed (as clinical safety
    labels always are): under high prevalence of one class, κ collapses toward zero even
    at high raw agreement (the "kappa paradox", Feinstein & Cicchetti 1990), whereas AC1
    stays interpretable. `pairs` is a list of (rater1_label, rater2_label) — here,
    (categorizer_prediction, human_expected).

        AC1 = (Pa - Pe) / (1 - Pe),  Pe = 1/(K-1) * Σ_k π_k (1 - π_k)

    where Pa is observed agreement and π_k is the prevalence of category k across both
    raters. Returns a value in roughly [-1, 1] (1 = perfect). Ref: Gwet 2008.
    """
    n = len(pairs)
    if n == 0:
        return 0.0

    cats = categories if categories is not None else sorted({c for p in pairs for c in p})
    k = len(cats)
    pa = sum(1 for a, b in pairs if a == b) / n
    if k <= 1:
        return 1.0 if pa == 1.0 else 0.0

    counts: Counter = Counter()
    for a, b in pairs:
        counts[a] += 1
        counts[b] += 1
    total = 2 * n
    pi = {c: counts.get(c, 0) / total for c in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (k - 1)
    if pe >= 1.0:
        return 1.0 if pa == 1.0 else 0.0
    return (pa - pe) / (1 - pe)
