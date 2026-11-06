"""Uncertainty for the views: Wilson intervals for proportions, and paired intervals for
CNN-vs-SNN differences on the same images (the two models see identical inputs, so their errors
are correlated and an unpaired interval would be too wide)."""
import numpy as np

Z = 1.96  # 95 %


def wilson(k, n, z=Z):
    """95 % Wilson interval for k successes out of n. Works on arrays."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return mid - half, mid + half


def paired_diff(a, b, z=Z):
    """Difference mean(b) - mean(a) of two paired 0/1 vectors, with a 95 % interval.

    Uses the discordant counts (McNemar-style Wald interval): only images where the two
    models differ carry information about the difference.
    Returns (diff, lo, hi, n).
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    n10 = np.sum(a & ~b)  # a yes, b no
    n01 = np.sum(~a & b)
    d = (n01 - n10) / n
    se = np.sqrt(max(n01 + n10 - (n01 - n10) ** 2 / n, 0)) / n
    return d, d - z * se, d + z * se, n
