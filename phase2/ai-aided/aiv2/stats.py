"""Uncertainty helpers for Phase II. AUROC with a bootstrap interval over images, and a
paired bootstrap for the difference of two AUROCs on the same images."""
import numpy as np
from sklearn.metrics import roc_auc_score

B = 1000


def auroc_ci(y, score, b=B, seed=0):
    """AUROC of `score` for the positive class `y`, with a 95 % percentile bootstrap interval."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, score)
    n = len(y)
    boots = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        if y[i].all() or (~y[i]).all():
            continue
        boots.append(roc_auc_score(y[i], score[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, lo, hi


def auroc_diff_ci(y, s1, s2, b=B, seed=0):
    """AUROC(s2) - AUROC(s1) on the same images, paired bootstrap 95 % interval."""
    y = np.asarray(y, bool)
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    rng = np.random.default_rng(seed)
    d = roc_auc_score(y, s2) - roc_auc_score(y, s1)
    n = len(y)
    boots = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        if y[i].all() or (~y[i]).all():
            continue
        boots.append(roc_auc_score(y[i], s2[i]) - roc_auc_score(y[i], s1[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d, lo, hi
