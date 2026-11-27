"""k-nearest neighbours in AkidaNet's embedding space, per corruption x severity.

Source: `results/akidanet_imagenetc.parquet` (data owner, 30 000 rows = 2000 images x 15
conditions). Columns used: image_id, corruption, severity, true_label, cnn_pred, cnn_conf,
akida_pred, akida_conf, embedding (512 floats).

What the embedding is: a 512-d vector per image and condition, signed (about 40 % of values are
negative) and continuous (505 222 distinct values in the first 1000 rows), so it is neither a
post-ReLU feature nor a 4-bit quantized activation. The script that wrote the parquet is not in
the repo. The only AkidaNet extraction code in the repo (Phase 1 - No AI/view_embeddings.ipynb,
float model, `model.layers[-2].input`, non-negative) gives *different* vectors for the same
1500 image/condition rows: row-wise cosine median 0.118, same as a shuffled baseline (0.115).
So the source layer and model are unknown (question to Gabriel, TODO-yoshi.md). What is known:
it is semantic (same-class check in notebook 01). kNN here explains where *a* CNN-side
representation is ambiguous, not what the chip computes.

Neighbours are always searched *within one condition* (same corruption and severity), so
severity cannot act as a confounder (lesson from Phase I, AI record row 3).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
PARQUET = ROOT / "results" / "akidanet_imagenetc.parquet"
CORRUPTIONS = ["fog", "gaussian_noise", "contrast"]
SEVERITIES = [1, 2, 3, 4, 5]
K = 15  # same k as the Phase I group-purity view
SINK_SHARE = 0.05  # a class is a "sink" in a condition if it takes >= 5 % of CNN predictions


def load():
    d = pd.read_parquet(PARQUET)
    assert len(d) == 30_000 and d.groupby(["corruption", "severity"]).size().eq(2000).all()
    d["agree"] = d.cnn_pred == d.akida_pred
    d["disagree"] = ~d.agree
    return d.reset_index(drop=True)


def unit(E):
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def neighbours(g, k=K):
    """Cosine kNN inside one condition. Returns (idx, dist), self excluded. idx is positional."""
    E = unit(np.vstack(g.embedding.values))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(E)
    dist, idx = nn.kneighbors(E)
    # drop self (column 0 is self unless duplicates exist; remove by identity to be safe)
    keep = idx != np.arange(len(E))[:, None]
    idx = np.array([r[m][:k] for r, m in zip(idx, keep)])
    dist = np.array([r[m][:k] for r, m in zip(dist, keep)])
    return idx, dist, E


def sinks(g, share=SINK_SHARE):
    vc = g.cnn_pred.value_counts(normalize=True)
    return set(vc[vc >= share].index)


def per_image(d, k=K, seed=0):
    """Add per-image neighbourhood features, condition by condition.

    consensus   share of the k neighbours whose CNN answer equals this image's CNN answer
    snn_support share of the k neighbours whose CNN answer equals this image's SNN answer
    rand_support same, for a class drawn from the condition's CNN answer distribution
                 (excluding this image's CNN answer): a frequency-matched baseline for
                 snn_support, needed because sinks make popular classes common among neighbours
    n_unique    number of distinct CNN answers among the k neighbours
    knn_dist    mean cosine distance to the k neighbours
    label_nbr   share of neighbours with the same true label (embedding quality check)
    in_sink     this image's CNN answer is a sink class of the condition
    """
    rng = np.random.default_rng(seed)
    parts = []
    for (c, s), g in d.groupby(["corruption", "severity"], sort=False):
        g = g.copy()
        idx, dist, _ = neighbours(g, k)
        cp, ap, lab = g.cnn_pred.values, g.akida_pred.values, g.true_label.values
        nbr = cp[idx]
        classes, counts = np.unique(cp, return_counts=True)
        rand = np.empty(len(g), int)
        for i in range(len(g)):
            p = counts * (classes != cp[i])
            rand[i] = rng.choice(classes, p=p / p.sum())
        g["consensus"] = (nbr == cp[:, None]).mean(1)
        g["snn_support"] = (nbr == ap[:, None]).mean(1)
        g["rand_support"] = (nbr == rand[:, None]).mean(1)
        g["n_unique"] = [len(set(r)) for r in nbr]
        g["knn_dist"] = dist.mean(1)
        g["label_nbr"] = (lab[idx] == lab[:, None]).mean(1)
        g["in_sink"] = np.isin(cp, list(sinks(g)))
        parts.append(g)
    return pd.concat(parts).sort_index()


def spread(d):
    """Per condition: mean distance of unit embeddings to their centroid (1 = spread out,
    0 = all images at one point), and mean kNN distance."""
    rows = []
    for (c, s), g in d.groupby(["corruption", "severity"]):
        E = unit(np.vstack(g.embedding.values))
        rows.append(dict(corruption=c, severity=s,
                         spread=np.linalg.norm(E - E.mean(0), axis=1).mean()))
    return pd.DataFrame(rows)
