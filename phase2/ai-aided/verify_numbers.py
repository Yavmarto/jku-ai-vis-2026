"""Recompute the headline numbers of findings/01.md without the aiv2 package or sklearn kNN.

Brute-force cosine neighbours with numpy and a rank-based AUROC (Mann-Whitney U), so a bug in
the notebook path does not repeat here. Prints one table; compare it with findings/01.md.

    python phase2/ai-aided/verify_numbers.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
K = 15


def auroc(pos, score):
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


d = pd.read_parquet(ROOT / "results" / "akidanet_imagenetc.parquet")
csv = pd.read_csv(ROOT / "Phase 1 - No AI" / "results_card" / "predictions.csv")
m = d.merge(csv, on=["image_id", "corruption", "severity"], suffixes=("", "_csv"))
assert len(m) == 30_000
assert (m.cnn_pred == m.keras_pred).all() and (m.akida_pred == m.akida_pred_csv).all()

rows = []
for (c, s), g in d.groupby(["corruption", "severity"]):
    E = np.vstack(g.embedding.values)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argsort(-S, axis=1)[:, :K]
    cp, ap, lab = g.cnn_pred.values, g.akida_pred.values, g.true_label.values
    dis = cp != ap
    cons = (cp[idx] == cp[:, None]).mean(1)
    sup = (cp[idx] == ap[:, None]).mean(1)
    vc = pd.Series(cp).value_counts(normalize=True)
    sink = np.isin(cp, vc[vc >= 0.05].index)
    knn_dist = (1 - np.take_along_axis(S, idx, 1)).mean(1)
    rows.append(dict(
        cond=f"{c[:5]} {s}", disagree=dis.mean(),
        partner_in_knn=(lab[idx] == lab[:, None]).any(1).mean(),
        spread=np.linalg.norm(E - E.mean(0), axis=1).mean(),
        sink_share=sink.mean(),
        dist_sink=knn_dist[sink].mean() if sink.any() else np.nan,
        dist_rest=knn_dist[~sink].mean(),
        auc_conf=auroc(dis, -g.cnn_conf.values),
        auc_cons=auroc(dis, -cons),
        snn_support_dis=sup[dis].mean(),
    ))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
