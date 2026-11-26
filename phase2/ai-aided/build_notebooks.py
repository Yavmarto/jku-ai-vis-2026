"""Generates the Phase II notebook (cells kept as code so they are reviewable in diffs).
Findings text lives in findings/01.md and is pasted into the last cell.

    python phase2/ai-aided/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def findings(name):
    p = HERE / "findings" / f"{name}.md"
    return p.read_text() if p.exists() else "## Findings\n\n_Not written yet._"


SETUP = '''import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path.cwd()))
from aiv2 import knn
from aiv2.stats import auroc_ci, auroc_diff_ci

FIG = knn.ROOT / "figures" / "phase2_ai"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"fog": "#4C78A8", "gaussian_noise": "#E45756", "contrast": "#54A24B"}
pd.set_option("display.width", 160)

d = knn.load()
f = knn.per_image(d)          # neighbourhood features per image and condition
print(f"{len(f)} rows, k = {knn.K}, sink = class with >= {knn.SINK_SHARE:.0%} of CNN answers")
'''

CHECK = '''# Is the embedding semantic at all? Each class has 2 images per condition, so at most one
# same-label neighbour exists. Chance of it being among 15 of 1999: 15/1999.
chk = f.groupby(["corruption", "severity"]).label_nbr.mean().unstack() * knn.K
chk.columns = [f"sev {s}" for s in chk.columns]
print("Share of images whose same-class partner is among its 15 neighbours")
print(chk.round(3).to_string())
print(f"chance: {knn.K / 1999:.4f}")
'''

C2 = '''# C2 (a): embedding spread per condition. (b): sinks are dense regions.
sp = knn.spread(d)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for c in knn.CORRUPTIONS:
    s = sp[sp.corruption == c]
    ax[0].plot(s.severity, s.spread, "o-", color=COL[c], label=c)
ax[0].set(xlabel="severity", ylabel="mean distance to centroid (unit vectors)",
          title="Embedding spread", xticks=knn.SEVERITIES)
ax[0].legend(frameon=False)

rows = []
for (c, s), g in f.groupby(["corruption", "severity"]):
    if g.in_sink.any():
        a, b = g.knn_dist[g.in_sink], g.knn_dist[~g.in_sink]
        rows.append(dict(corruption=c, severity=s, n_sink=len(a), sink=a.mean(), rest=b.mean(),
                         p=mannwhitneyu(a, b, alternative="less").pvalue))
dense = pd.DataFrame(rows)
for c in knn.CORRUPTIONS:
    t = dense[dense.corruption == c]
    ax[1].plot(t.severity, t.sink, "o-", color=COL[c], label=f"{c}: sink answers")
    ax[1].plot(t.severity, t.rest, "o--", color=COL[c], alpha=.5, label=f"{c}: other answers")
ax[1].set(xlabel="severity", ylabel="mean cosine distance to 15 nbrs",
          title="Sink answers sit in dense regions", xticks=knn.SEVERITIES)
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "c2_spread_and_sink_density.png", dpi=150)
mono = {c: bool(np.all(np.diff(sp[sp.corruption == c].sort_values("severity").spread) < 0))
        for c in knn.CORRUPTIONS}
print("spread falls monotonically:", mono)
print(dense.round(4).to_string(index=False))
'''

C1 = '''# C1: disagreements are near-ties.
# (a) low CNN confidence predicts disagreement. (b) the SNN answer is supported by the CNN's
# own neighbourhood more than a frequency-matched random class.
rows = []
for (c, s), g in f.groupby(["corruption", "severity"]):
    a, lo, hi = auroc_ci(g.disagree, -g.cnn_conf)
    dis = g[g.disagree]
    diff = (dis.snn_support - dis.rand_support).values
    rng = np.random.default_rng(0)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(1000)]
    rows.append(dict(corruption=c, severity=s, n_dis=len(dis), auc_conf=a, lo=lo, hi=hi,
                     cnn_support=dis.consensus.mean(),
                     snn_support=dis.snn_support.mean(), rand_support=dis.rand_support.mean(),
                     diff=diff.mean(), d_lo=np.percentile(bs, 2.5), d_hi=np.percentile(bs, 97.5)))
c1 = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, c in enumerate(knn.CORRUPTIONS):
    t = c1[c1.corruption == c]
    x = t.severity + (i - 1) * .1
    ax[0].errorbar(x, t.auc_conf, [t.auc_conf - t.lo, t.hi - t.auc_conf], fmt="o-",
                   color=COL[c], capsize=2, label=c)
    ax[1].errorbar(x, t.snn_support, fmt="o-", color=COL[c], label=f"{c}: SNN answer")
    ax[1].plot(x, t.rand_support, "o:", color=COL[c], alpha=.5, label=f"{c}: random class")
ax[0].axhline(.5, color="grey", lw=.8)
ax[0].set(xlabel="severity", ylabel="AUROC (95 % bootstrap)", ylim=(.45, .9),
          title="Low CNN confidence flags disagreement", xticks=knn.SEVERITIES)
ax[0].legend(frameon=False)
ax[1].set(xlabel="severity", ylabel="share of 15 nbrs voting for it",
          title="Disagreeing images: neighbours back the SNN answer", xticks=knn.SEVERITIES)
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "c1_near_ties.png", dpi=150)
print(c1.round(3).to_string(index=False))
'''

C3 = '''# C3: the two signals trade places with severity; combining them helps.
rows = []
for (c, s), g in f.groupby(["corruption", "severity"]):
    y = g.disagree.values
    X = np.column_stack([g.cnn_conf, g.consensus, g.knn_dist])
    a_conf = roc_auc_score(y, -g.cnn_conf)
    a_cons = roc_auc_score(y, -g.consensus)
    dd, lo, hi = auroc_diff_ci(y, -g.cnn_conf, -g.consensus)
    p = cross_val_predict(LogisticRegression(max_iter=1000), X[:, :2], y, cv=5,
                          method="predict_proba")[:, 1]
    ns = g[~g.in_sink]  # confound check: is consensus only "the image fell into a sink"?
    rows.append(dict(corruption=c, severity=s, auc_conf=a_conf, auc_cons=a_cons,
                     cons_minus_conf=dd, lo=lo, hi=hi, auc_both_cv=roc_auc_score(y, p),
                     auc_cons_no_sink=roc_auc_score(ns.disagree, -ns.consensus)))
c3 = pd.DataFrame(rows)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, c in zip(axs, knn.CORRUPTIONS):
    t = c3[c3.corruption == c]
    ax.plot(t.severity, t.auc_conf, "o-", color="#333", label="confidence")
    ax.plot(t.severity, t.auc_cons, "s-", color=COL[c], label="neighbour consensus")
    ax.plot(t.severity, t.auc_both_cv, "^:", color="#999", label="both (5-fold CV)")
    ax.axhline(.5, color="grey", lw=.8)
    ax.set(title=c, xlabel="severity", xticks=knn.SEVERITIES)
axs[0].set_ylabel("AUROC for disagreement")
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Which signal explains CNN-SNN disagreement, per condition")
fig.tight_layout()
fig.savefig(FIG / "c3_signals_trade_places.png", dpi=150)
for c in knn.CORRUPTIONS:
    t = c3[c3.corruption == c]
    r = spearmanr(t.severity, t.cons_minus_conf)
    print(f"{c}: Spearman(severity, cons - conf) = {r.statistic:+.2f}")
c3["both_ge_best"] = c3.auc_both_cv >= c3[["auc_conf", "auc_cons"]].max(1) - 1e-9
print(c3.round(3).to_string(index=False))
'''

UMAP = '''# Map of the embedding for three severities per corruption (UMAP on the 512-d vectors of
# one condition at a time, cosine, n_neighbors=15, min_dist=0.1, seed 0). Positions are not
# comparable between panels, and cluster sizes mean nothing (Lecture 03).
import umap
from matplotlib.lines import Line2D

fig, axs = plt.subplots(3, 3, figsize=(12, 11))
maps = {}
for i, c in enumerate(knn.CORRUPTIONS):
    for j, s in enumerate([1, 3, 5]):
        g = f[(f.corruption == c) & (f.severity == s)]
        E = knn.unit(np.vstack(d.loc[g.index, "embedding"].values))
        xy = umap.UMAP(n_neighbors=15, min_dist=.1, metric="cosine",
                       random_state=0).fit_transform(E)
        maps[(c, s)] = (g, xy)
        ax = axs[i, j]
        ax.scatter(*xy[g.agree.values].T, s=3, c="#bbb", label="agree")
        ax.scatter(*xy[g.disagree.values].T, s=4, c=COL[c], label="disagree")
        sk = g.in_sink.values
        if sk.any():
            ax.scatter(*xy[sk].T, s=8, facecolors="none", edgecolors="k", lw=.4,
                       label="CNN answer is a sink")
        ax.set(title=f"{c}, severity {s}", xticks=[], yticks=[])
handles = [Line2D([], [], ls="", marker="o", ms=4, color="#bbb", label="CNN and SNN agree"),
           Line2D([], [], ls="", marker="o", ms=4, color="#555",
                  label="disagree (row colour = corruption)"),
           Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="k",
                  label=f"CNN answer is a sink (>= {knn.SINK_SHARE:.0%} of answers)")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, .03, 1, 1))
fig.savefig(FIG / "umap_embedding_agreement.png", dpi=150)
'''

INTERACTIVE = '''# Interactive version for exploration: hover shows the image and both answers.
# Saved as HTML next to the figures; open it in a browser.
import plotly.express as px
from plotly.subplots import make_subplots

H = pd.read_csv(knn.ROOT / "phase1" / "ai-aided" / "aiv" / "imagenet_hierarchy.csv")
name = H.set_index("label").name
panels = [("gaussian_noise", 3), ("contrast", 5), ("fog", 5)]
fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{c} {s}" for c, s in panels])
for k, (c, s) in enumerate(panels, 1):
    g, xy = maps[(c, s)]
    t = pd.DataFrame(dict(x=xy[:, 0], y=xy[:, 1], image=g.image_id.values,
                          true=name[g.true_label].values, cnn=name[g.cnn_pred].values,
                          snn=name[g.akida_pred].values, conf=g.cnn_conf.round(3).values,
                          consensus=g.consensus.values,
                          status=np.where(g.agree, "agree", "disagree")))
    sc = px.scatter(t, x="x", y="y", color="status", hover_data=["image", "true", "cnn", "snn",
                    "conf", "consensus"], color_discrete_map={"agree": "#bbb",
                    "disagree": COL[c]})
    for tr in sc.data:
        tr.showlegend = k == 1
        fig.add_trace(tr, row=1, col=k)
fig.update_traces(marker_size=4)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_layout(height=450, width=1300, title="Embedding maps, hover for details")
fig.write_html(FIG / "umap_interactive.html", include_plotlyjs="cdn")
print("wrote", FIG / "umap_interactive.html")
'''


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Phase II (AI-aided): kNN in embedding space\n\n"
           "Claims and decision rules: `claims.md`. Data: `results/akidanet_imagenetc.parquet` "
           "(card run; predictions identical to `results_card/predictions.csv`). "
           "Run from `phase2/ai-aided/`."),
        code(SETUP),
        md("## 0. Embedding sanity check"),
        code(CHECK),
        md("## C2. Does corruption collapse the embedding into sinks?"),
        code(C2),
        md("## C1. Are disagreements near-ties?"),
        code(C1),
        md("## C3. Confidence vs neighbourhood, and both combined"),
        code(C3),
        md("## Maps"),
        code(UMAP),
        code(INTERACTIVE),
        md(findings("01")),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nbf.write(nb, HERE / "01_knn_embedding.ipynb")


if __name__ == "__main__":
    build()
