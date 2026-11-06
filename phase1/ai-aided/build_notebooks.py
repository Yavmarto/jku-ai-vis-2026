"""Generates the three view notebooks (kept as code so the cells are reviewable in diffs).
Findings text lives in findings/*.md and is pasted into the last cell of each notebook.

    python phase1/ai-aided/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md = nbf.v4.new_markdown_cell


def split_semicolons(src):
    """Turn `a; b` into two statements (ruff E702) using the tokenizer, so strings and
    continuation lines are untouched. The new statement gets the indent of the logical line."""
    import io
    import tokenize

    lines = src.splitlines(keepends=True)
    cuts = {}
    depth, stmt_col, fresh = 0, 0, True
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            fresh = True
            continue
        if t.type in (tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER):
            continue
        if fresh:
            stmt_col, fresh = t.start[1], False
        if t.type == tokenize.OP and t.string in "([{":
            depth += 1
        elif t.type == tokenize.OP and t.string in ")]}":
            depth -= 1
        elif t.type == tokenize.OP and t.string == ";" and depth == 0:
            cuts.setdefault(t.start[0] - 1, []).append((t.start[1], stmt_col))
    out = []
    for i, line in enumerate(lines):
        if i not in cuts:
            out.append(line)
            continue
        prev = 0
        for j, (c, col) in enumerate(cuts[i]):
            head = line[prev:c] if j == 0 else " " * col + line[prev:c].strip()
            out.append(head.rstrip() + "\n")
            prev = c + 1
        tail = line[prev:].strip()
        if tail:
            out.append(" " * cuts[i][-1][1] + tail + "\n")
    return "".join(out).rstrip("\n")


def code(src):
    return nbf.v4.new_code_cell(split_semicolons(src))


def findings(name):
    p = HERE / "findings" / f"{name}.md"
    return p.read_text() if p.exists() else "## Findings\n\n_Not written yet._"


ENV = '''%matplotlib inline
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")'''

SETUP = '''import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiv.load import CLASSES, OUTCOMES, data_source, paired, pick
from aiv.stats import wilson
from lab.config import CORRUPTIONS, SEVERITIES

CNN, SNN = pick()
W = paired(CNN, SNN)
COND = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
COL = dict(zip(OUTCOMES, ["#d9d9d9", "#d62728", "#1f77b4", "#7f7f7f", "#ff7f0e"]))
print(f"CNN = {CNN} | SNN = {SNN} | data = {data_source()} | rows = {len(W):,}")'''

ACC = '''from aiv.load import accuracy

acc = pd.concat([accuracy(CNN), accuracy(SNN)])
k = (acc.top1 * acc.n_images).round()
acc["ci95"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(*wilson(k, acc.n_images))]
acc.pivot_table(index=["corruption", "severity"], columns="model", values="top1",
                sort=False).round(3)'''

THUMBS = '''from lab.data import load_clean, load_corrupted

_X = {}
def images(c, s):
    if (c, s) not in _X:
        _X[(c, s)] = load_clean()[0] if c == "clean" else load_corrupted(c, s)[0]
    return _X[(c, s)]

def thumbs(df, n=24, cols=8, title=""):
    """The actual (corrupted) images behind a selection of rows of W."""
    df = df.head(n)
    if df.empty:
        print("nothing selected"); return
    rows = int(np.ceil(len(df) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.75), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axs.flat, df.iterrows()):
        ax.imshow(images(r.corruption, int(r.severity))[int(r.image_id)])
        ax.set_title(f"#{int(r.image_id)} {CLASSES[r.label][:5]}\\n"
                     f"C:{CLASSES[int(r.pred_cnn)][:5]} S:{CLASSES[int(r.pred_snn)][:5]}", fontsize=6.5)
    fig.suptitle(title, fontsize=9); plt.tight_layout(); plt.show()'''

# ------------------------------------------------------------------ 01 confusion
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md("# 01 Confusion views: where do CNN and SNN make *different* mistakes?\n\n"
       "Black box. Rows = true class, columns = predicted class, row-normalised. "
       "Difference = SNN − CNN on the **same images**: red = the SNN makes this mistake more often. "
       "A dot marks cells whose paired 95 % interval excludes 0; unmarked cells are within noise.\n\n"
       "Tests H1, H2, H3, H4 from `hypotheses.md`."),
    code(ENV),
    code(SETUP),
    md("Accuracy per condition, for orientation only (the views below are the analysis)."),
    code(ACC),
    md("## Class order by error similarity\n"
       "Hierarchical clustering (average linkage) of the symmetrised off-diagonal confusion, both "
       "models and all conditions pooled, so classes confused with each other sit together."),
    code('''from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

def counts(d, m):
    c = np.zeros((10, 10))
    np.add.at(c, (d.label.values, d[f"pred_{m}"].values.astype(int)), 1)
    return c

def rate(d, m):
    c = counts(d, m)
    return c / np.maximum(c.sum(1, keepdims=True), 1)

off = (rate(W, "cnn") + rate(W, "snn")) / 2
np.fill_diagonal(off, 0)
sim = (off + off.T) / 2
dist = 1 - sim / sim.max()
np.fill_diagonal(dist, 0)
ORDER = leaves_list(linkage(squareform(dist, checks=False), "average"))
NAMES = [CLASSES[i] for i in ORDER]
print("order:", NAMES)'''),
    md("## Paired differences per cell\nFor true class t and predicted class p: share of t-images "
       "the SNN sends to p minus share the CNN sends to p, with a paired (McNemar-style) 95 % interval."),
    code('''from aiv.stats import paired_diff

rows = []
for (c, s), d in W.groupby(["corruption", "severity"], sort=False):
    for t in range(10):
        dt = d[d.label == t]
        for p in range(10):
            diff, lo, hi, n = paired_diff(dt.pred_cnn.values == p, dt.pred_snn.values == p)
            rows.append((c, s, t, p, diff, lo, hi, n))
DIFF = pd.DataFrame(rows, columns=["corruption", "severity", "t", "p", "diff", "lo", "hi", "n"])
DIFF["sig"] = (DIFF.lo > 0) | (DIFF.hi < 0)
off_sig = DIFF[(DIFF.t != DIFF.p) & DIFF.sig]
print(f"{DIFF.sig[(DIFF.t != DIFF.p)].mean():.1%} of off-diagonal cells differ significantly; "
      f"per condition: n per true class ≈ {int(DIFF.n.median())}")'''),
    md("## Small multiples: corruption × severity, for CNN, SNN, and the difference\n"
       "Error panels mask the diagonal so the colour scale is spent on mistakes. Panel titles carry "
       "accuracy with its 95 % Wilson interval (CNN, SNN) or the paired accuracy difference (SNN − CNN)."),
    code('''def err(m):
    m = m.copy(); np.fill_diagonal(m, np.nan); return m[np.ix_(ORDER, ORDER)]

for kind in ["cnn", "snn", "diff"]:
    fig, axs = plt.subplots(len(CORRUPTIONS), 6, figsize=(17, 9), sharex=True, sharey=True)
    for i, corr in enumerate(CORRUPTIONS):
        for j, (c, s) in enumerate([("clean", 0)] + [(corr, s) for s in SEVERITIES]):
            d = W[(W.corruption == c) & (W.severity == s)]
            ax = axs[i, j]
            if kind == "diff":
                M = err(rate(d, "snn")) - err(rate(d, "cnn"))
                im = ax.imshow(M, cmap="RdBu_r", vmin=-.2, vmax=.2)
                sg = DIFF[(DIFF.corruption == c) & (DIFF.severity == s) & DIFF.sig & (DIFF.t != DIFF.p)]
                pos = {k: n for n, k in enumerate(ORDER)}
                ax.scatter([pos[p] for p in sg.p], [pos[t] for t in sg.t], s=4, c="k")
                dd, lo, hi, _ = paired_diff(d.cnn_ok, d.snn_ok)
                t = f"{c} {s}\\nΔacc {dd:+.3f} [{lo:+.3f},{hi:+.3f}]"
            else:
                M = err(rate(d, kind))
                im = ax.imshow(M, cmap="Reds", vmin=0, vmax=.5)
                ok = d[f"{kind}_ok"]
                lo, hi = wilson(ok.sum(), len(ok))
                t = f"{c} {s}\\nacc {ok.mean():.3f} [{lo:.3f},{hi:.3f}]"
            ax.set_title(t, fontsize=7)
            if j == 0:
                ax.set_ylabel(corr, fontsize=9)
    for ax in axs[-1]:
        ax.set_xticks(range(10)); ax.set_xticklabels(NAMES, rotation=90, fontsize=6)
    for ax in axs[:, 0]:
        ax.set_yticks(range(10)); ax.set_yticklabels(NAMES, fontsize=6)
    fig.colorbar(im, ax=axs, shrink=.6,
                 label="SNN − CNN share (dot: 95 % CI excludes 0)" if kind == "diff" else "error share")
    fig.suptitle({"cnn": f"CNN ({CNN})", "snn": f"SNN ({SNN})", "diff": "Difference SNN − CNN"}[kind])
    plt.show()'''),
    md("## Largest significant divergences\nTop off-diagonal cells by |SNN − CNN|, only where the 95 % "
       "interval excludes 0, per corruption at severity 5."),
    code('''top = off_sig[off_sig.severity == 5].assign(a=lambda x: x["diff"].abs())
top = top.sort_values("a", ascending=False).groupby("corruption").head(6)
top.assign(true=[CLASSES[t] for t in top.t], pred=[CLASSES[p] for p in top.p])[
    ["corruption", "true", "pred", "diff", "lo", "hi", "n"]].round(3)'''),
    md("## H1/H2: named pairs across severity\nShare of true-class images sent to the named class, "
       "with 95 % Wilson bands. Solid = CNN, dashed = SNN."),
    code('''PAIRS = [("cat", "dog"), ("dog", "cat"), ("deer", "horse"),
         ("automobile", "truck"), ("truck", "automobile"), ("airplane", "ship"), ("ship", "airplane")]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
colors = plt.cm.tab10(np.arange(len(PAIRS)))
for ax, corr in zip(axs, CORRUPTIONS):
    for (t, p), col in zip(PAIRS, colors):
        for m, ls in (("cnn", "-"), ("snn", "--")):
            ys, los, his = [], [], []
            for s in [0] + SEVERITIES:
                d = W[(W.corruption == (corr if s else "clean")) & (W.severity == s)
                      & (W.label == CLASSES.index(t))]
                k = (d[f"pred_{m}"] == CLASSES.index(p)).sum()
                lo, hi = wilson(k, len(d)); ys.append(k / len(d)); los.append(lo); his.append(hi)
            ax.plot([0] + SEVERITIES, ys, ls, color=col, marker="o", ms=3,
                    label=f"{t}→{p}" if (m == "cnn" and corr == CORRUPTIONS[0]) else None)
            ax.fill_between([0] + SEVERITIES, los, his, color=col, alpha=.12)
    ax.set_title(corr); ax.set_xlabel("severity (0 = clean)")
axs[0].set_ylabel("share of true class predicted as target")
fig.legend(fontsize=8, ncol=7, loc="lower center", bbox_to_anchor=(.5, -.08))
plt.tight_layout(); plt.show()'''),
    md("## Where do the errors go? Sink classes\nShare of *all wrong* predictions that land in each "
       "class. A class that absorbs errors as severity grows is a 'sink'."),
    code('''rows = []
for (c, s), d in W.groupby(["corruption", "severity"], sort=False):
    for m in ("cnn", "snn"):
        wrong = d[~d[f"{m}_ok"]]
        share = np.bincount(wrong[f"pred_{m}"].astype(int), minlength=10) / max(len(wrong), 1)
        rows += [(c, s, m, CLASSES[k], share[k], len(wrong)) for k in range(10)]
SINK = pd.DataFrame(rows, columns=["corruption", "severity", "model", "pred", "share", "n_wrong"])
fig, axs = plt.subplots(2, 3, figsize=(15, 6), sharex=True, sharey=True)
for i, m in enumerate(("cnn", "snn")):
    for j, corr in enumerate(CORRUPTIONS):
        d = SINK[(SINK.model == m) & SINK.corruption.isin(["clean", corr])]
        H = d.pivot_table(index="pred", columns="severity", values="share").reindex(NAMES)
        im = axs[i, j].imshow(H.values, cmap="Purples", vmin=0, vmax=.6, aspect="auto")
        for (r, cc), v in np.ndenumerate(H.values):
            axs[i, j].text(cc, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                           color="w" if v > .35 else "k")
        axs[i, j].set_title(f"{m.upper()} · {corr}", fontsize=9)
        axs[i, j].set_xticks(range(6)); axs[i, j].set_yticks(range(10)); axs[i, j].set_yticklabels(NAMES, fontsize=7)
fig.colorbar(im, ax=axs, label="share of wrong predictions"); plt.show()'''),
    md("## Drill-down: images behind a cell"),
    code(THUMBS),
    code('''T, P, CORR, SEV = "dog", "deer", "contrast", 5     # largest divergence; change to inspect any cell
sel = W[(W.label == CLASSES.index(T)) & (W.corruption == CORR) & (W.severity == SEV)
        & (W.pred_snn == CLASSES.index(P)) & (W.pred_cnn != CLASSES.index(P))]
print(len(sel), f"images: SNN says {P}, CNN does not")
thumbs(sel, title=f"true {T}: SNN→{P}, CNN≠{P} ({CORR} {SEV})")'''),
    md(findings("01")),
]

# ------------------------------------------------------------------ 02 projection
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md("# 02 Projection: are disagreements at class borders, and do they follow drift?\n\n"
       "UMAP of the float CNN's global-average-pool features (`aiv/features.py`, fixed subset of "
       "test images) for clean and corrupted inputs, coloured by CNN/SNN outcome. The SNN enters only "
       "through its predictions.\n\n"
       "**Caveats (Lecture 03):** UMAP cluster sizes and gaps are not meaningful, so the border and "
       "drift questions are answered in the original feature space (kNN label purity, cosine drift), "
       "not from the picture. Parameters are printed. Tests H5."),
    code(ENV),
    code(SETUP),
    code('''from lab.config import DATA
F = np.load(DATA / "features" / "cnn-float.npz")
FI = pd.DataFrame({"image_id": F["image_id"], "corruption": F["corruption"], "severity": F["severity"]})
X = F["features"].astype(np.float32)
X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
D = FI.merge(W, on=["image_id", "corruption", "severity"], how="left", validate="one_to_one")
D["condition"] = D.corruption + " " + D.severity.astype(str)
D["cnn"] = np.array(CLASSES)[D.pred_cnn.astype(int)]; D["snn"] = np.array(CLASSES)[D.pred_snn.astype(int)]
print(len(np.unique(FI.image_id)), "images x", FI.groupby(["corruption", "severity"]).ngroups, "conditions")'''),
    code('''import umap
from sklearn.decomposition import PCA
SEV_SHOWN = [0, 3, 5]
keep = D.severity.isin(SEV_SHOWN).values
P = {"n_neighbors": 30, "min_dist": 0.1, "metric": "cosine", "random_state": 0}
pca = PCA(50, random_state=0).fit(X[keep])
Z = umap.UMAP(**P).fit_transform(pca.transform(X[keep]))
U = D[keep].reset_index(drop=True).assign(x=Z[:, 0], y=Z[:, 1])
print("PCA 50 -> UMAP", P, "| points:", len(U),
      f"| PCA variance kept {pca.explained_variance_ratio_.sum():.2f}")'''),
    md("## Outcome map\nGrey = both right; colour only where something goes wrong."),
    code('''fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), sharex=True, sharey=True)
for ax, c in zip(axs, ["clean"] + CORRUPTIONS):
    d = U[(U.corruption == c)]
    for o in OUTCOMES:
        e = d[d.outcome == o]
        ax.scatter(e.x, e.y, s=2 if o == "both right" else 4, c=COL[o], alpha=.5 if o == "both right" else .8, label=o)
    ax.set_title(c + ("" if c == "clean" else " (sev 3+5)")); ax.set_xticks([]); ax.set_yticks([])
axs[0].legend(markerscale=3, fontsize=7); plt.tight_layout(); plt.show()'''),
    code('''fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), sharex=True, sharey=True)
for ax, c in zip(axs, ["clean"] + CORRUPTIONS):
    d = U[U.corruption == c]
    ax.scatter(d.x, d.y, s=2, c=d.label, cmap="tab10", alpha=.6)
    ax.set_title(f"{c}: true class"); ax.set_xticks([]); ax.set_yticks([])
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i / 9), label=n)
                    for i, n in enumerate(CLASSES)], loc="lower center", ncol=10, fontsize=8,
           bbox_to_anchor=(.5, -.06)); plt.tight_layout(); plt.show()'''),
    md("Interactive version (hover for image id, labels, predictions) for exploring; "
       "use the ids in the drill-down cell."),
    code('''import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook_connected"  # load plotly.js from CDN instead of embedding 3 MB
fig = px.scatter(U, x="x", y="y", color="outcome", color_discrete_map=COL, facet_col="corruption",
                 facet_col_wrap=2, height=750, opacity=.6,
                 hover_data=["image_id", "condition", "true", "cnn", "snn"])
fig.update_traces(marker_size=3); fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
fig.show()'''),
    md("## Border test in feature space\nLabel purity = share of an image's 15 nearest *clean* "
       "neighbours (cosine, original 1280-d features) with the same true class. Low purity = near a "
       "class border. Disagreement rate per purity bin, with 95 % Wilson intervals."),
    code('''from sklearn.neighbors import NearestNeighbors
cl = (FI.corruption == "clean").values
nn = NearestNeighbors(n_neighbors=16, metric="cosine").fit(X[cl])
lab_clean = D.label.values[cl]
_, ix = nn.kneighbors(X)
neigh = np.where(cl[:, None], ix[:, 1:], ix[:, :15])
D["purity"] = (lab_clean[neigh] == D.label.values[:, None]).mean(1)
D["pbin"] = pd.cut(D.purity, [-.01, .2, .4, .6, .8, .99, 1.0], labels=["≤.2", ".2–.4", ".4–.6", ".6–.8", ".8–.99", "1"])
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
for ax, c in zip(axs, ["clean"] + CORRUPTIONS):
    for s, ls in ((0, "-"),) if c == "clean" else ((1, ":"), (3, "--"), (5, "-")):
        g = D[(D.corruption == c) & (D.severity == s)].groupby("pbin", observed=False)
        k, n = g.agree.apply(lambda a: (~a).sum()), g.size()
        lo, hi = wilson(k, n)
        xs = np.arange(len(n))
        ax.plot(xs, k / n, ls, marker="o", ms=3, label=f"sev {s}")
        ax.fill_between(xs, lo, hi, alpha=.15)
    ax.set_xticks(range(6)); ax.set_xticklabels(n.index, fontsize=7); ax.set_title(c); ax.set_xlabel("kNN label purity")
axs[0].set_ylabel("CNN ≠ SNN rate"); axs[-1].legend(fontsize=7); plt.tight_layout(); plt.show()
print(D.groupby("pbin", observed=False).size().rename("n").to_frame().T)'''),
    md("## Drift test\nCosine distance between each image's corrupted and clean feature vector. "
       "Drift grows with severity, so the test is done **within** a severity: disagreement rate per "
       "drift quintile of that severity, with 95 % Wilson intervals."),
    code('''clean_vec = pd.Series(range(len(FI)))[cl].set_axis(FI.image_id[cl].values)
D["drift"] = 1 - (X * X[clean_vec.loc[D.image_id].values]).sum(1)
fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
for ax, corr in zip(axs, CORRUPTIONS):
    for s, ls in ((1, ":"), (3, "--"), (5, "-")):
        d = D[(D.corruption == corr) & (D.severity == s)].copy()
        d["q"] = pd.qcut(d.drift, 5, labels=False)
        g = d.groupby("q")
        k, n = g.agree.apply(lambda a: (~a).sum()), g.size()
        lo, hi = wilson(k, n)
        ax.plot(k.index + 1, k / n, ls, marker="o", ms=3, label=f"sev {s}")
        ax.fill_between(k.index + 1, lo, hi, alpha=.15)
    ax.set_title(corr); ax.set_xlabel("drift quintile within severity (1 = least)"); ax.set_xticks(range(1, 6))
axs[0].set_ylabel("CNN ≠ SNN rate"); axs[2].legend(fontsize=7)
for corr in CORRUPTIONS:
    m = D[D.corruption == corr].groupby("severity").drift.median()
    axs[3].plot(m.index, m.values, marker="o", label=corr)
axs[3].set_xlabel("severity"); axs[3].set_ylabel("median cosine drift from clean"); axs[3].legend(fontsize=7)
plt.tight_layout(); plt.show()'''),
    md("## Drill-down"),
    code(THUMBS),
    code('''sel = D[(D.outcome == "only CNN right") & (D.corruption == "contrast") & (D.severity == 5)]
thumbs(sel.sort_values("purity", ascending=False), title="only CNN right, contrast 5, highest purity first")'''),
    md(findings("02")),
]

# ------------------------------------------------------------------ 03 small multiples
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    md("# 03 Small multiples: which classes break first, and does the SNN break differently?\n\n"
       "Per class × corruption × severity, with 95 % intervals. Confidence is compared through "
       "AUROC (how well a model's own confidence separates its right from its wrong answers), "
       "because CNN confidence is a softmax probability and SNN confidence is a raw Akida potential. "
       "Tests H3, H4, H6, H7."),
    code(ENV),
    code(SETUP),
    md("## Outcome composition per class × corruption\nStacked share of images per outcome vs severity."),
    code('''fig, axs = plt.subplots(10, 3, figsize=(11, 19), sharex=True, sharey=True)
for j, corr in enumerate(CORRUPTIONS):
    d = W[W.corruption.isin(["clean", corr])]
    for k in range(10):
        t = (d[d.label == k].groupby("severity").outcome.value_counts(normalize=True)
             .unstack(fill_value=0).reindex(columns=OUTCOMES, fill_value=0))
        axs[k, j].stackplot(t.index, t.T.values, colors=list(COL.values()))
        if j == 0:
            axs[k, j].set_ylabel(CLASSES[k], fontsize=8)
    axs[0, j].set_title(corr, fontsize=10)
axs[-1, 1].set_xlabel("severity (0 = clean)")
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()], OUTCOMES, loc="upper center",
           ncol=5, fontsize=8, bbox_to_anchor=(.5, 1.01)); plt.tight_layout(); plt.show()'''),
    md("## Per-class accuracy gap SNN − CNN (paired 95 % CI)\nNegative = SNN loses more. "
       "One panel per corruption; each class shows severities 1–5 left to right."),
    code('''from aiv.stats import paired_diff

rows = []
for (c, s), d in W.groupby(["corruption", "severity"], sort=False):
    for k in range(10):
        dk = d[d.label == k]
        rows.append((c, s, CLASSES[k], *paired_diff(dk.cnn_ok, dk.snn_ok)))
G = pd.DataFrame(rows, columns=["corruption", "severity", "cls", "diff", "lo", "hi", "n"])
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, corr in zip(axs, CORRUPTIONS):
    for k, name in enumerate(CLASSES):
        g = G[(G.corruption == corr) & (G.cls == name)]
        xs = k + (g.severity.values - 3) * .14
        ax.errorbar(xs, g["diff"], yerr=[g["diff"] - g.lo, g.hi - g["diff"]], fmt="o", ms=3,
                    color=plt.cm.viridis((g.severity.values - 1) / 4)[0], ecolor="grey", lw=.8)
        ax.scatter(xs, g["diff"], c=plt.cm.viridis((g.severity.values - 1) / 4), s=12, zorder=3)
    c0 = G[G.corruption == "clean"].set_index("cls")["diff"]
    ax.scatter(range(10), c0.reindex(CLASSES), marker="_", s=200, c="k", label="clean")
    ax.axhline(0, c="k", lw=.6); ax.set_title(corr)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, fontsize=8)
axs[0].set_ylabel("SNN acc − CNN acc"); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.show()'''),
    md("## Disagreement rate: class × severity, one heatmap per corruption\n"
       "Cell text = rate; ± = half-width of the 95 % Wilson interval."),
    code('''fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), sharey=True)
for ax, corr in zip(axs, CORRUPTIONS):
    d = W[W.corruption.isin(["clean", corr])].assign(dis=lambda x: ~x.agree)
    k = d.pivot_table(index="label", columns="severity", values="dis", aggfunc="sum")
    n = d.pivot_table(index="label", columns="severity", values="dis", aggfunc="size")
    r = k / n; lo, hi = wilson(k.values, n.values)
    im = ax.imshow(r.values, cmap="Oranges", vmin=0, vmax=.6, aspect="auto")
    for (i, j), v in np.ndenumerate(r.values):
        ax.text(j, i, f"{v:.2f}\\n±{(hi[i, j] - lo[i, j]) / 2:.2f}", ha="center", va="center",
                fontsize=5.5, color="w" if v > .4 else "k")
    ax.set_xticks(range(6)); ax.set_xticklabels(r.columns); ax.set_title(corr); ax.set_xlabel("severity")
axs[0].set_yticks(range(10)); axs[0].set_yticklabels(CLASSES, fontsize=8)
fig.colorbar(im, ax=axs, label="share CNN ≠ SNN"); plt.show()'''),
    md("## H6: does confidence still separate right from wrong?\nAUROC of each model's own "
       "confidence for predicting its own correctness (0.5 = useless), 95 % bootstrap interval "
       "(200 resamples). Scale-free, so CNN probability and SNN potential are comparable."),
    code('''from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
rows = []
for (c, s), d in W.groupby(["corruption", "severity"], sort=False):
    for m in ("cnn", "snn"):
        y, sc = d[f"{m}_ok"].values, d[f"confidence_{m}"].values
        auc = roc_auc_score(y, sc)
        bs = []
        for _ in range(200):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                bs.append(roc_auc_score(y[i], sc[i]))
        rows.append((c, s, m, auc, *np.percentile(bs, [2.5, 97.5])))
A = pd.DataFrame(rows, columns=["corruption", "severity", "model", "auroc", "lo", "hi"])
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, corr in zip(axs, CORRUPTIONS):
    for m, col in (("cnn", "#d62728"), ("snn", "#1f77b4")):
        a = A[(A.model == m) & A.corruption.isin(["clean", corr])].sort_values("severity")
        ax.plot(a.severity, a.auroc, marker="o", color=col, label=m.upper())
        ax.fill_between(a.severity, a.lo, a.hi, color=col, alpha=.2)
    ax.axhline(.5, c="k", lw=.5, ls=":"); ax.set_title(corr); ax.set_xlabel("severity")
axs[0].set_ylabel("AUROC confidence → correct"); axs[0].legend(); plt.tight_layout(); plt.show()
A.round(3).pivot_table(index=["corruption", "severity"], columns="model", values="auroc", sort=False)'''),
    md("## Energy as an attribute (H7)"),
    code('''from lab.config import RESULTS
e = sorted(RESULTS.glob("*/energy.csv"))
print("energy.csv files:", [str(p.relative_to(RESULTS)) for p in e] or "none - lab.evaluate does not "
      "write energy yet, and the simulator reports no power. H7 cannot be tested in this run.")'''),
    md(findings("03")),
]

def strip_tf_noise(path):
    """Drop TensorFlow/absl start-up chatter from executed notebooks (stderr streams only)."""
    nb = nbf.read(path, 4)
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = [o for o in c.outputs if not (o.output_type == "stream" and o.name == "stderr")]
    nbf.write(nb, path)


if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--strip":
    for p in HERE.glob("0*.ipynb"):
        strip_tf_noise(p)
    raise SystemExit

for name, nb in [("01_confusion_views", nb1), ("02_projection", nb2), ("03_small_multiples", nb3)]:
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / f"{name}.ipynb")
print("ok")
