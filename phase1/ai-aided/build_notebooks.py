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
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")'''

SETUP = '''# shared setup for all three notebooks, so not every name is used in each
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: F401

from aiv.load import CORRUPTIONS, GROUPS, NAMES, OUTCOMES, SEVERITIES, paired  # noqa: F401
from aiv.load import GROUP as GROUP_OF, table, wordnet_distance  # noqa: F401
from aiv.stats import paired_diff, wilson  # noqa: F401

RUN = "card"                      # "card" = AKD1000 run, "sim" = software simulator run
W = paired(RUN)
COND = [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
COL = dict(zip(OUTCOMES, ["#d9d9d9", "#d62728", "#1f77b4", "#7f7f7f", "#ff7f0e"]))
print(f"run = {RUN} | {W.image_id.nunique()} images x {len(COND)} conditions = {len(W):,} rows | "
      f"{W.label.nunique()} classes in {len(GROUPS)} WordNet groups")'''

LOOKUP = '''from pathlib import Path

# Thumbnails need the image subset from scripts/make_imagenet_subset.py (not in the repo).
IMAGENET_C_DIR = Path("../../data/imagenet_subset/imagenet-c")

def show(df, n=16, cols=8, title=""):
    """Images behind a selection; falls back to a table if the subset is not on this machine."""
    df = df.head(n).assign(cnn=lambda d: NAMES[d.pred_cnn], snn=lambda d: NAMES[d.pred_snn])
    if not IMAGENET_C_DIR.exists():
        print(f"{title}  (no {IMAGENET_C_DIR}: table only)")
        return df[["image_id", "corruption", "severity", "true", "cnn", "snn"]]
    import matplotlib.image as mpimg
    rows = int(np.ceil(len(df) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axs.flat, df.iterrows()):
        ax.imshow(mpimg.imread(IMAGENET_C_DIR / r.corruption / str(r.severity) / r.image_id))
        ax.set_title(f"{r.true[:12]}\\nC:{r.cnn[:12]}\\nS:{r.snn[:12]}", fontsize=6)
    fig.suptitle(title, fontsize=9); plt.tight_layout(); plt.show()'''

# ------------------------------------------------------------------ 01
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md("# 01 Divergence: same accuracy, different answers\n\n"
       "Quantized Keras AkidaNet (CNN) vs the same weights converted and run on the **AKD1000** (SNN), "
       "2000 ImageNet-C images (2 per class), fog / Gaussian noise / contrast, severities 1–5. "
       "Data: `Phase 1 - No AI/results_card/` (Yoshi's run). All intervals 95 %.\n\n"
       "With 1000 classes and 2 images each, class-level confusion is noise, so classes are grouped "
       "into 14 WordNet groups (`aiv/imagenet_hierarchy.csv`) and distances between predictions are "
       "measured along the WordNet tree. Tests H1–H4 in `hypotheses.md`."),
    code(ENV),
    code(SETUP),
    md("## Is the card the same as the simulator?\nIf the hardware differed from the software "
       "simulator, every later view would need two versions."),
    code('''S = paired("sim")
m = W.merge(S, on=["image_id", "corruption", "severity"], suffixes=("", "_sim"))
print(f"Akida predictions card == sim: {(m.pred_snn == m.pred_snn_sim).mean():.4f}")
print(f"Keras predictions run card == run sim: {(m.pred_cnn == m.pred_cnn_sim).mean():.4f} "
      f"({(m.pred_cnn != m.pred_cnn_sim).sum()} of {len(m)} rows differ)")'''),
    md("## Accuracy is equal, agreement is not\nTop: accuracy of both models with Wilson bands. "
       "Bottom: share of images where CNN and SNN give the *same* class, and the paired accuracy "
       "difference SNN − CNN."),
    code('''rows = []
for (c, s), d in W.groupby(["corruption", "severity"], sort=False):
    lo_c, hi_c = wilson(d.cnn_ok.sum(), len(d))
    lo_s, hi_s = wilson(d.snn_ok.sum(), len(d))
    lo_a, hi_a = wilson(d.agree.sum(), len(d))
    g, glo, ghi, _ = paired_diff(d.cnn_ok, d.snn_ok)
    rows.append((c, s, d.cnn_ok.mean(), lo_c, hi_c, d.snn_ok.mean(), lo_s, hi_s,
                 d.agree.mean(), lo_a, hi_a, g, glo, ghi))
T = pd.DataFrame(rows, columns=["corruption", "severity", "cnn", "cnn_lo", "cnn_hi", "snn", "snn_lo",
                                "snn_hi", "agree", "agree_lo", "agree_hi", "gap", "gap_lo", "gap_hi"])
fig, axs = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
for j, corr in enumerate(CORRUPTIONS):
    t = T[T.corruption == corr]
    for m, col in (("cnn", "#d62728"), ("snn", "#1f77b4")):
        axs[0, j].plot(t.severity, t[m], marker="o", color=col, label=m.upper())
        axs[0, j].fill_between(t.severity, t[f"{m}_lo"], t[f"{m}_hi"], color=col, alpha=.2)
    axs[1, j].plot(t.severity, t.agree, marker="o", color="k", label="agreement")
    axs[1, j].fill_between(t.severity, t.agree_lo, t.agree_hi, color="k", alpha=.15)
    ax2 = axs[1, j].twinx()
    ax2.errorbar(t.severity + .08, t.gap, yerr=[t.gap - t.gap_lo, t.gap_hi - t.gap], fmt="s", ms=4,
                 color="#9467bd", label="SNN − CNN accuracy")
    ax2.axhline(0, color="#9467bd", lw=.5, ls=":"); ax2.set_ylim(-.06, .06)
    axs[0, j].set_title(corr); axs[1, j].set_xlabel("severity"); axs[1, j].set_ylim(.55, .85)
    if j == 2:
        ax2.set_ylabel("SNN − CNN accuracy", color="#9467bd")
axs[0, 0].set_ylabel("top-1 accuracy"); axs[1, 0].set_ylabel("CNN = SNN (agreement)")
axs[0, 0].legend(); plt.tight_layout(); plt.show()
T.round(3)[["corruption", "severity", "cnn", "snn", "gap", "gap_lo", "gap_hi", "agree"]]'''),
    md("## Collapse onto sink classes\nHow many distinct classes each model still predicts, and "
       "which classes absorb the predictions at high severity."),
    code('''div = W.groupby(["corruption", "severity"], sort=False).agg(
    cnn=("pred_cnn", "nunique"), snn=("pred_snn", "nunique")).reset_index()
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
for ax, corr in zip(axs, CORRUPTIONS):
    d = div[div.corruption == corr]
    ax.plot(d.severity, d.cnn, "o-", color="#d62728", label="CNN")
    ax.plot(d.severity, d.snn, "s--", color="#1f77b4", label="SNN")
    ax.set_title(corr); ax.set_xlabel("severity")
axs[0].set_ylabel("distinct predicted classes (of 1000)"); axs[0].legend(); plt.tight_layout(); plt.show()

rows = []
for (c, s), d in W.groupby(["corruption", "severity"], sort=False):
    if s in (3, 5):
        for m in ("cnn", "snn"):
            vc = d[f"pred_{m}"].value_counts(normalize=True).head(3)
            rows.append((c, s, m.upper(), "; ".join(f"{NAMES[k]} {v:.0%}" for k, v in vc.items())))
pd.DataFrame(rows, columns=["corruption", "severity", "model", "top-3 predicted classes"])'''),
    md("## Group-level confusion: CNN, SNN, and the paired difference\nRows = true WordNet group, "
       "columns = predicted group, row-normalised. Difference panels: red = the SNN sends more images "
       "there; a dot = paired 95 % interval excludes 0."),
    code('''from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

GI = {g: i for i, g in enumerate(GROUPS)}
def gmat(d, m):
    c = np.zeros((len(GROUPS), len(GROUPS)))
    np.add.at(c, (d.group.map(GI).values, d[f"group_{m}"].map(GI).values), 1)
    return c / np.maximum(c.sum(1, keepdims=True), 1)

off = (gmat(W, "cnn") + gmat(W, "snn")) / 2
np.fill_diagonal(off, 0)
sim = (off + off.T) / 2
dist = 1 - sim / sim.max()
np.fill_diagonal(dist, 0)
ORDER = leaves_list(linkage(squareform(dist, checks=False), "average"))
GN = [GROUPS[i] for i in ORDER]

rows = []
for (c, s), d in W.groupby(["corruption", "severity"], sort=False):
    for t in GROUPS:
        dt = d[d.group == t]
        for p in GROUPS:
            rows.append((c, s, t, p, *paired_diff(dt.group_cnn.values == p, dt.group_snn.values == p)))
GD = pd.DataFrame(rows, columns=["corruption", "severity", "t", "p", "diff", "lo", "hi", "n"])
GD["sig"] = (GD.lo > 0) | (GD.hi < 0)

SHOW = [1, 3, 5]
fig, axs = plt.subplots(3, 9, figsize=(24, 8.5), sharex=True, sharey=True)
for i, corr in enumerate(CORRUPTIONS):
    for j, s in enumerate(SHOW):
        d = W[(W.corruption == corr) & (W.severity == s)]
        for k, kind in enumerate(("cnn", "snn", "diff")):
            ax = axs[i, j * 3 + k]
            if kind == "diff":
                M = (gmat(d, "snn") - gmat(d, "cnn"))[np.ix_(ORDER, ORDER)]
                im_d = ax.imshow(M, cmap="RdBu_r", vmin=-.15, vmax=.15)
                sg = GD[(GD.corruption == corr) & (GD.severity == s) & GD.sig]
                ax.scatter([GN.index(p) for p in sg.p], [GN.index(t) for t in sg.t], s=5, c="k")
            else:
                im = ax.imshow(gmat(d, kind)[np.ix_(ORDER, ORDER)], cmap="Purples", vmin=0, vmax=.6)
            ax.set_title(f"{corr[:8]} {s} · {kind.upper()}", fontsize=8)
for ax in axs[-1]:
    ax.set_xticks(range(len(GN))); ax.set_xticklabels(GN, rotation=90, fontsize=6)
for ax in axs[:, 0]:
    ax.set_yticks(range(len(GN))); ax.set_yticklabels(GN, fontsize=6)
fig.colorbar(im, ax=axs, shrink=.4, location="right", label="share (CNN / SNN)")
fig.colorbar(im_d, ax=axs, shrink=.4, location="left", label="SNN − CNN")
plt.show()
print(f"{GD.sig.mean():.1%} of group cells differ significantly")'''),
    md("## When they disagree, how far apart are the two answers?\nWordNet distance (edges via the "
       "lowest common ancestor) between the CNN's and the SNN's class, only where they differ. "
       "Grey band: the same distance for random class pairs (5000 draws, 25–75 %)."),
    code('''rng = np.random.default_rng(0)
a, b = rng.integers(0, 1000, 5000), rng.integers(0, 1000, 5000)
RND = np.array([wordnet_distance(x, y) for x, y in zip(a, b) if x != y])
D = W[~W.agree].copy()
D["dist"] = [wordnet_distance(x, y) for x, y in zip(D.pred_cnn, D.pred_snn)]
D["same_group"] = D.group_cnn == D.group_snn
fig, axs = plt.subplots(1, 2, figsize=(14, 3.8))
for corr, col in zip(CORRUPTIONS, ["#2ca02c", "#ff7f0e", "#1f77b4"]):
    g = D[D.corruption == corr].groupby("severity")
    q = g.dist.quantile([.25, .5, .75]).unstack()
    axs[0].plot(q.index, q[.5], "o-", color=col, label=corr)
    axs[0].fill_between(q.index, q[.25], q[.75], color=col, alpha=.12)
    k, n = g.same_group.sum(), g.size()
    lo, hi = wilson(k, n)
    axs[1].plot(k.index, k / n, "o-", color=col, label=corr)
    axs[1].fill_between(k.index, lo, hi, color=col, alpha=.15)
axs[0].axhspan(*np.percentile(RND, [25, 75]), color="grey", alpha=.2, label="random pairs (IQR)")
axs[0].axhline(np.median(RND), color="grey", lw=.8)
axs[0].set_ylabel("WordNet distance CNN vs SNN answer"); axs[0].legend(fontsize=7)
axs[1].set_ylabel("share of disagreements in the same group")
for ax in axs:
    ax.set_xlabel("severity"); ax.set_xticks(SEVERITIES)
plt.tight_layout(); plt.show()
gr = np.array(GROUP_OF)
print(f"random pairs: median distance {np.median(RND):.0f}, same group "
      f"{np.mean(gr[a] == gr[b]):.2f}")'''),
    md("## Drill-down\nPick any cell above; with the image subset on disk this shows the images."),
    code(LOOKUP),
    code('''sel = W[(W.corruption == "contrast") & (W.severity == 3) & (W.outcome == "only SNN right")]
print(len(sel), "images where only the SNN is right at contrast 3")
show(sel, title="only SNN right, contrast 3")'''),
    md(findings("01")),
]

# ------------------------------------------------------------------ 02
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md("# 02 Per image: do CNN and SNN break at the same severity?\n\n"
       "No features or images are exported, so each image is described by its **behaviour profile**: "
       "right/wrong for both models in all 15 conditions plus agreement (45 binary values). "
       "The projection groups images that behave alike; the *breaking-severity* view asks per image "
       "and corruption up to which severity each model is still right. Tests H5, H6.\n\n"
       "Caveat (Lecture 03): UMAP cluster sizes and gaps are not meaningful; parameters are printed."),
    code(ENV),
    code(SETUP),
    md("## Breaking severity, paired per image\nFor each image and corruption: the highest severity at "
       "which the model is still right (0 = already wrong at severity 1). The diagonal = both models "
       "break at the same step. Only images that at least one model gets right at severity 1."),
    code('''def last_ok(d, m):
    """Per image: how many severities in a row (from 1) the model is right, and whether it is
    right again after a miss (non-monotonic)."""
    piv = d.pivot_table(index="image_id", columns="severity", values=f"{m}_ok", aggfunc="first")
    ok = piv[SEVERITIES].values.astype(bool)
    run = np.cumprod(ok, axis=1).sum(1)              # 0..5: still right up to this severity
    recover = (np.diff(ok.astype(int), axis=1) > 0).any(1)
    return pd.Series(run, index=piv.index), pd.Series(recover, index=piv.index)

BR = []
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, corr in zip(axs, CORRUPTIONS):
    d = W[W.corruption == corr]
    bc, rc = last_ok(d, "cnn")
    bs, rs = last_ok(d, "snn")
    b = pd.DataFrame({"cnn": bc, "snn": bs, "recover_cnn": rc, "recover_snn": rs}).assign(corruption=corr)
    BR.append(b.reset_index())
    b = b[(b.cnn > 0) | (b.snn > 0)]
    M = pd.crosstab(b.snn, b.cnn).reindex(index=range(6), columns=range(6), fill_value=0)
    im = ax.imshow(M.values, cmap="Greens", origin="lower")
    for (i, j), v in np.ndenumerate(M.values):
        ax.text(j, i, v, ha="center", va="center", fontsize=8, color="w" if v > M.values.max() * .6 else "k")
    diag = np.trace(M.values) / M.values.sum()
    ax.set_title(f"{corr}: same step {diag:.0%} (n={M.values.sum()})")
    ax.set_xlabel("CNN still right up to severity"); ax.set_ylabel("SNN still right up to severity")
plt.tight_layout(); plt.show()
BR = pd.concat(BR, ignore_index=True)'''),
    md("Is the off-diagonal mass symmetric? A paired sign test per corruption: images where the SNN "
       "lasts longer vs the CNN lasts longer."),
    code('''from scipy.stats import binomtest
rows = []
for corr, b in BR.groupby("corruption", sort=False):
    b = b[(b.cnn > 0) | (b.snn > 0)]
    s_longer, c_longer = (b.snn > b.cnn).sum(), (b.cnn > b.snn).sum()
    p = binomtest(s_longer, s_longer + c_longer).pvalue
    rows.append((corr, len(b), (b.snn == b.cnn).mean(), s_longer, c_longer, p,
                 b.recover_cnn.mean(), b.recover_snn.mean()))
pd.DataFrame(rows, columns=["corruption", "n", "same step", "SNN lasts longer", "CNN lasts longer",
                            "sign-test p", "CNN right again after a miss", "SNN right again after a miss"]).round(3)'''),
    md("## Behaviour-profile projection\nUMAP (Hamming distance) of the 45-value profiles. Colour = "
       "how many of the 15 conditions the CNN gets right, and = WordNet group."),
    code('''import umap
P = []
for m in ("cnn_ok", "snn_ok", "agree"):
    P.append(W.pivot_table(index="image_id", columns=["corruption", "severity"], values=m, aggfunc="first"))
PROF = pd.concat(P, axis=1).astype(int)
meta = W.drop_duplicates("image_id").set_index("image_id").loc[PROF.index]
UP = {"n_neighbors": 30, "min_dist": 0.2, "metric": "hamming", "random_state": 0}
Z = umap.UMAP(**UP).fit_transform(PROF.values.astype(bool))
print("UMAP", UP, "| images:", len(PROF), "| dims:", PROF.shape[1])
n_ok = PROF.iloc[:, :15].sum(1)
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
sc = axs[0].scatter(Z[:, 0], Z[:, 1], c=n_ok, cmap="viridis", s=6)
fig.colorbar(sc, ax=axs[0], label="conditions the CNN gets right (of 15)")
cmap = plt.cm.tab20(np.linspace(0, 1, len(GROUPS)))
for g, col in zip(GROUPS, cmap):
    k = (meta.group == g).values
    axs[1].scatter(Z[k, 0], Z[k, 1], s=6, color=col, label=g)
axs[1].legend(fontsize=6, markerscale=2, ncol=2)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()'''),
    md("Does a group sit in its own region? kNN group purity in profile space (15 neighbours, Hamming) "
       "against the purity expected by chance, per group."),
    code('''from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=16, metric="hamming").fit(PROF.values)
_, ix = nn.kneighbors(PROF.values)
g = meta.group.to_numpy(dtype=object)
pur = (g[ix[:, 1:]] == g[:, None]).mean(1)
chance = pd.Series(g).value_counts(normalize=True)
R = pd.DataFrame({"group": g, "purity": pur}).groupby("group").purity.mean().to_frame()
R["chance"] = chance.reindex(R.index)
R["ratio"] = R.purity / R.chance
R.sort_values("ratio", ascending=False).round(3)'''),
    md("## Which groups survive longest?\nMean number of conditions right (of 15) per group, both models."),
    code('''S = meta.assign(cnn=PROF.iloc[:, :15].sum(1).values, snn=PROF.iloc[:, 15:30].sum(1).values)
S.groupby("group")[["cnn", "snn"]].agg(["mean", "count"]).round(2).sort_values(("cnn", "mean"), ascending=False)'''),
    md(findings("02")),
]

# ------------------------------------------------------------------ 03
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    md("# 03 Small multiples: groups × corruption × severity, and energy as an attribute\n\n"
       "Per WordNet group (n = 2 × classes in the group). Energy comes from the AKD1000 power meter "
       "(`results_card/energy.csv`): one measurement per condition, so no interval. Tests H5, H7."),
    code(ENV),
    code(SETUP),
    md("## Accuracy per group, both models\nWilson bands; groups ordered by size."),
    code('''fig, axs = plt.subplots(len(GROUPS), 3, figsize=(12, 26), sharex=True, sharey="row")
for i, g in enumerate(GROUPS):
    for j, corr in enumerate(CORRUPTIONS):
        d = W[(W.group == g) & (W.corruption == corr)].groupby("severity")
        for m, col, ls in (("cnn", "#d62728", "-"), ("snn", "#1f77b4", "--")):
            k, n = d[f"{m}_ok"].sum(), d.size()
            lo, hi = wilson(k, n)
            axs[i, j].plot(k.index, k / n, ls, color=col, marker="o", ms=3, label=m.upper())
            axs[i, j].fill_between(k.index, lo, hi, color=col, alpha=.15)
        if j == 0:
            axs[i, j].set_ylabel(f"{g}\\n(n={int(n.iloc[0])})", fontsize=8)
        if i == 0:
            axs[i, j].set_title(corr)
axs[0, 0].legend(fontsize=7); axs[-1, 1].set_xlabel("severity"); plt.tight_layout(); plt.show()'''),
    md("## Agreement per group × severity\nCell = share of images where CNN and SNN give the same class "
       "(± half-width of the Wilson interval)."),
    code('''fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
for ax, corr in zip(axs, CORRUPTIONS):
    d = W[W.corruption == corr]
    k = d.pivot_table(index="group", columns="severity", values="agree", aggfunc="sum").reindex(GROUPS)
    n = d.pivot_table(index="group", columns="severity", values="agree", aggfunc="size").reindex(GROUPS)
    r = k / n
    lo, hi = wilson(k.values, n.values)
    im = ax.imshow(r.values, cmap="Greens", vmin=.4, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(r.values):
        ax.text(j, i, f"{v:.2f}\\n±{(hi[i, j] - lo[i, j]) / 2:.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(5)); ax.set_xticklabels(SEVERITIES); ax.set_title(corr); ax.set_xlabel("severity")
axs[0].set_yticks(range(len(GROUPS))); axs[0].set_yticklabels(GROUPS, fontsize=8)
fig.colorbar(im, ax=axs, label="CNN = SNN"); plt.show()'''),
    md("## Energy per inference as an attribute of the divergence\nLeft: AKD1000 energy per frame vs "
       "severity. Right: the same numbers against agreement and SNN accuracy of the condition. The "
       "card's floor power (idle, ~924 mW in the hardware check) is included, so most of the energy "
       "is time × floor power, not activity."),
    code('''from scipy.stats import spearmanr

E = table("energy.csv", "card")
print("source:", E.source.iloc[0])
E = E.merge(W.groupby(["corruption", "severity"]).agg(agree=("agree", "mean"), snn=("snn_ok", "mean"))
            .reset_index(), on=["corruption", "severity"])
fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))
for corr, col in zip(CORRUPTIONS, ["#2ca02c", "#ff7f0e", "#1f77b4"]):
    e = E[E.corruption == corr]
    axs[0].plot(e.severity, e.energy_per_inference, "o-", color=col, label=corr)
    axs[1].scatter(e.agree, e.energy_per_inference, color=col, label=corr)
    axs[2].scatter(e.snn, e.energy_per_inference, color=col, label=corr)
    for _, r in e.iterrows():
        axs[1].annotate(int(r.severity), (r.agree, r.energy_per_inference), fontsize=7)
        axs[2].annotate(int(r.severity), (r.snn, r.energy_per_inference), fontsize=7)
axs[0].set_xlabel("severity"); axs[0].set_xticks(SEVERITIES)
axs[1].set_xlabel("agreement CNN = SNN"); axs[2].set_xlabel("SNN top-1 accuracy")
for ax in axs:
    ax.set_ylabel("AKD1000 mJ / frame (incl. floor)")
axs[0].legend(fontsize=8); plt.tight_layout(); plt.show()
for corr in CORRUPTIONS:
    e = E[E.corruption == corr]
    r = spearmanr(e.severity, e.energy_per_inference)
    print(f"{corr:15s} range {e.energy_per_inference.min():.2f}–{e.energy_per_inference.max():.2f} mJ, "
          f"Spearman(severity) = {r.statistic:+.2f}")
print(f"all: max/min = {E.energy_per_inference.max() / E.energy_per_inference.min() - 1:.1%} spread")'''),
    md("Layer specification of the converted model (`results_sim/layers.csv`): context for Phase II. "
       "It lists layer sizes and bit widths only; per-image activity is not exported yet."),
    code('''table("layers.csv", "sim")'''),
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
