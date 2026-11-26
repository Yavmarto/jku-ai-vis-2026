"""Per-image, per-layer activity of the converted AkidaNet (Akida simulator). DRAFT, UNTESTED.

Written by Claude without akida installed (no macOS wheels), so the akida calls below are not
verified. Run `--check` first: it stops with an AssertionError if any assumption is wrong.

Output: phase2/ai-aided/results/layers_activity.csv with the schema from the proposal:
    image_id, corruption, severity, layer_index, layer_name, nonzero_count, total_units

How: for each layer L, a copy of the .fbz model is cut after L with `pop_layer()`, and
`forward()` returns that layer's integer activations. Counting non-zeros per image gives the
activity. Simulator only: Phase I showed the card and the simulator give identical predictions
on all 30 000 rows, and the simulator needs no card time.

    # on the card machine, env from README (TF_USE_LEGACY_KERAS=1, akida 2.19.3)
    python phase2/ai-aided/export_layer_activity.py --check            # 8 images, asserts
    python phase2/ai-aided/export_layer_activity.py                    # all 15 conditions
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import akida  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FBZ = ROOT / "Model prep" / "akidanet_imagenet_a05_224_v1.fbz"
C_DIR = ROOT / "data" / "imagenet_subset" / "imagenet-c"
PRED = ROOT / "Phase 1 - No AI" / "results_card" / "predictions.csv"
LAYERS = ROOT / "Phase 1 - No AI" / "results_sim" / "layers.csv"
OUT = Path(__file__).resolve().parent / "results" / "layers_activity.csv"
CORRUPTIONS = ["fog", "gaussian_noise", "contrast"]
SEVERITIES = [1, 2, 3, 4, 5]
BATCH = 100


def load_images(paths):
    # ImageNet-C images are already 224x224 uint8 (same as the Phase I notebook)
    x = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths])
    assert x.shape[1:] == (224, 224, 3), x.shape
    return x


def truncated_models(n_layers):
    """One model per layer index L, ending at layer L. UNVERIFIED: relies on
    akida.Model(path) and Model.pop_layer()."""
    models = []
    for L in range(n_layers):
        m = akida.Model(str(FBZ))
        while len(m.layers) > L + 1:
            m.pop_layer()
        models.append(m)
    return models


def activity(models, x):
    """nonzero_count per image for each layer -> array (n_layers, n_images)."""
    out = []
    for m in models:
        a = np.asarray(m.forward(x))
        out.append(a.reshape(len(x), -1).astype(bool).sum(1))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="8 images of fog 1, assertions only")
    args = ap.parse_args()

    spec = pd.read_csv(LAYERS)
    full = akida.Model(str(FBZ))
    n = len(full.layers)
    assert n == len(spec), f"{n} layers in the .fbz, {len(spec)} in layers.csv"
    models = truncated_models(n)
    for L, m in enumerate(models):
        assert m.layers[-1].name == spec.name[L], (L, m.layers[-1].name, spec.name[L])
        dims = "x".join(map(str, m.output_shape))
        assert dims == spec.output_dims[L], (L, dims, spec.output_dims[L])
    total = [int(np.prod(m.output_shape)) for m in models]

    pred = pd.read_csv(PRED)
    conds = [(c, s) for c in CORRUPTIONS for s in SEVERITIES]
    if args.check:
        conds = conds[:1]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT if not args.check else OUT.with_suffix(".check.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "corruption", "severity", "layer_index", "layer_name",
                    "nonzero_count", "total_units"])
        for c, s in conds:
            p = pred[(pred.corruption == c) & (pred.severity == s)].reset_index(drop=True)
            if args.check:
                p = p.head(8)
            for i in range(0, len(p), BATCH):
                b = p.iloc[i:i + BATCH]
                x = load_images([C_DIR / c / str(s) / iid for iid in b.image_id])
                act = activity(models, x)
                if args.check:
                    # the last cut model must equal the full model, and the full model must
                    # reproduce the Phase I Akida predictions
                    last = np.asarray(models[-1].forward(x)).reshape(len(x), -1)
                    ref = np.asarray(full.forward(x)).reshape(len(x), -1)
                    assert np.array_equal(last, ref), "cut model != full model"
                    yhat = np.asarray(full.predict(x)).reshape(len(x), -1).argmax(1)
                    assert (yhat == b.akida_pred.values).all(), (yhat, b.akida_pred.values)
                for j, iid in enumerate(b.image_id):
                    for L in range(n):
                        w.writerow([iid, c, s, L, spec.name[L], int(act[L, j]), total[L]])
            print(f"{c} {s}: {len(p)} images")
    if args.check:
        print("check passed: layer names, output dims, cut == full, predictions match Phase I")


if __name__ == "__main__":
    main()
