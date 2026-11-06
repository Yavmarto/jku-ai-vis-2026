"""Load the AkidaNet / ImageNet-C results produced by the data owner (Yoshi's notebooks).

Source folders (read-only for us):
    Phase 1 - No AI/results_card/   AKD1000 run: predictions, accuracy, agreement, energy
    Phase 1 - No AI/results_sim/    software simulator run: same files + layers.csv (layer specs)

predictions.csv: image_id (wnid/file), corruption, severity, label, keras_pred, akida_pred.
2000 ImageNet val images (2 per class, 1000 classes), fog / gaussian_noise / contrast at
severities 1-5. No clean condition and no confidence scores are exported.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = {"card": ROOT / "Phase 1 - No AI" / "results_card",
           "sim": ROOT / "Phase 1 - No AI" / "results_sim"}
CORRUPTIONS = ["fog", "gaussian_noise", "contrast"]
SEVERITIES = [1, 2, 3, 4, 5]
OUTCOMES = ["both right", "only CNN right", "only SNN right", "both wrong, same",
            "both wrong, differ"]
COLUMNS = ["image_id", "corruption", "severity", "label", "keras_pred", "akida_pred"]


def hierarchy():
    """label, wnid, name, group, path (WordNet) for the 1000 classes; see make_hierarchy.py."""
    return pd.read_csv(HERE / "imagenet_hierarchy.csv")


H = hierarchy()
NAMES = H.name.values
GROUP = H.group.values
GROUPS = list(H.group.value_counts().index)  # largest first
_PATHS = [p.split(">") for p in H.path]


def wordnet_distance(a, b):
    """Edges between two classes through their lowest common WordNet ancestor (0 = same)."""
    pa, pb = _PATHS[a], _PATHS[b]
    common = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        common += 1
    return (len(pa) - common) + (len(pb) - common)


def predictions(run="card"):
    df = pd.read_csv(RESULTS[run] / "predictions.csv")
    assert list(df.columns) == COLUMNS, df.columns
    return df


def table(name, run="card"):
    return pd.read_csv(RESULTS[run] / name)


def paired(run="card"):
    """One row per (image, condition): quantized Keras CNN vs Akida SNN, with outcomes."""
    w = predictions(run).rename(columns={"keras_pred": "pred_cnn", "akida_pred": "pred_snn"})
    w["cnn_ok"] = w.pred_cnn == w.label
    w["snn_ok"] = w.pred_snn == w.label
    w["agree"] = w.pred_cnn == w.pred_snn
    w["outcome"] = np.select(
        [w.cnn_ok & w.snn_ok, w.cnn_ok & ~w.snn_ok, ~w.cnn_ok & w.snn_ok, w.agree],
        OUTCOMES[:4], OUTCOMES[4])
    w["true"] = NAMES[w.label]
    w["group"] = GROUP[w.label]
    for m in ("cnn", "snn"):
        w[f"group_{m}"] = GROUP[w[f"pred_{m}"]]
    return w
