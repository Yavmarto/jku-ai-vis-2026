"""Load results/<model>/{accuracy,predictions}.csv as written by lab.evaluate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.config import CLASSES, CSV_COLUMNS, DATA, RESULTS  # noqa: E402

OUTCOMES = ["both right", "only CNN right", "only SNN right", "both wrong, same",
            "both wrong, differ"]


def runs():
    """Model run names that have a predictions.csv."""
    return sorted(p.parent.name for p in RESULTS.glob("*/predictions.csv"))


def pick(cnn=None, snn=None):
    """Default pair: cnn-float vs snn-hw if it exists, else snn-sim."""
    have = runs()
    cnn = cnn or "cnn-float"
    snn = snn or ("snn-hw" if "snn-hw" in have else "snn-sim")
    missing = [m for m in (cnn, snn) if m not in have]
    if missing:
        raise FileNotFoundError(f"no results/<model>/predictions.csv for {missing}; have {have}")
    return cnn, snn


def predictions(model):
    df = pd.read_csv(RESULTS / model / "predictions.csv")
    assert list(df.columns) == CSV_COLUMNS["predictions.csv"], df.columns
    return df


def accuracy(model):
    return pd.read_csv(RESULTS / model / "accuracy.csv")


def data_source():
    marker = DATA / "CIFAR-10-C" / "REGENERATED.txt"
    return "regenerated CIFAR-10-C (not official)" if marker.exists() else "official CIFAR-10-C"


def paired(cnn, snn):
    """One row per (image, condition): both models side by side.

    `confidence` is a softmax probability for the float CNN but a raw potential for Akida, so the
    two are not on one scale. `rank_*` is the within-model, within-condition percentile of
    confidence, which is comparable between models; use it instead of raw confidence.
    """
    key = ["image_id", "corruption", "severity", "label"]
    a = predictions(cnn).drop(columns="model")
    b = predictions(snn).drop(columns="model")
    w = a.merge(b, on=key, suffixes=("_cnn", "_snn"), validate="one_to_one")
    for m in ("cnn", "snn"):
        w[f"rank_{m}"] = w.groupby(["corruption", "severity"])[f"confidence_{m}"].rank(pct=True)
    w["cnn_ok"] = w.pred_cnn == w.label
    w["snn_ok"] = w.pred_snn == w.label
    w["agree"] = w.pred_cnn == w.pred_snn
    w["outcome"] = np.select(
        [w.cnn_ok & w.snn_ok, w.cnn_ok & ~w.snn_ok, ~w.cnn_ok & w.snn_ok, w.agree],
        OUTCOMES[:4], OUTCOMES[4])
    w["true"] = np.array(CLASSES)[w.label]
    return w
