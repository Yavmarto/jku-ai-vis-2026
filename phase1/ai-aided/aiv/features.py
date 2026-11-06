"""CNN penultimate (global-average-pool) features for the projection view.

lab.evaluate does not export features, so this reads models/float.keras and the same images as
lab.data, for a fixed subset of test images, and writes data/features/cnn-float.npz (gitignored,
derived data). Only the float CNN is projected; the SNN appears through its predictions.

    python phase1/ai-aided/aiv/features.py --n 2000      (from the repo root)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lab.config import DATA  # noqa: E402

OUT = DATA / "features" / "cnn-float.npz"


def subset(n, seed=0):
    return np.sort(np.random.default_rng(seed).choice(10000, size=n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    from tensorflow import keras

    from lab.data import conditions
    from lab.models import FLOAT_PATH, preprocess

    model = keras.models.load_model(FLOAT_PATH)
    gap = next(layer for layer in model.layers if "GlobalAveragePooling" in type(layer).__name__)
    feat = keras.Model(model.inputs, gap.output)
    ids = subset(a.n)
    F, C, S = [], [], []
    for corruption, sev, x, _ in conditions():
        F.append(feat.predict(preprocess(x[ids]), batch_size=256, verbose=0).astype(np.float16))
        C += [corruption] * len(ids)
        S += [sev] * len(ids)
        print(corruption, sev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, features=np.concatenate(F), image_id=np.tile(ids, len(F)),
                        corruption=np.array(C), severity=np.array(S))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
