"""Write a regenerated CIFAR-10-C to data/CIFAR-10-C/ in the official layout, so lab.data reads it
unchanged: <corruption>.npy is uint8 (50000, 32, 32, 3), severities 1..5 stacked, and
labels.npy is the test labels repeated 5 times. A REGENERATED.txt marker records that these are
not the official files. Use the official download instead whenever it is available.

    python phase1/ai-aided/make_cifar10c.py        (from the repo root)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiv.corruptions import CORRUPTIONS, corrupt  # noqa: E402
from lab.config import DATA, SEVERITIES  # noqa: E402
from lab.data import load_clean  # noqa: E402


def main():
    out = DATA / "CIFAR-10-C"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "labels.npy").exists() and not (out / "REGENERATED.txt").exists():
        sys.exit(f"{out} holds the official files; not overwriting them")
    x, y = load_clean()
    ids = np.arange(len(x))
    for name in CORRUPTIONS:
        arr = np.concatenate([corrupt(x, ids, name, s) for s in SEVERITIES])
        np.save(out / f"{name}.npy", arr)
        print(name, arr.shape)
    np.save(out / "labels.npy", np.tile(y, len(SEVERITIES)).astype(np.uint8))
    (out / "REGENERATED.txt").write_text(
        "Regenerated with phase1/ai-aided/aiv/corruptions.py (make_cifar_c.py formulas, "
        "CIFAR-10 constants, seed 0). Not the official CIFAR-10-C files.\n")


if __name__ == "__main__":
    main()
