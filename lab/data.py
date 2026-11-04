"""CIFAR-10 test set (clean) and CIFAR-10-C (fog, gaussian_noise, contrast; 5 severities)."""
import numpy as np

from .config import CORRUPTIONS, DATA, SEVERITIES

N = 10000  # images per severity; CIFAR-10-C stacks severities 1..5 in one 50000 array


def load_clean():
    from tensorflow.keras.datasets import cifar10

    (_, _), (x, y) = cifar10.load_data()
    return x, y.ravel()


def load_corrupted(name, severity):
    root = DATA / "CIFAR-10-C"
    x = np.load(root / f"{name}.npy", mmap_mode="r")
    y = np.load(root / "labels.npy")
    s = slice((severity - 1) * N, severity * N)
    return np.asarray(x[s]), y[s]


def conditions():
    """Yield (corruption, severity, x_uint8, labels). Clean is ("clean", 0)."""
    x, y = load_clean()
    yield "clean", 0, x, y
    for name in CORRUPTIONS:
        for sev in SEVERITIES:
            x, y = load_corrupted(name, sev)
            yield name, sev, x, y
