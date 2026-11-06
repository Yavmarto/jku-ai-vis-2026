"""CIFAR-10 test set (clean) and CIFAR-10-C (fog, gaussian_noise, contrast; 5 severities)."""
import numpy as np

from .config import CORRUPTIONS, DATA, SEVERITIES

N = 10000  # images per severity; CIFAR-10-C stacks severities 1..5 in one 50000 array


def _local_batches(names):
    """Read CIFAR-10 python batches from data/cifar-10-batches-py/ (for machines without
    internet). Returns None if the folder is not there."""
    import pickle

    root = DATA / "cifar-10-batches-py"
    if not root.is_dir():
        return None
    xs, ys = [], []
    for n in names:
        with (root / n).open("rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        ys.append(np.array(d[b"labels"], dtype=np.uint8))
    return np.concatenate(xs), np.concatenate(ys)


def load_train():
    local = _local_batches([f"data_batch_{i}" for i in range(1, 6)])
    if local is not None:
        return local
    from tensorflow.keras.datasets import cifar10

    (x, y), _ = cifar10.load_data()
    return x, y.ravel()


def load_clean():
    local = _local_batches(["test_batch"])
    if local is not None:
        return local
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
