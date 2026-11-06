"""Regenerated CIFAR-10-C corruptions (fog, gaussian_noise, contrast).

Ported from Hendrycks & Dietterich's make_cifar_c.py with the CIFAR-10 severity constants.
Only for working without the official files (zenodo 2535967): pixels differ from the
official set because the random draws differ. Results made with these are labelled
"regenerated" and are not the official benchmark.
"""
import numpy as np

CORRUPTIONS = ["fog", "gaussian_noise", "contrast"]


def _plasma_fractal(rng, mapsize=32, wibbledecay=3.0):
    """Diamond-square fractal in [0,1], ported from make_cifar_c.py."""
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(a):
        return a / 4 + wibble * rng.uniform(-wibble, wibble, a.shape)

    while stepsize >= 2:
        # squares
        corner = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        sq = corner + np.roll(corner, -1, axis=0)
        sq += np.roll(sq, -1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(sq)
        # diamonds
        dr = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ul = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        lt = dr + np.roll(dr, 1, axis=0) + ul + np.roll(ul, -1, axis=1)
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(lt)
        tt = dr + np.roll(dr, 1, axis=1) + ul + np.roll(ul, -1, axis=0)
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(tt)
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def gaussian_noise(x, severity, rng):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    x = x / 255.0
    return np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255


def contrast(x, severity, rng=None):
    c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]
    x = x / 255.0
    means = x.mean(axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def fog(x, severity, rng):
    c = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x = x / 255.0
    max_val = x.max()
    x = x + c[0] * _plasma_fractal(rng, wibbledecay=c[1])[:32, :32][..., None]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


_FNS = {"gaussian_noise": gaussian_noise, "contrast": contrast, "fog": fog}


def corrupt(images, image_ids, corruption, severity, seed=0):
    """Corrupt a batch. Each image gets its own RNG seeded by (seed, id, corruption, severity),
    so any subset can be regenerated identically later (e.g. for drill-down thumbnails)."""
    if corruption == "clean":
        return images.copy()
    fn = _FNS[corruption]
    cidx = CORRUPTIONS.index(corruption)
    out = np.empty_like(images)
    for i, (img, iid) in enumerate(zip(images, image_ids)):
        rng = np.random.default_rng([seed, int(iid), cidx, severity])
        out[i] = np.uint8(np.round(fn(img.astype(np.float64), severity, rng)))
    return out


