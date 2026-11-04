"""Deterministic synthetic corruptions for CIFAR-10.

Values were chosen based on the paper "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" by Hendrycks & Dietterich (2019).
https://arxiv.org/abs/1903.12261
"""

import numpy as np

def apply_clean(images: np.ndarray, severity: int = 0) -> np.ndarray:
    """Returns the original image array without any corruption"""
    return images.copy()

def apply_gaussian_noise(images: np.ndarray, severity: int = 3) -> np.ndarray:
    """Adds Gaussian noise to image array"""

    sigmas = [0.04, 0.08, 0.12, 0.18, 0.26]
    sigma = sigmas[severity - 1]

    rng = np.random.default_rng(seed=42 + severity)
    noise = rng.normal(0, sigma, images.shape)

    normalized = images.astype(np.float32) / 255.0 + noise
    return (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def apply_low_contrast(images: np.ndarray, severity: int = 3) -> np.ndarray:
    """Reduces image contrast towards mean pixel values"""

    factors = [0.75, 0.55, 0.35, 0.20, 0.10]
    factor = factors[severity - 1]

    means = np.mean(images.astype(np.float32), axis=(1, 2, 3), keepdims=True)
    corrupted = (images.astype(np.float32) - means) * factor + means

    return np.clip(corrupted, 0.0, 255.0).astype(np.uint8)


def apply_fog(images: np.ndarray, severity: int = 3) -> np.ndarray:
    """Applies atmospheric transmission fog attenuation"""

    alphas = [0.20, 0.35, 0.50, 0.65, 0.80]
    alpha = alphas[severity - 1]

    fog_layer = np.ones_like(images, dtype=np.float32) * 230.0
    corrupted = (1.0 - alpha) * images.astype(np.float32) + alpha * fog_layer

    return np.clip(corrupted, 0.0, 255.0).astype(np.uint8)

