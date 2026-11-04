"""Shared constants. No heavy imports here, CI imports this file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"  # CIFAR-10-C goes in data/CIFAR-10-C/ (gitignored, download it yourself)
MODELS = ROOT / "models"
RESULTS = ROOT / "results"  # results/<model name>/{accuracy,predictions}.csv

CORRUPTIONS = ["fog", "gaussian_noise", "contrast"]  # CIFAR-10-C file names
SEVERITIES = [1, 2, 3, 4, 5]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# `model` is the run name and says where it ran, e.g. cnn-float, snn-sim, snn-hw.
# Never mix simulator and hardware rows under one name.
CSV_COLUMNS = {
    "accuracy.csv": ["model", "corruption", "severity", "n_images", "top1"],
    # Phase I needs per-image outputs (confusion matrices, agreement, confidence).
    "predictions.csv": [
        "image_id", "model", "corruption", "severity", "label", "pred", "confidence",
    ],
    "energy.csv": ["model", "corruption", "severity", "energy_per_inference", "source"],
    "layers.csv": [
        "image_id", "corruption", "severity", "layer_index", "layer_name",
        "nonzero_count", "total_units",
    ],
}
