import os
import pickle
import tarfile
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
from pathlib import Path
from mobilenetv2_cifar10_utils import get_mobilenetv2_cifar10
import corruptions

def load_cifar10_raw(tar_path: str = "cifar-10-python.tar.gz"): #CIFAR-10 from local archive, if needed download with curl -O https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz (or official https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz)
    if not os.path.exists(tar_path):
        raise FileNotFoundError(
            f"{tar_path} not found. Place the file in the workspace."
        )

    with tarfile.open(tar_path, "r:gz") as tar:

        def get_batch(name):
            member = [m for m in tar.getmembers() if m.name.endswith(name)][0]
            return pickle.load(tar.extractfile(member), encoding="bytes")

        x_te, y_te = (
            get_batch("test_batch")[b"data"],
            get_batch("test_batch")[b"labels"],
        )

    x_test = x_te.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_te).flatten()
    return x_test, y_test


def get_preprocessed_test_data(tar_path: str = "cifar-10-python.tar.gz", num_samples: int = 500): # test images resized to 224x224
    x_test, y_test = load_cifar10_raw(tar_path)
    x_slice = x_test[:num_samples]
    y_slice = y_test[:num_samples]

    x_resized = tf.image.resize(x_slice, [224, 224]).numpy().astype(np.uint8)
    return x_resized, y_slice


sys.path.append(str(Path.cwd().parent / "Phase 1 - No AI"))

WEIGHTS_PATH = "../models/float.keras"  # Replace with path to .h5/.keras weights once teammate completes training, this is a placeholder for now
NUM_SAMPLES = 500    # 500 for fast development; 2000-5000 for final runs?
ARCHIVE_PATH = "cifar-10-python.tar.gz"

model, feat_extractor = get_mobilenetv2_cifar10(WEIGHTS_PATH)
images_clean, labels = get_preprocessed_test_data(ARCHIVE_PATH, num_samples=NUM_SAMPLES)
print(f"Loaded {len(images_clean)} test samples.")

corruptions_map = {
    "clean": corruptions.apply_clean,
    "gaussian_noise": corruptions.apply_gaussian_noise,
    "low_contrast": corruptions.apply_low_contrast,
    "fog": corruptions.apply_fog,
}

severities = [1, 3, 5]  # sweep across low, medium, and high severity
records = []

for c_name, c_func in corruptions_map.items():
    active_severities = [0] if c_name == "clean" else severities
    for sev in active_severities:
        print(f"Corruption: {c_name} Severity {sev}")
        corrupted_imgs = c_func(images_clean, sev)
        
        # normalized to float32 [0, 255] or preprocess_input
        probs = model.predict(corrupted_imgs.astype(np.float32), batch_size=64, verbose=0) 
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        feats = feat_extractor.predict(corrupted_imgs.astype(np.float32), batch_size=64, verbose=0)
        
        for idx in range(len(corrupted_imgs)):
            records.append({
                "sample_id": idx,
                "corruption": c_name,
                "severity": sev,
                "true_label": int(labels[idx]),
                "cnn_pred": int(preds[idx]),
                "cnn_conf": float(confs[idx]),
                "cnn_correct": bool(preds[idx] == labels[idx]),
                "embedding": feats[idx].tolist()
            })

df_results = pd.DataFrame(records)
os.makedirs("../results", exist_ok=True)
output_file = "../results/mobilenetv2_evaluation.parquet"
df_results.to_parquet(output_file, engine="pyarrow")
print(f"Successfully exported {len(df_results)} evaluation rows to {output_file}")

