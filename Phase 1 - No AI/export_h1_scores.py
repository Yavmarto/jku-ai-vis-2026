"""Export top-two quantized Keras scores for the H1 image groups."""

import hashlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
from cnn2snn import AkidaVersion, load_quantized_model, set_akida_version

if len(sys.argv) != 4:
    raise SystemExit('Usage: python export_h1_scores.py PHASE_DIRECTORY CHECKPOINT OUTPUT_CSV')

phase = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
output = Path(sys.argv[3])
image_root = phase / 'imagenet_subset/imagenet-c'
keys = ['image_id', 'corruption', 'severity']
predictions = pd.read_csv(phase / 'results_card/predictions.csv')
superclasses = pd.read_csv(phase / 'imagenet_superclasses.csv').set_index('label')['superclass']
selected = predictions[predictions.label.map(superclasses).isin(['dog', 'device/instrument'])]

expected_sha256 = '1d9493115d43625f2644f8265f71b8487c4019047fc331e892a233a3d6520371'
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha256:
    raise ValueError('Expected the original 4-bit AkidaNet ImageNet checkpoint.')
with set_akida_version(AkidaVersion.v1):
    model = load_quantized_model(str(checkpoint), compile_model=False)


def load_image(path):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.resize_with_crop_or_pad(image, 224, 224)


results = []
for (corruption, severity), group in selected.groupby(['corruption', 'severity'], sort=False):
    paths = [str(image_root / corruption / str(severity) / image_id) for image_id in group.image_id]
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(load_image, num_parallel_calls=4).batch(250).prefetch(1)
    scores = np.concatenate([np.asarray(model.predict_on_batch(images.numpy())) for images in dataset])
    assert scores.shape == (len(group), 1000) and np.isfinite(scores).all()
    top1 = scores.argmax(axis=1)
    row = np.arange(len(scores))
    best = scores[row, top1].copy()
    scores[row, top1] = -np.inf
    top2 = scores.argmax(axis=1)
    table = group[keys].assign(keras_top1=top1, keras_top1_score=best,
                               keras_top2=top2, keras_top2_score=scores[row, top2])
    print(f'{corruption} {severity}: {np.sum(top1 == group.keras_pred.to_numpy())}/{len(group)} top-1 predictions reproduced', flush=True)
    results.append(table)

with TemporaryDirectory(dir=output.parent) as staging_dir:
    staged_output = Path(staging_dir) / output.name
    pd.concat(results, ignore_index=True).to_csv(staged_output, index=False)
    staged_output.replace(output)
