"""Check: does argmax(model_akida.forward(x)) give the same class as model_akida.predict(x)?

BrainChip API docs (akida_apis.html): forward returns raw int potentials; predict "obtains the output values
from the model discrete potentials by applying a shift and a scale" (models without last-layer activation).
This compares both against the quantized Keras model on 100 fog/1 images (first image of every 10th class)
and on the 10 BrainChip demo images.

Run on the card machine from ~/akida-try in akida-env, with nothing else holding /dev/akida0:
    python check_forward_vs_predict.py
"""
import os, re
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import akida
from akida_models import akidanet_imagenet_pretrained
from akida_models.imagenet import get_preprocessed_samples
from cnn2snn import convert, set_akida_version, AkidaVersion

FOG_DIR = "imagenet_subset/imagenet-c/fog/1"

with set_akida_version(AkidaVersion.v1):
    model_keras = akidanet_imagenet_pretrained(0.5, quantized=True, bitwidth=4)
    model_akida = convert(model_keras)

devices = akida.devices()
assert devices, "no Akida device (another process holding /dev/akida0? check: fuser -v /dev/akida0)"
device = devices[0]
model_akida.map(device)
device.soc.power_measurement_enabled = True

last = model_akida.layers[-1]
print("last layer:", last.name, type(last).__name__)
try:
    for name in last.variables.names:
        v = np.asarray(last.variables[name])
        print(f"  variable {name}: shape {v.shape}, min {v.min()}, max {v.max()}, distinct {len(np.unique(v))}")
except Exception as e:   # informational only
    print("  could not list variables:", e)

def read_energy():
    m = re.search(r"energy consumed \(mJ/frame\):\s*([0-9.]+)", str(model_akida.statistics))
    return float(m.group(1)) if m else None

def compare(tag, x, y):
    print(f"\n== {tag}: x {x.dtype} {x.shape} min {x.min()} max {x.max()}")
    k = np.asarray(model_keras.predict(x, batch_size=100)).reshape(len(x), -1).argmax(1)
    fwd = np.asarray(model_akida.forward(x))
    e_fwd = read_energy()
    prd = np.asarray(model_akida.predict(x))
    e_prd = read_energy()
    print(f"forward out {fwd.dtype} {fwd.shape} | predict out {prd.dtype} {prd.shape}")
    f = fwd.reshape(len(x), -1).argmax(1)
    p = prd.reshape(len(x), -1).argmax(1)
    print(f"keras   top-1 {np.mean(k == y):.3f}")
    print(f"forward top-1 {np.mean(f == y):.3f}  agree with keras {np.mean(f == k):.3f}")
    print(f"predict top-1 {np.mean(p == y):.3f}  agree with keras {np.mean(p == k):.3f}")
    print(f"forward vs predict agree {np.mean(f == p):.3f}")
    print(f"energy mJ/frame after forward {e_fwd}, after predict {e_prd}")

x_demo, y_demo = get_preprocessed_samples(224, 3)
compare("10 demo images", np.asarray(x_demo), np.asarray(y_demo))

wnids = sorted(d for d in os.listdir(FOG_DIR) if os.path.isdir(os.path.join(FOG_DIR, d)))
assert len(wnids) == 1000, len(wnids)
xs, ys = [], []
for idx in range(0, 1000, 10):
    f = sorted(os.listdir(os.path.join(FOG_DIR, wnids[idx])))[0]
    img = tf.io.decode_jpeg(tf.io.read_file(os.path.join(FOG_DIR, wnids[idx], f)), channels=3).numpy()
    assert img.shape == (224, 224, 3), img.shape
    xs.append(img); ys.append(idx)
compare("100 fog/1 images", np.stack(xs).astype(np.uint8), np.array(ys))
