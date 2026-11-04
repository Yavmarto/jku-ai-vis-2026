"""Model plumbing: float MobileNetV2, conversion to Akida, and predictors with one interface.

A predictor maps uint8 images (N,32,32,3) to class probabilities/scores (N,10).
The akida/cnn2snn calls follow the BrainChip docs from memory and are UNVERIFIED until the first
run on the server. Fix them there. Whether MobileNetV2 converts for Akida 1.0 is also unverified;
if not, swap `build_float` for an AkidaNet-style network.
"""
import numpy as np

from .config import MODELS

FLOAT_PATH = MODELS / "float.keras"
AKIDA_PATH = MODELS / "snn.fbz"


def build_float():
    from tensorflow import keras

    # weights=None: ImageNet weights do not exist for 32x32 input, so we train on CIFAR-10.
    return keras.applications.MobileNetV2(input_shape=(32, 32, 3), weights=None, classes=10)


def preprocess(x):
    return x.astype("float32") / 127.5 - 1.0


def to_akida(float_model):
    """Quantize to 4-bit and convert. UNVERIFIED: check args against the cnn2snn docs."""
    import cnn2snn

    quantized = cnn2snn.quantize(
        float_model, weight_quantization=4, activ_quantization=4, input_weight_quantization=8
    )
    return cnn2snn.convert(quantized)


def float_predictor(model):
    return lambda x: model.predict(preprocess(x), batch_size=256, verbose=0)


def akida_predictor(model, map_to_hw):
    if map_to_hw:
        import akida

        model.map(akida.devices()[0])
    # UNVERIFIED: the akida model should take uint8 input and return raw scores per class.
    return lambda x: np.asarray(model.predict(x.astype("uint8"))).reshape(len(x), -1)
