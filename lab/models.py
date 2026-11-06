"""Model plumbing: float MobileNetV2, conversion to Akida, and predictors with one interface.

A predictor maps uint8 images (N,32,32,3) to class scores (N,10).

Checked on 2026-11-04 with akida/cnn2snn 2.19.3, quantizeml 1.2.5, TF 2.19 (simulator only):
- The model takes raw 0-255 input; scaling to [-1, 1] is a Rescaling layer inside the model, so
  the Akida model gets uint8 directly. A nested model (Sequential wrapping MobileNetV2) is not
  accepted by quantizeml, so the network is built flat.
- No softmax at the end: the model outputs logits (quantizeml does not quantize softmax).
- `cnn2snn.quantize` (legacy quantization API) fails on MobileNetV2 ("block_2_add must have only
  one inbound layer"). quantizeml.quantize + cnn2snn.convert works and gives an Akida 2.0 model.
- Converting for Akida 1.0 (the AKD1000 card) fails on the residual Add layers too. MobileNetV2
  cannot run on the AKD1000; snn-hw needs an Akida 1.0 network without skip connections.

    python -m lab.models          # models/float.keras -> models/snn.fbz (Akida 2.0)
"""
import numpy as np

from .config import MODELS

FLOAT_PATH = MODELS / "float.keras"
AKIDA_PATH = MODELS / "snn.fbz"


def build_float():
    from tensorflow import keras

    # weights=None: ImageNet weights do not exist for 32x32 input, so we train on CIFAR-10.
    inp = keras.Input((32, 32, 3))
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inp)
    net = keras.applications.MobileNetV2(
        input_tensor=x, weights=None, classes=10, classifier_activation=None)
    # Keras sets BN momentum 0.999 (tuned for ImageNet fine-tuning). Training from scratch, the
    # moving averages lag so far behind that validation stayed at chance (0.10) for 4 epochs.
    for layer in net.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.momentum = 0.9
    return keras.Model(inp, net.output, name="mobilenetv2_cifar10")


def preprocess(x):
    return x.astype("float32")  # 0-255; the model rescales


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_akida(float_model, samples, weight_bits=4, activation_bits=4, input_weight_bits=8):
    """Post-training quantization with quantizeml, then conversion to an Akida 2.0 model.
    `samples` are uint8 training images used for calibration."""
    from cnn2snn import convert
    from quantizeml.models import QuantizationParams, quantize

    qp = QuantizationParams(input_weight_bits=input_weight_bits, weight_bits=weight_bits,
                            activation_bits=activation_bits)
    quantized = quantize(float_model, qparams=qp, samples=preprocess(samples),
                         num_samples=len(samples), batch_size=64)
    return convert(quantized)


def float_predictor(model):
    """Softmax probabilities, so `confidence` is a probability for the CNN."""
    return lambda x: softmax(model.predict(preprocess(x), batch_size=256, verbose=0))


def akida_predictor(model, map_to_hw):
    if map_to_hw:
        import akida

        model.map(akida.devices()[0])
    # Dequantized output potentials, shape (N,1,1,10) -> (N,10). Not probabilities.
    return lambda x: np.asarray(model.predict(x.astype("uint8"))).reshape(len(x), -1)


def main(n_samples=1024):
    from tensorflow import keras

    from .data import load_train

    x, _ = load_train()
    samples = x[np.random.default_rng(0).choice(len(x), n_samples, replace=False)]
    akida_model = to_akida(keras.models.load_model(FLOAT_PATH), samples)
    akida_model.summary()
    akida_model.save(str(AKIDA_PATH))
    print("wrote", AKIDA_PATH)


if __name__ == "__main__":
    main()
