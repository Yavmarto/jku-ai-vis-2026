import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def get_mobilenetv2_cifar10(weights_path=None):
    """Returns MobileNetV2 with a 10-class head.

    If weights_path is None, returns an initialized/placeholder model.
    """

    inputs = layers.Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)
    base_model.trainable = False

    # final label only tells you that the model made a mistake, penultimate embedding shows you why
    x = layers.GlobalAveragePooling2D(name="penultimate_features")(base_model.output) # Penultimate pooling layer used for latent feature extraction
    outputs = layers.Dense(10, activation="softmax", name="cifar10_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded trained weights from: {weights_path}")
    else:
        print("UNTRAINED placeholder model for pipeline scaffolding.")

    # extractor model for UMAP latent projections
    feat_extractor = models.Model(inputs=model.input, outputs=model.get_layer("penultimate_features").output,)
    return model, feat_extractor