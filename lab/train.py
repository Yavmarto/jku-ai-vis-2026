"""Train the float baseline on CIFAR-10 and save models/float.keras.
Run: python -m lab.train"""
from .config import MODELS
from .data import load_clean
from .models import FLOAT_PATH, build_float, preprocess


def main(epochs=30):
    from tensorflow import keras

    (x, y), _ = keras.datasets.cifar10.load_data()
    model = build_float()
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    xt, yt = load_clean()
    model.fit(preprocess(x), y.ravel(), epochs=epochs, validation_data=(preprocess(xt), yt))
    MODELS.mkdir(exist_ok=True)
    model.save(FLOAT_PATH)


if __name__ == "__main__":
    main()
