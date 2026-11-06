"""Train the float baseline on CIFAR-10 and save models/float.keras.
Run: python -m lab.train [--epochs 30]"""
import argparse

from .config import MODELS
from .data import load_clean, load_train
from .models import FLOAT_PATH, build_float, preprocess


def main(epochs=30):
    import tensorflow as tf
    from tensorflow import keras

    x, y = load_train()
    xt, yt = load_clean()
    aug = keras.Sequential([keras.layers.RandomFlip("horizontal"),
                            keras.layers.RandomTranslation(0.1, 0.1)])
    train = (tf.data.Dataset.from_tensor_slices((preprocess(x), y)).shuffle(50000, seed=0)
             .batch(128).map(lambda a, b: (aug(a, training=True), b),
                             num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    model = build_float()
    model.compile(keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(
                      1e-3, decay_steps=epochs * len(train))),
                  keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(train, epochs=epochs, validation_data=(preprocess(xt), yt), batch_size=256)
    MODELS.mkdir(exist_ok=True)
    model.save(FLOAT_PATH)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    main(p.parse_args().epochs)
