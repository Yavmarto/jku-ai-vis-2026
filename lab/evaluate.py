"""Run a model over clean + corrupted CIFAR-10 and write accuracy.csv and predictions.csv to
results/<name>/.

  python -m lab.evaluate cnn-float                # float Keras model, any machine
  python -m lab.evaluate snn-sim  --akida         # converted model on the akida software simulator
  python -m lab.evaluate snn-hw   --akida --hw    # converted model on the AKD1000 (server only)
Take the chip lock first on the server: flock /tmp/akida.lock python -m lab.evaluate ...
"""
import argparse
import csv

import numpy as np

from .config import CSV_COLUMNS, RESULTS
from .data import conditions
from .models import AKIDA_PATH, FLOAT_PATH, akida_predictor, float_predictor


def write(path, name, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS[name])
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--akida", action="store_true")
    p.add_argument("--hw", action="store_true")
    a = p.parse_args()

    if a.akida:
        import akida

        predict = akida_predictor(akida.Model(str(AKIDA_PATH)), a.hw)
    else:
        from tensorflow import keras

        predict = float_predictor(keras.models.load_model(FLOAT_PATH))

    acc, preds = [], []
    for corruption, sev, x, y in conditions():
        scores = predict(x)
        pred = scores.argmax(axis=1)
        conf = scores.max(axis=1)  # a probability for float; raw potential for akida (not comparable)
        acc.append([a.name, corruption, sev, len(y), float(np.mean(pred == y))])
        preds += [
            [i, a.name, corruption, sev, int(y[i]), int(pred[i]), float(conf[i])]
            for i in range(len(y))
        ]
        print(corruption, sev, acc[-1][-1])

    out = RESULTS / a.name
    out.mkdir(parents=True, exist_ok=True)
    write(out / "accuracy.csv", "accuracy.csv", acc)
    write(out / "predictions.csv", "predictions.csv", preds)


if __name__ == "__main__":
    main()
