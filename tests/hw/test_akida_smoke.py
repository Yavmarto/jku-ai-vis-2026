"""Smoke tests that need the AKD1000 card. Run with `pytest -m hw`.

The akida calls below follow the Akida 1.x Python API from memory and are unverified until the
first run on the server. Fix them there; the first green run is the hardware check.
"""
import os
from pathlib import Path

import pytest

akida = pytest.importorskip("akida")

pytestmark = pytest.mark.hw


def test_device_visible():
    devices = akida.devices()
    assert devices, "no Akida device visible to the akida package"
    print("devices:", devices)


def test_model_maps_and_predicts_one_image():
    path = os.environ.get("AKIDA_SMOKE_MODEL")
    if not path or not Path(path).is_file():
        pytest.skip("set AKIDA_SMOKE_MODEL to a .fbz file to run the one-image check")

    import numpy as np

    model = akida.Model(path)
    model.map(akida.devices()[0])
    x = np.zeros((1, *model.input_shape), dtype=np.uint8)
    out = model.predict(x)
    assert out.shape[0] == 1
    # Does the card report power or energy? Read this from the log after the first run.
    print("statistics:", model.statistics)
