"""Shared plumbing for both pairs.

quantizeml/cnn2snn (MetaTF 2.19) only accept tf_keras (Keras 2) models, so every module that
imports TensorFlow must see TF_USE_LEGACY_KERAS=1 before the first `import tensorflow`.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
