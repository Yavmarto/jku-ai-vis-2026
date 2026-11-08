# Hypotheses

## H1: Dogs vs. devices

Dogs get a higher disagreement rate between the Keras model and the Akida model (keras_pred != akida_pred) than devices/instruments, in each corruption and severity. Dog breeds look alike, so many images are near-ties that the conversion rounding can flip.

We reject H1 if dogs are not higher (bootstrap 95% CI) in most of the 15 conditions.

To dig deeper, we look at which breeds the disagreements come from and whether they have a small margin between the top two classes. For this we show the dog classes as a disagreement matrix (Keras prediction vs. Akida prediction), ordered by similarity, next to the same matrix for devices.

## H2: Noise vs. fog and contrast on animals

For animal images, a smaller share of the disagreements under gaussian noise cross superclasses than under fog and contrast. Noise removes texture but keeps the shape, so flips stay among similar species. Fog and contrast remove edges and color.

We reject H2 if the share for noise is not lower than both others (non-overlapping CIs) in most of severities 1-3.

To dig deeper, we look at where the crossing disagreements end up and whether fog and contrast send them to the same few classes. For this we show the superclass confusion of disagreements only, with one panel per corruption.
