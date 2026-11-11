# Hypotheses

## H1: Dogs vs. devices

Dogs get a higher disagreement rate between the Keras model and the Akida model (keras_pred != akida_pred) than devices/instruments, in each corruption and severity. Dog breeds look alike, so many images are near-ties that the conversion rounding can flip.

We reject H1 if dogs are not higher (bootstrap 95% CI) in most of the 15 conditions.

To dig deeper, we look at which breeds the disagreements come from and whether they have a small margin between the top two classes. For this we show the dog classes as a disagreement matrix (Keras prediction vs. Akida prediction), ordered by similarity, next to the same matrix for devices.

## H2: Noise vs. fog and contrast on animals

For animal images, a smaller share of the disagreements under gaussian noise cross superclasses than under fog and contrast. Noise removes texture but keeps the shape, so flips stay among similar species. Fog and contrast remove edges and color.

We reject H2 if the share for noise is not lower than both others (non-overlapping CIs) in most of severities 1-3.

To dig deeper, we look at where the crossing disagreements end up and whether fog and contrast send them to the same few classes. For this we show the superclass confusion of disagreements only, with one panel per corruption.


## H3: Fog creates background-driven error sub-clusters (UMAP View)
Under high severity fog, the latent embeddings of misclassified images do not just form one random mass; they form distinct, dense sub-clusters grouped by shared background colors (e.g., all sky/outdoor backgrounds) rather than primary objects. 

We reject H1 if: The UMAP error clusters do not visually correlate with shared background traits.

To dig deeper: We will isolate the densest error clusters in the 2D projection, inspect the original input images within them, and determine exactly what visual features (like color palettes or backgrounds) distinguish these dense sub-clusters from the sparser regions of the latent space.


## H4: Low contrast forces a systemic collapse into "textureless" fallback classes (Confusion Matrix View)
The top classes most degraded by low contrast will systematically shift their predictions into a few specific "fallback" classes that naturally lack fine internal edges or details.

We reject H4 if: The difference matrix shows the false predictions spreading uniformly across many classes rather than concentrating in 1 or 2 distinct columns.

To dig deeper: We will identify the specific fallback classes dominating the columns of the difference matrix and inspect the images to understand the shared visual properties (e.g., smooth shapes, gray tones) that cause the model's behavior to default to them.