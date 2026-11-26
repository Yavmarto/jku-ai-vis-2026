# TODO (Yoshi, card machine): Phase II layer-wise activity

Blocked here because `akida` has no macOS wheels and the ImageNet-C subset is only on the card
machine. Everything else in Phase II (AI-aided) runs without it.

- [ ] **Run the layer export.** On the card machine, in the Phase I environment:

  ```bash
  python phase2/ai-aided/export_layer_activity.py --check
  python phase2/ai-aided/export_layer_activity.py
  ```

  The script is an **untested Claude draft** (AI record, Phase II row 5). `--check` runs 8
  images and stops on the first wrong assumption: layer names and output dims vs
  `results_sim/layers.csv`, the cut model equal to the full model, and predictions equal to
  Phase I. If `pop_layer()` or `forward()` behave differently in akida 2.19.3, fix the
  `truncated_models` function only, and log the fix in the AI record (tag: invented-API if
  the call does not exist).
  Output: `phase2/ai-aided/results/layers_activity.csv`, about 420 000 rows
  (30 000 images x 14 layers). Commit it.
- [ ] **Confirm where the embedding comes from.** `results/akidanet_imagenetc.parquet` has a
  512-d `embedding` column, but the script that wrote it (`view_eval_pipeline.py`) is not in
  the repo. Which model (quantized Keras / float / Akida) and which layer? Add the script or a
  line to `aiv2/knn.py`'s docstring. The kNN findings assume quantized Keras, penultimate.
- [ ] **Then (with Claude):** build `02_layer_activity.ipynb`. Test L1 in `claims.md`: heatmap of
  layer x severity coloured by non-zero share, one per corruption; and, within each condition,
  last-layer activity of disagreeing vs agreeing images, and vs kNN consensus (combines the two
  techniques).
- [ ] Agree with Ammar and Omer that the shared Phase II technique is layer sparsity and that
  our second one is kNN in embedding space (chosen 26.11), so the human-only pair picks a
  different second technique.
