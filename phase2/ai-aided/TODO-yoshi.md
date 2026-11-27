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
- [ ] **Ask Gabriel where the embedding comes from.** `results/akidanet_imagenetc.parquet` has a
  512-d `embedding` column, but the script that wrote it is not in the repo
  (`view_eval_pipeline.py` in git history is the old MobileNet/CIFAR one). Checked 27.11: the
  values are signed and continuous, so not post-ReLU and not 4-bit quantized; and they do
  **not** match `Phase 1 - No AI/results_card/predictions_with_embeddings.parquet` (Gabriel's
  float-model `layers[-2].input`, 1500 rows): cosine 0.12, same as shuffled. Which model and
  layer, and which preprocessing? Add the script to the repo and one line to `aiv2/knn.py`'s
  docstring. Findings now say "unknown" instead of "quantized Keras, penultimate".
- [ ] **Then:** `02_layer_activity.ipynb` is built (27.11, tested on synthetic data only).
  Copy `layers_activity.csv` into `phase2/ai-aided/results/`, run
  `jupyter nbconvert --to notebook --execute --inplace 02_layer_activity.ipynb`, fix what
  breaks, then write `findings/02.md` against the L1 rules in `claims.md` and re-run
  `python build_notebooks.py` + both notebooks so the findings cell is filled.
- [ ] Agree with Ammar and Omer that the shared Phase II technique is layer sparsity and that
  our second one is kNN in embedding space (chosen 26.11), so the human-only pair picks a
  different second technique.
