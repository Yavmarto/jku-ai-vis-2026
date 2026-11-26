# Phase II – AI-aided pair (Yoshi, Gabriel)

The pairs switched roles after the midpoint: this pair was human-only in Phase I and is
AI-aided in Phase II. Tool use is logged in `../../ai4vis/ai-interaction-record.md` (Phase II
rows).

Question: *why* do the quantized Keras AkidaNet (CNN) and its Akida conversion (SNN) give
different answers on 21–37 % of ImageNet-C images, at almost the same accuracy (Phase I)?

| File | What |
|---|---|
| `claims.md` | Testable claims C1–C3 and L1 with decision rules, written before the full run; verdicts |
| `01_knn_embedding.ipynb` | Technique: kNN in embedding space, combined with CNN confidence. Embedding sanity check, collapse and sinks (C2), near-ties (C1), confidence vs neighbourhood (C3), UMAP maps |
| `findings/01.md` | Findings of notebook 01, also pasted into its last cell |
| `verify_numbers.py` | Recomputes the headline numbers by a separate path (numpy brute-force kNN, rank AUROC) |
| `aiv2/knn.py`, `aiv2/stats.py` | Neighbour features per condition; bootstrap AUROC intervals |
| `export_layer_activity.py` | **Untested draft** for the per-image layer export (needs akida + images, card machine) |
| `TODO-yoshi.md` | What is blocked on the card machine |
| `build_notebooks.py` | Generates the notebook. Edit this file, not the `.ipynb` |

Figures: `../../figures/phase2_ai/` (PNG, plus `umap_interactive.html` with hover).

## Run (from this folder)

```bash
pip install -r requirements.txt
python build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace 01_knn_embedding.ipynb
python verify_numbers.py
```

## Status (26.11)

- kNN technique: done, C1 supported, C2 and C3 partly supported (`findings/01.md`).
- Layer-wise sparsity: blocked on the export (`TODO-yoshi.md`).
