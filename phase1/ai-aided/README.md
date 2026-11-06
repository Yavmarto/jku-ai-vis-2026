# Phase I – AI-aided pair

Views on the shared AkidaNet / ImageNet-C results:
- **CNN:** quantized Keras AkidaNet
- **SNN:** the same weights converted and run on the AKD1000

The data owner produces the results. We only read them:

| Folder | What |
|---|---|
| `../../Phase 1 - No AI/results_card/` | AKD1000 run: `predictions.csv`, `accuracy.csv`, `agreement.csv`, `energy.csv` |
| `../../Phase 1 - No AI/results_sim/` | Simulator run, the same files plus `layers.csv` (layer specs) |

Tool use is logged in `../../ai4vis/ai-interaction-record.md`. Only Claude has been used so far.

## Files

| File | What |
|---|---|
| `hypotheses.md` | H1–H7: expectation (tested), suspected mechanism (guess, untested), verdict |
| `findings/0*.md` | Findings per view. Also pasted into the last cell of each notebook |
| `01_confusion_views.ipynb` | Card vs simulator check, accuracy vs agreement with CIs, sink classes, WordNet-group confusion (CNN / SNN / paired difference), WordNet distance between disagreeing answers, drill-down |
| `02_projection.ipynb` | Paired per-image breaking severity, UMAP of behaviour profiles, group purity, group survival |
| `03_small_multiples.ipynb` | Accuracy per group × corruption with Wilson bands, agreement heatmaps, energy per inference as an attribute |
| `aiv/load.py` | Reads the result CSVs, pairs CNN and SNN per image, WordNet groups and distances |
| `aiv/stats.py` | Wilson intervals and paired (McNemar-style) difference intervals |
| `aiv/imagenet_hierarchy.csv` | 1000 classes: name, one of 14 WordNet groups, hypernym path. Built by `make_hierarchy.py` (needs nltk once) |
| `build_notebooks.py` | Generates the notebooks. Edit this file, not the `.ipynb` |

## Run (from this folder)

```bash
pip install -r requirements.txt
python build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
python build_notebooks.py --strip      # drop TF start-up noise from the outputs
```

`RUN = "card"` in the setup cell switches to `"sim"`. The drill-down shows images if the subset
from `scripts/make_imagenet_subset.py` is in `data/imagenet_subset/`. Otherwise it shows a table.

## Limits of this data

- **No clean condition:** severity 1 is the reference.
- **No confidence scores**, so calibration views are not possible.
- **Two images per class**, so class-level views are noise. Classes are grouped into 14
  WordNet groups.
- **Energy:** one measurement per condition, including floor power.
