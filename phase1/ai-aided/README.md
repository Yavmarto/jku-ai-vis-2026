# Phase I – AI-aided pair

Same task, models and data contract as everyone else: we read what `lab.evaluate` writes to
`results/<model>/` and add views on top. Tool use is logged in `../../ai4vis/ai-interaction-record.md`.
So far only Claude has been used.

## Files

| File | What |
|---|---|
| `hypotheses.md` | H1–H7: expectation (tested) and suspected mechanism (a guess, untested) |
| `findings/0*.md` | Findings per view, also pasted into the last cell of each notebook |
| `01_confusion_views.ipynb` | Error matrices per corruption × severity for CNN, SNN and the paired difference (significant cells marked), class order by error similarity, named pairs with Wilson bands, sink classes, image drill-down |
| `02_projection.ipynb` | UMAP of CNN features coloured by outcome and class. Border test (kNN label purity) and drift test in the original feature space, drill-down |
| `03_small_multiples.ipynb` | Outcome composition per class, per-class SNN−CNN accuracy gap with paired CIs, disagreement heatmaps, confidence AUROC |
| `aiv/load.py` | Reads `results/<model>/predictions.csv`, pairs CNN and SNN per image |
| `aiv/stats.py` | Wilson intervals and paired (McNemar-style) difference intervals |
| `aiv/features.py` | CNN pool features for the projection → `data/features/cnn-float.npz` (gitignored) |
| `aiv/corruptions.py`, `make_cifar10c.py` | Regenerates CIFAR-10-C in the official layout, only when the official files are not available |
| `build_notebooks.py` | Generates the notebooks. Edit this file, not the `.ipynb` |

## Run (from the repo root)

```bash
pip install -r requirements-akida.txt -r phase1/ai-aided/requirements.txt   # Python 3.10-3.12
python -m lab.train                     # models/float.keras
python -m lab.models                    # models/snn.fbz (Akida 2.0)
python -m lab.evaluate cnn-float
python -m lab.evaluate snn-sim --akida
python phase1/ai-aided/aiv/features.py
python phase1/ai-aided/build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace phase1/ai-aided/0*.ipynb
```

The notebooks pick `snn-hw` over `snn-sim` when both exist. `aiv.load.pick(cnn, snn)` overrides that.

## The run behind the committed outputs and findings

- **Data:** CIFAR-10 test set (MD5 matches torchvision). CIFAR-10-C was **regenerated** with
  `make_cifar10c.py` because the official download was blocked. Re-run with the official files
  before quoting numbers.
- **Models:** `cnn-float` is MobileNetV2 trained here with the fixed `lab.train` (30 epochs).
  `snn-sim` is its 4/4/8-bit post-training quantization converted to **Akida 2.0** and run on
  the akida software simulator.
- **Not covered:** `snn-hw` (MobileNetV2 does not convert for Akida 1.0 / AKD1000, see
  `lab/models.py`), energy (H7), and per-layer activity (`layers.csv`, Phase II).
