# AI and Visualization Lab: quantized AkidaNet vs its Akida SNN on ImageNet-C

Proposal: `akida-ai-vis-proposal.md`. CI: `docs/ci.md`.

We compare BrainChip's pretrained, quantized **AkidaNet** (ImageNet, alpha 0.5, 224x224, 4-bit) as
a Keras CNN with the same model converted by MetaTF to **Akida 1.0** and run on the **AKD1000** card,
on ImageNet-C fog, Gaussian noise and contrast at severities 1 to 5. No training.

MobileNetV2 on CIFAR-10-C was the original plan. It was dropped because MobileNetV2 converts only
for Akida 2.0; the AKD1000 is Akida 1.0 (see the proposal, "Task and models").

## What goes where

| Path | What | Who |
|---|---|---|
| `Model prep/` | Load the pretrained AkidaNet, convert for Akida 1.0, save `akidanet_imagenet_a05_224_v1.fbz`, check it on the card. `check_forward_vs_predict.py` shows why class scores come from `predict`, not `forward`. | Data owner |
| `scripts/make_imagenet_subset.py` | Cuts 2 images per class per condition out of the ImageNet-C archives. | Data owner |
| `Phase 1 - No AI/akidanet-imagenetc-{card,sim}-executed.ipynb` | Evaluation runs: Keras baseline vs converted model, on the card and on the simulator. | Data owner |
| `Phase 1 - No AI/results_card/` | AKD1000 run: `accuracy.csv`, `agreement.csv`, `energy.csv`, `predictions.csv`. | Data owner |
| `Phase 1 - No AI/results_sim/` | Simulator run: `accuracy.csv`, `agreement.csv`, `predictions.csv`, `layers.csv` (layer metadata). | Data owner |
| `Phase 1 - No AI/` (rest) | Human-only pair: `team-plan.md`, `process-log.md`, view notebooks. | Human-only pair |
| `ai4vis/` | AI interaction record and GenAI reflection (midpoint deliverables). On `main`. | AI-aided pair |
| `data/`, `models/*.fbz` | ImageNet-C subset and converted models. Gitignored. | local |
| `Research/` | Course material. Gitignored, never push. | local |

Run names in the CSVs: `akidanet_keras_quantized` (CNN baseline, CPU), `akidanet_akida_akd1000`
(card), `akidanet_akida_simulator` (software simulator). Never mix card and simulator rows.

## Pipeline

```bash
# once, where the ImageNet-C archives are
python scripts/make_imagenet_subset.py --c-dir <archives> --out data/imagenet_subset --n-per-class 2
# convert and check on the card: run Model prep/akidanet-model-prep.ipynb
# evaluate: run Phase 1 - No AI/akidanet-imagenetc-card.ipynb (card) or -sim.ipynb (simulator)
```

Environment of the card runs: Python 3.11, TensorFlow 2.19.1, akida/cnn2snn 2.19.3,
quantizeml 1.2.5, akida_models 1.14.2, `TF_USE_LEGACY_KERAS=1`.

## Known gaps (06.11)

- **No clean baseline yet.** The clean ImageNet validation images are not on the card machine.
  BrainChip publishes 61.30% clean top-1 for this model (cited, not measured).
- **No confidence column.** `predictions.csv` has `keras_pred` and `akida_pred` only. Views that
  colour by confidence need the class scores exported too.
- **Two result formats.** These CSVs do not match the `lab/` schema on `main`
  (`results/<model>/predictions.csv` with `pred` and `confidence`, CIFAR-10 classes). `lab/` and the
  AI-aided pair's views (PR #1) are still on MobileNetV2/CIFAR-10. Agree one format before the views
  are rebuilt.
- **Human-only pair's views use the old setup.** `view_eval_pipeline`, `view_confusion` and
  `view_projections` were built on MobileNetV2/CIFAR-10 with synthetic corruptions, and
  `results/mobilenetv2_evaluation.parquet`, `models/float.keras` and `figures/phase1_human/` come from a
  model at chance accuracy (11% clean). Point the views at `results_card/` / `results_sim/` and
  regenerate the figures.
- `layers.csv` holds layer metadata, not per-image `nonzero_count` yet (Phase II).
