# AI and Visualization Lab: CNN vs Akida SNN on corrupted CIFAR-10

Proposal: `akida-ai-vis-proposal.md`. CI: `docs/ci.md`.

## What goes where

| Path | What | Who |
|---|---|---|
| `lab/` | Shared plumbing: config, data loading, models, train, evaluate. Same for both pairs. | Data owner |
| `results/<model>/` | `accuracy.csv`, `predictions.csv` per run. Model name says where it ran (`cnn-float`, `snn-sim`, `snn-hw`). | Data owner |
| `phase1/human-only/` | Our pair (human-only): team plan, process log, notebooks. Do not copy the other pair's pipeline. | Us |
| `phase1/ai-aided/` | The AI-aided pair's analysis and views. | Other pair |
| `ai4vis/` | AI interaction record and GenAI reflection (midpoint deliverables). | AI-aided pair, plus both for the comparison |
| `data/`, `models/` | Downloaded CIFAR-10-C and trained models. Gitignored. | local |
| `Research/` | Course material. Gitignored, never push. | local |

## Pipeline

```bash
python -m lab.train                       # float MobileNetV2 -> models/float.keras
python -m lab.evaluate cnn-float          # -> results/cnn-float/
# on the server, after conversion (lab.models.to_akida) wrote models/snn.fbz:
flock /tmp/akida.lock python -m lab.evaluate snn-hw --akida --hw
```

`energy.csv` and `layers.csv` are not produced yet. Add them after the hardware check shows
what the SDK reports. The akida/cnn2snn calls in `lab/models.py` are unverified.
