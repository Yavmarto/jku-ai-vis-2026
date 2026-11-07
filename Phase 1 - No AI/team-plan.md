# Team plan

Short version for the two of us. Full details are in `akida-ai-vis-proposal.md`; where things live is in `README.md`.

# Phase I: our pair (human-only)

Everything for our own work until the midpoint on 11.11.

## Goal

Treat the model as a black box. Find where the CNN and the SNN behave differently, and explain why. A table of scores is not enough.

## Rules that affect us

| Rule | What it means for us |
|---|---|
| We are the **human-only pair** | No generative AI for ideas, views or code. Allowed: light code completion (not generation) and AI for finding or understanding documentation. |
| The other pair is **AI-aided** | They use ChatGPT/Claude and keep a prompt log. Same model and task as us. |
| Common direction first, then separate work | We agree questions, data and starting view with all four, then each pair works on its own. Don't copy their pipeline. |
| Notebooks are fine | Views go in notebooks. Shared code stays in `.py`. |

## Who does what

| | **Yoshi (card and data)** | **Gabriel (views)** |
|---|---|---|
| Hardware | Convert AkidaNet for Akida 1.0, run it on the card | Nothing needed, no card required |
| Data | Card and simulator CSVs in `results_card/` and `results_sim/`. Generate `results/akidanet_imagenetc.parquet` (confidence + embeddings) with `view_eval_pipeline.py`, see `RUN_PIPELINE.md` | Read the CSVs and the parquet; no model runs needed |
| Views | Projection (UMAP/t-SNE) with drill-down to images | Confusion matrices, small multiples, difference matrix |
| Analysis | Explain the projection patterns in depth | Explain the confusion-matrix patterns in depth |
| Together | Hypotheses, the common direction with the other pair, the process log, the 5-min video | Same |

## Timeline

Dates in bold come from the course. The others are our own targets.

| By | What |
|---|---|
| 05.11 | Chip check done. **Done.** Moodle entry updated (one entry, no duplicate): **open**, still names MobileNetV2. Both write hypotheses: **open**. |
| 07.11 | Network chosen and converted: **done** (AkidaNet, 05.11). Clean accuracy for both models: **open**, cite BrainChip's published 61.30% until measured. |
| 09.11 | Corruption runs exported: **done** (`results_card/`, `results_sim/`). Parquet for the projection view: **done** (07.11). |
| 10.11 | Views run on the AkidaNet data, patterns explained, energy shown in a view. Compare with the other pair. |
| **11.11 23:59** | **Midpoint:** public GitHub repo, AI record and GenAI reflection in the repo, 5-min video. |

## How we work

1. **Hypotheses before plots.** Write down what we expect each corruption to break, for example cat/dog or car/truck mix-ups.
2. **Sketch on paper first.** Ask of each view: does it help find a pattern, or only show a score?
3. **Explain every pattern.** For a cluster: why is it dense or sparse, does it have sub-clusters, what makes it different?
4. **Log as we go** in `Phase 1 - No AI/process-log.md`: time spent, dead ends, decisions. We need it for the comparison with the other pair.
5. **Git:** work on a branch, open a PR, wait for CI. Clear notebook outputs before committing. Never push `Research/` or course files.

## Open questions

- What exactly did the 29.10 session say about the pairs and switching? Check the recording.
- Energy: the card reports about 84-85 mJ/frame (incl. floor power) for the Akida model. The Keras baseline runs on the CPU, so there is no matching number for it. Check whether the dynamic part varies across conditions before reading anything into it.
- Clean baseline: can the ImageNet validation images go on the card machine before 11.11?

# After Phase I

## The project in 3 lines

- We compare BrainChip's pretrained, quantized AkidaNet (Keras) with its Akida 1.0 conversion on the AKD1000, on ImageNet-C fog, Gaussian noise and contrast, severities 1-5. (MobileNetV2 was dropped: it only converts for Akida 2.0.)
- **Phase I:** black box, described above.
- **Phase II:** open the box and explain the Phase I findings, with at least two techniques.

## Phase II and final

| When | What |
|---|---|
| **12.11** | Lab session on Phase II and the final submission. Then the two pairs **switch roles**: we become the AI-aided pair and start a prompt log. |
| 12.11 | All four agree the Phase II direction and which Phase I findings to follow up. |
| **09.12 23:59** | **Final:** code from both pairs, AI records, 10-min video, slides. A random subset presents live, so keep all four available. |
