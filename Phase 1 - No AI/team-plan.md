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

| | **You (card and data)** | **Teammate (views)** |
|---|---|---|
| Hardware | Run the chip check, convert the model, run on the card | Nothing needed, no card required |
| Data | Run `lab.evaluate`, push CSVs to `results/` | Get CIFAR-10-C, run the float model on a laptop to start early |
| Views | Projection (UMAP/t-SNE) with drill-down to images | Confusion matrices, small multiples, difference matrix |
| Analysis | Explain the projection patterns in depth | Explain the confusion-matrix patterns in depth |
| Together | Hypotheses, the common direction with the other pair, the process log, the 5-min video | Same |

## Timeline

Dates in bold come from the course. The others are our own targets.

| By | What |
|---|---|
| 05.11 | Chip check done. Moodle entry updated (one entry, no duplicate). Both write hypotheses. Book **one** feedback slot (05.11 or 06.11, booking link on Moodle). |
| 07.11 | Network chosen and converted. Clean accuracy for both models. |
| 09.11 | Corruption runs exported to `results/`. |
| 10.11 | Views built. Patterns explained. Compare with the other pair. |
| **11.11 23:59** | **Midpoint:** public GitHub repo, AI record and GenAI reflection in the repo, 5-min video. |

## How we work

1. **Hypotheses before plots.** Write down what we expect each corruption to break, for example cat/dog or car/truck mix-ups.
2. **Sketch on paper first.** Ask of each view: does it help find a pattern, or only show a score?
3. **Explain every pattern.** For a cluster: why is it dense or sparse, does it have sub-clusters, what makes it different?
4. **Log as we go** in `phase1/human-only/process-log.md`: time spent, dead ends, decisions. We need it for the comparison with the other pair.
5. **Git:** work on a branch, open a PR, wait for CI. Clear notebook outputs before committing. Never push `Research/` or course files.

## Open questions

- What exactly did the 29.10 session say about the pairs and switching? Check the recording.
- Is the shared `lab/` code OK for the human-only pair? It was written with AI help. Ask at the feedback slot.
- Does the card report energy, and can we measure the CNN baseline the same way? The chip check decides this.

# After Phase I

## The project in 3 lines

- We compare a quantized CNN (MobileNetV2) with its SNN version on the BrainChip AKD1000, on CIFAR-10 plus fog, noise and low contrast.
- **Phase I:** black box, described above.
- **Phase II:** open the box and explain the Phase I findings, with at least two techniques.

## Phase II and final

| When | What |
|---|---|
| **12.11** | Lab session on Phase II and the final submission. Then the two pairs **switch roles**: we become the AI-aided pair and start a prompt log. |
| 12.11 | All four agree the Phase II direction and which Phase I findings to follow up. |
| **09.12 23:59** | **Final:** code from both pairs, AI records, 10-min video, slides. A random subset presents live, so keep all four available. |
