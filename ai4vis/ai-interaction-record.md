# AI interaction record (written by the AI-aided pair)

One log for the AI-aided pair. There is no second log in `phase1/ai-aided/`.
Tool so far: Claude only. If ChatGPT is used, log it here too.

| Phase | Tool | Important prompt | What came back | Accepted / modified / rejected | Why | Effect on later decisions |
|---|---|---|---|---|---|---|
| I | Claude | Repo link + "what is there to do here? I was told Phase 1 AI-guided, Phase 2 not AI-guided" | Summary of proposal, team plan and deadlines; said we are the pair that switches to human-only on 12.11 | Accepted | Matched the course setup | Defined our scope: Phase I views + this record |
| I | Claude | "yes do my part" | Own folder `Phase 1 - AI/` with its own evaluate script, own CSV schema, synthetic demo data with injected structure, 3 notebooks, a second prompt log, draft hypotheses | **Rejected** after review | Did not match the shared `lab/` contract: model names (`cnn_quantized` vs `cnn-float`/`snn-sim`/`snn-hw`), column `conf` vs `confidence`, flat `results/` vs `results/<model>/`. Invented Akida energy calls. Demo labels did not match the demo images. Injected cat↔dog structure made the views prove nothing. Never ran on a real model. No CIs, no findings, empty verdicts | Rebuilt on top of `lab/`. Rule from now on: read the shared code before generating anything, and never show synthetic data in the views |
| I | Claude | Review list (7 mismatches, 9 gaps) + "verify that everything at least works, use the simulator" | Rebuild in `phase1/ai-aided/` reading `results/<model>/`. Ran the shared pipeline end to end (train → convert → `snn-sim` → views). See the checks below | Accepted, after the checks below | Every claim was checked against a run, not against docs | Findings and verdicts are from a real run. The hardware blocker went to the team |
| I | Claude | Teammate review: "you did not update it to use the switch to AkidaNet and ImageNet-C" | Rebased onto `yoshi-phase-1` and rebuilt the three views on `results_card/` + `results_sim/` (1000 classes, no clean, no confidence, energy on the card). Dropped the CIFAR/MobileNet pipeline and the `lab/` changes | Accepted, after the checks below | The first PR was built on `main` only. The AI never looked at the other branches, so it missed the switch | New rule: before building, list all remote branches and ask which one the team builds on. Findings and verdicts redone on the card data |
| I | Claude | "Write the AI record and the GenAI reflection from what happened" | Drafts of this record and of `genai-reflection.md`, built from the session and the checks tables | Modified | Reviewed against what actually happened. The comparison with the human-only pair is added by us after the pair meeting | Records are kept as we go, not reconstructed at the end |

## Checks and fixes made during row 3

| What the AI had claimed or written | Check | Result | Failure tag |
|---|---|---|---|
| Akida energy via `model.statistics.energy` (first version) | Read the installed akida 2.19.3 API and ran the simulator | The simulator `statistics` only reports fps. No energy on the simulator. Code removed, H7 marked untestable | invented-API |
| `lab.models.to_akida` uses `cnn2snn.quantize(...)` (shared code, also AI-assisted) | Ran it on MobileNetV2 | Fails: `block_2_add must have only one inbound layer`. Replaced by `quantizeml.quantize` + `cnn2snn.convert` | invented-API / outdated API |
| MobileNetV2 can run on the AKD1000 | Converted under `set_akida_version(AkidaVersion.v1)` | Fails on the residual Add layers. Converts only for Akida 2.0. The AKD1000 (Akida 1.0) cannot run it. Raised with the team | overclaim |
| `lab.build_float` (softmax head, scaling outside the model) | Conversion | quantizeml rejects the nested model; Akida needs uint8 input. Moved Rescaling into a flat model, logits out | other |
| `lab.train` trains MobileNetV2 from scratch | Watched validation | val_accuracy stayed at 0.10 for 4 epochs. Cause: Keras BN momentum 0.999. Checked with an A/B on 10k images (0.999: val 0.10; 0.9: val 0.34) and fixed | other |
| `lab` imports Keras 3, quantizeml needs tf_keras | Import test | Set `TF_USE_LEGACY_KERAS=1` in `lab/__init__.py` | other |
| Comparing CNN `confidence` (softmax) with SNN `confidence` (raw potential) | Read `lab.evaluate` | Not on one scale. Views use AUROC and within-model ranks instead | wrong-aggregation (avoided) |
| Views without uncertainty, about 200 images per class in the first version | Review | Full test set (about 1000 per class per condition). Wilson intervals on rates, paired intervals on CNN−SNN differences, bootstrap on AUROC. Non-significant difference cells are left unmarked | hides-pattern / overclaim |
| Drift test pooled over severities (AI's first notebook version) | Thought about confounding | Drift rises with severity, so pooling mixes the two. Redone within each severity. At severity 5 the curve flattens; finding 02 §3 changed accordingly | wrong-aggregation |
| Drill-down default "cat→dog, contrast 5" (from the hypothesis, not the data) | Ran it | Selected 1 image. Changed to the largest measured divergence (dog→deer, 427 images) | hides-pattern |
| Every number in `findings/*.md` | Recomputed from `results/*/predictions.csv` in a separate script, not read off the plots | Two wordings corrected: noise-3 gap, frog disagreement range | overclaim (caught) |
| Rebuild of the notebooks after an edit | Checked file sizes | The rebuild wiped the executed outputs of 01 and 03. Re-executed all three | other |
| CIFAR-10-C source | Download blocked in the session | Regenerated with the `make_cifar_c.py` formulas, marked `REGENERATED.txt`. Visually checked a sample grid. Not the official set | — |
| CIFAR-10 source | Downloaded through a GitHub mirror | MD5 of `test_batch` and `data_batch_1/5` match torchvision's published checksums | — |

## Checks and fixes made during row 4

| What the AI had claimed or written | Check | Result | Failure tag |
|---|---|---|---|
| "MobileNetV2 can't run on the AKD1000, so `snn-hw` needs a new network" (PR 1) | The team had already switched to the pretrained Akida 1.0 AkidaNet on `yoshi-phase-1` | The claim was right, but the work was stale. The AI had looked only at `main` | other (stale context) |
| Class-level confusion matrices (as in the CIFAR version) | 1000 classes × 2 images per class | Meaningless at that size. Replaced with 14 WordNet groups and WordNet distance between answers | hides-pattern (avoided) |
| Labels = index of the wnid in sorted order (assumed) | Checked against all 1000 wnids in `predictions.csv` | Holds | — |
| Card and simulator give different predictions (assumed: show both) | Compared all 30 000 rows | Akida: identical. The Keras baseline differs on 92 rows between the two runs. Reported, card run used | — |
| Group "clothing" (first hierarchy) | Read the top sink classes | prayer rug and doormat sat under "clothing". Renamed the group "clothing/covering" | misleading label |
| Hypotheses H1–H7 | Written after a first look at the aggregate numbers | Marked as not blind in `hypotheses.md`. H1/H4 are treated as descriptions | overclaim (avoided) |
| "Same top class in 14 of 15 conditions" (draft) | Recounted | 13 of 15. Corrected | overclaim (caught) |
| "Disagreement 21 % at severity 1" (draft) | Recomputed per corruption | 21–26 %. Corrected | overclaim (caught) |
| Energy as a divergence signal | Read the card notebook: floor 924 mW of ~958 mW | Spread is 1.4 % with one measurement per condition. Reported as a tiny effect, not used as a signal | overclaim (avoided) |
| Every number in `findings/*.md` | Recomputed in a separate script from the CSVs | Matches after the two corrections above | — |

# Phase II (AI-aided pair: Yoshi, Gabriel)

The pairs switched roles after the midpoint. From here on this log is written by Yoshi and
Gabriel, who were the human-only pair in Phase I. Tool so far: Claude (Claude Code, Opus).

| Phase | Tool | Important prompt | What came back | Accepted / modified / rejected | Why | Effect on later decisions |
|---|---|---|---|---|---|---|
| II | Claude | "Please execute Phase 2, I am on the AI side" | Read proposal, team plan, both Phase I logs and all branches first (rule from Phase I row 4). Found the switch (we are now AI-aided) and the blocker: per-image layer activity needs `akida` (no macOS wheels) and the ImageNet-C images, neither on this laptop. Asked instead of guessing | Accepted | The AI stopped at the blocker instead of producing views on synthetic data (Phase I row 2 failure) | Layer export moved to `TODO-yoshi.md` |
| II | Claude | Choice offered: kNN in embedding space / Grad-CAM / wait for the other pair. We picked kNN | kNN on the 512-d embeddings already in `results/akidanet_imagenetc.parquet`, no model run needed | Accepted | Runs on existing data; ties to the Phase I projection work | Second technique fixed on 26.11; the other pair should pick a different one |
| II | Claude | (continuation) prototype before the notebook | Per-condition AUROCs for confidence, neighbour consensus and kNN distance. Then wrote `claims.md` with decision rules | Modified | Claims were fixed *after* the prototype. Noted in `claims.md` as an honesty note; the mechanisms come from Phase I, written before | Verdicts follow the written rules even where that makes them "partly" |
| II | Claude | (continuation) build notebook, findings | `01_knn_embedding.ipynb`, `aiv2/`, `findings/01.md`, figures in `figures/phase2_ai/` incl. interactive UMAP | Accepted after checks below | Every number checked by a second script | C1 supported, C2/C3 partly |
| II | Claude | (continuation) layer export | `export_layer_activity.py`, a draft using `akida.Model(path)`, `pop_layer()`, `forward()` | **Untested**, kept as a draft | akida could not be installed here, so the calls are unverified. Built-in `--check` asserts against `layers.csv` and Phase I predictions so a wrong call fails loudly | To run on the card machine; result goes into this table |
| II | Claude | "Please continue the work you can continue" (27.11) | (1) Traced the embedding source: compared the parquet with Gabriel's float-model extraction, found the earlier "quantized Keras penultimate" assumption wrong. (2) Wrote L1 decision rules before any layer data. (3) Built `02_layer_activity.ipynb` and ran it on a synthetic CSV with the export schema | Accepted; rules modified | The synthetic run showed a floor effect in rule (c) (a layer already dead at sev 2 cannot drop further, so contrast looked *least* affected) and that a constant last layer gives a fake AUROC of 0.5. Both rules fixed before real data. Synthetic figures deleted, never committed | Findings say "unknown model and layer"; question to Gabriel in the TODO |
| II | Claude | "Can you use the simulator instead?" (card server wiped, 27.11) | Checked PyPI: akida 2.19.3 has no macOS wheel but a Linux arm64 one, so the simulator runs in Docker on the Mac. Found the ImageNet-C subset missing locally and that `scripts/make_imagenet_subset.py` rebuilds it deterministically from Zenodo. Tried to SSH to a lab host to look for the images; blocked by the tool's permission check, and then asked us instead | Accepted; SSH attempt rejected | Wheel claim checked by an actual download, not by docs. Probing another machine without asking was wrong | Layer export can run on the simulator once the images are back; server rebuild only needed for card runs |
| II | Claude | "Write a prompt for the Omarchy agent to install all of this" | Setup prompt for the wiped card server (Arch): driver, Python 3.11 env with the Phase I versions from the README, repo, dataset, final check. First version included the GitHub runner; we removed it (set up in Phase I, never used) | Modified | Versions come from the README; the driver repo name is a guess and the prompt says so | Driver name to be confirmed by the server agent (checks table) |

## Checks and fixes made during Phase II rows 3–4

| What the AI had claimed or written | Check | Result | Failure tag |
|---|---|---|---|
| The parquet's predictions are the card run | Merged with `results_card/predictions.csv` on image, corruption, severity | Identical on all 30 000 rows (both models) | — |
| The embedding is the quantized Keras penultimate layer | Generator script not in the repo. Checked values: 512-d, 40 % negative | Plausible, not proven. Stated as an assumption in `knn.py` and the findings; question to the data owner in the TODO. **27.11: wrong.** Values are continuous (not 4-bit) and do not match the only extraction code in the repo (float model, cosine 0.12 = shuffled). Wording changed to "unknown model and layer" | overclaim (caught late) |
| The embedding carries class information | Same-class partner among 15 nbrs vs chance | 21–25 % vs 0.75 % at severity 1 | — |
| "SNN answer is supported by neighbours" (raw share) | A popular class is common among neighbours anyway, especially with sinks | Added a frequency-matched random-class baseline; effect holds in all 15 conditions | wrong-aggregation (avoided) |
| "Neighbour consensus explains disagreement at high severity" | Could just be sink membership | Recomputed on non-sink images only: AUROC still 0.58–0.66 at severity 4–5 | hides-pattern (avoided) |
| "Consensus overtakes confidence at severity 5" (from the line chart) | Paired bootstrap of the AUROC difference | Intervals include 0 (noise 5, contrast 5). Reworded to "catches up" | overclaim (caught) |
| "Combining both beats either" | Applied the written rule per condition | Holds in 8 of 15 only, never at severity 1. Verdict partly | overclaim (caught) |
| Two counts in the first draft of `findings/01.md` (6 of 15; 48–79 %) | Recounted from the notebook tables | 7 of 15 fail; 48–71 %. Corrected | overclaim (caught) |
| UMAP legend | Looked at the PNG | Sink marker missing (legend drawn on a panel without sinks). Replaced with a figure-level legend | misleading label |
| Every number in `findings/01.md` | `verify_numbers.py`: numpy brute-force kNN, rank AUROC, no shared code | Matches the notebook | — |
| Driver repo `akida_dw_edma` (in the server setup prompt) | Not checked yet; the prompt tells the server agent to verify it in BrainChip's docs | Open | possible invented-API |
| Notebook 02 runs and the L1 rules work | Ran on a **synthetic** CSV with the export schema (exception to our Phase I rule "no synthetic data in a view"; no synthetic figure kept) | Runs end to end. Found two flawed rules: floor effect in (c), constant last layer giving AUROC 0.5 in (d). Both fixed before real data. Re-check on a small real run (`--check`) | hides-pattern (caught) |
