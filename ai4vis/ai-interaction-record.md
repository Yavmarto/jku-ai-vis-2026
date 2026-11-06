# AI interaction record (written by the AI-aided pair)

One log for the AI-aided pair. There is no second log in `phase1/ai-aided/`.
Tool so far: Claude only. If ChatGPT is used, log it here too.
Rows 1–3 were drafted by Claude from the session transcript. Ammar has to confirm or correct
them before the midpoint.

| Phase | Tool | Important prompt | What came back | Accepted / modified / rejected | Why | Effect on later decisions |
|---|---|---|---|---|---|---|
| I | Claude | Repo link + "what is there to do here? I was told Phase 1 AI-guided, Phase 2 not AI-guided" | Summary of proposal, team plan and deadlines; said we are the pair that switches to human-only on 12.11 | Accepted | Matched the course setup | Defined our scope: Phase I views + this record |
| I | Claude | "yes do my part" | Own folder `Phase 1 - AI/` with its own evaluate script, own CSV schema, synthetic demo data with injected structure, 3 notebooks, a second prompt log, draft hypotheses | **Rejected** after review | Did not match the shared `lab/` contract: model names (`cnn_quantized` vs `cnn-float`/`snn-sim`/`snn-hw`), column `conf` vs `confidence`, flat `results/` vs `results/<model>/`. Invented Akida energy calls. Demo labels did not match the demo images. Injected cat↔dog structure made the views prove nothing. Never ran on a real model. No CIs, no findings, empty verdicts | Rebuilt on top of `lab/`. Rule from now on: read the shared code before generating anything, and never show synthetic data in the views |
| I | Claude | Review list (7 mismatches, 9 gaps) + "verify that everything at least works, use the simulator" | Rebuild in `phase1/ai-aided/` reading `results/<model>/`. Ran the shared pipeline end to end (train → convert → `snn-sim` → views). See the checks below | Accepted, after the checks below | Every claim was checked against a run, not against docs | Findings and verdicts are from a real run. The hardware blocker went to the team |

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
