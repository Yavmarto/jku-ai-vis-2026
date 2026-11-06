# AI and Visualization Lab: Akida project proposal

Issue: CEL-1192 · Updated 2026-11-06 (switched to AkidaNet on ImageNet-C) · Replaces `akida-ai-vis-proposal-v2.md` (deleted). `akida-ai-vis-course-feasibility.md` stays for reference.

## The idea in one paragraph

We compare a **quantized CNN baseline** with its **SNN-converted counterpart** running on a real BrainChip AKD1000, using ImageNet-C images with three corruptions: fog, Gaussian noise and contrast. Phase I investigates where and for which inputs the two models diverge, using accuracy degradation, confidence and on-chip energy per inference for both models. Phase II explains that divergence by looking inside the converted model, using layer-wise activity/sparsity as a proxy for "spike death" under corruption. AI4VIS logs how ChatGPT/Claude help us build these visualizations and reports the failure patterns and best practices we find.

## Scope (what is core, what is stretch)

| Part | Core deliverable | Stretch / limitation |
|---|---|---|
| Phase I | Clean + fog/noise/low-contrast accuracy for both models, degradation vs severity, measured on-chip energy per inference for both | Extra corruptions (blur, JPEG, occlusion) |
| Phase II | Layer-wise activity/sparsity per corruption and severity, used as a proxy for spike death | Full spike-raster visualization, only if MetaTF/Akida makes it easy; otherwise reported as a limitation |
| AI4VIS | Log of important prompts and accepted/rejected/refined decisions, with reasons, while ChatGPT/Claude help build the Phase I/II views. Report of failure patterns and best practices. | Qualitative test of whether LLMs can design event-based (neuromorphic) visualizations, if time allows |

## Dates (from the course PDF)

| Date | What | Status |
|---|---|---|
| 21.10 | Register group of 4 on Moodle | Past. Group informed, one entry only. |
| 28.10 | Register task and model on Moodle | Past. Update the existing entry if already registered, do not create a second one. |
| 05.11 / 06.11 | Feedback sessions (not mandatory) | 10-min slots, Thu 05.11 12:00-15:15 (S3 055 and online) and Fri 06.11 08:30-10:00 (online). The booking link is posted on Moodle. A team (both pairs) can book only one slot. |
| 11.11 23:59 | Midpoint submission (end of Phase I) | Next. Public GitHub repo link, AI interaction record for Phase I (in the repo), reflection on GenAI usage (in the repo), recorded progress video of at most 5 min (over the limit costs points). |
| 09.12 23:59 | Final submission (end of Phase II) | Complete code from both pairs, AI interaction records from both phases, 10-min recorded presentation plus slide deck (over the limit costs points). |

Final presentation: the last two lab sessions are live presentations. A random subset of projects presents live instead of playing the recording, and the whole group or one member can present. Keep all four available. Source: Moodle assignment text. The 29.10 session itself is not in our files.

## Task and models

- **Task/model:** BrainChip's pretrained, quantized **AkidaNet** (ImageNet, alpha 0.5, 224x224, 8-bit first layer, 4-bit weights and activations, `akidanet_imagenet_pretrained(0.5, quantized=True, bitwidth=4)`) vs its converted Akida 1.0 model on the AKD1000 via MetaTF. **No training.**
- **Baseline:** the quantized Keras AkidaNet (run name `akidanet_keras_quantized`). Same weights as the converted model, so the difference between the two is the conversion, not quantization.
- **Converted model:** the same network converted with `cnn2snn.convert` inside `set_akida_version(AkidaVersion.v1)` and mapped to the chip (`akidanet_akida_akd1000`). The simulator run of the same `.fbz` is `akidanet_akida_simulator`. Class scores come from `predict`, not `forward` (see `Model prep/check_forward_vs_predict.py`).
- **Data:** ImageNet-C (Zenodo record 2235448): fog, Gaussian noise and contrast, severities 1 to 5. Cut to 2 images per class (2000 images per condition) with `scripts/make_imagenet_subset.py`.
- **Why not MobileNetV2 on CIFAR-10-C (the original plan):** MobileNetV2 does not convert for Akida 1.0 (the AKD1000). Keras MobileNet's depthwise blocks and residual Add layers are not supported there; it only converts for Akida 2.0, which the card is not. AkidaNet is BrainChip's own Akida 1.0 network and comes pretrained and quantized, so this is the fallback the plan named.
- **Clean baseline:** not measured yet, because the clean ImageNet validation images are not on the card machine. BrainChip publishes 61.30% clean top-1 for this model; cite it as published, not measured, until we run the clean images.

## Phase I: accuracy degradation and energy

1. Evaluate both models on clean test images. Record top-1 accuracy. (Open: needs the ImageNet validation images, see Task and models.)
2. Evaluate both models on fog, noise and low-contrast at severities 1 to 5.
3. Report accuracy per corruption and severity, and the drop relative to clean accuracy.
4. Report measured on-chip energy per inference for the converted model, from the AKD1000 power meter (`model.statistics`, mJ/frame including floor power). The Keras baseline runs on the CPU and has no on-chip energy; if we report anything for it, label the method and do not compare the two as if measured alike.

Outputs: `accuracy.csv` (`model, corruption, severity, n_images, top1`), `agreement.csv` (`corruption, severity, n_images, agreement`), `energy.csv` (`model, corruption, severity, energy_per_inference, source`) and `predictions.csv` (`image_id, corruption, severity, label, keras_pred, akida_pred`). Currently in `Phase 1 - No AI/results_card/` (AKD1000) and `results_sim/` (simulator). Both pairs read these files; neither pair runs the models in its own notebooks.

## Phase II: layer-wise activity as a spike-death proxy

- For the converted model, record per layer the share of non-zero activations (sparsity) for each corruption and severity, for the same images used in Phase I.
- Visualize activity per layer against severity. The question is whether activity fades toward the last layers as corruption grows ("spike death"), or whether it behaves differently per corruption.
- State a testable claim up front, for example "low contrast makes activity die out before the last layers", and report it even if it turns out false.
- **Two techniques:** Moodle requires each pair to use at least two explanation techniques in Phase II and to compare or combine them. Layer-wise sparsity is one. Pick a second that fits a conv net, for example saliency/gradient maps on the float model, or k-nearest neighbors in embedding space. Decide this at the start of Phase II.
- **Spike rasters:** attempt only if MetaTF/Akida exposes per-timestep or per-event data without much effort. Otherwise list it as a stretch goal and limitation.

Wording note: the AKD1000 computes only on non-zero activations, but the values are small integers, not single-bit spikes. Phase II measures "sparse activity", and we call it a proxy for spike death in the write-up.

Output: `layers.csv` (`image_id, corruption, severity, layer_index, layer_name, nonzero_count, total_units`).

## AI4VIS: LLM support for visualization design and analysis

The course asks us to record how generative AI supports and influences the design and analysis process, across Phase I, Phase II and the presentation.

**Pair setup (Moodle):** the group splits into two pairs. The **AI-aided pair** uses ChatGPT/Claude as an active part of the process (brainstorming, interpreting views, generating views and code). The **human-only pair** uses no generative AI and relies on its own reasoning and code. It may still use AI lightly for code completion (not generation) and for searching and understanding documentation. Both pairs work on the same model and task. **The pairs switch roles at the end of Phase I.**

- At the start of each phase all four agree a common direction: main questions, data, starting visualization, which Phase I findings go to Phase II.
- Then the pairs work separately. They may talk, but must not make the two pipelines identical, so the differences can be discussed.
- At the end of each phase the pairs compare process and findings: insights one pair missed, different visualization choices, breadth and depth, speed versus verification effort, confidence and its evidence. Not a contest.
- The AI-aided pair is expected to build richer interaction or broaden the analysis.

**Records (AI-aided pair, each phase):**

- Keep a prompt log from day one: important prompts, tool (ChatGPT or Claude), result, accepted/rejected/refined, and why.
- Use the LLMs for sketching views, writing plotting code, annotating results and transforming data, including the Phase I and Phase II views above.
- Report failure patterns (wrong aggregation, misleading color scales, invented API calls, charts that hide the pattern) and the best practices we arrive at for using AI effectively.
- Stretch: ask the LLMs to propose and critique visualizations specifically for event-based, per-layer sparsity data, and report qualitatively how well that works. Layer activity is unusual data, so there is little for the model to lean on.

## Visualization

Phase I views are investigative tools with the black box closed. The course marks a chart that only shows a performance metric as the wrong approach, so accuracy numbers go into the views below and are not plotted alone. Plain Python notebooks (Plotly/Altair or matplotlib) reading the three CSV files. Suggested views:

- Confusion matrices per corruption and severity for both models, plus a difference matrix. With 1000 ImageNet classes a full matrix is unreadable: group classes (for example WordNet superclasses) or show the top-k confused pairs, then reorder by error similarity to reveal blocks.
- Projection (UMAP or t-SNE) of the test images or model embeddings, colored by whether CNN and SNN agree and by confidence, with drill-down to the images. Report the parameters (perplexity or neighbors) and do not read cluster sizes as meaningful (Lecture 03).
- Small multiples by corruption and severity, so patterns can be compared across cases.
- Energy per inference shown as an attribute of these views, not as a separate metric chart.
- Hypotheses from the training data and class similarity (for example cat/dog or car/truck), tested against the confusion matrices.
- Phase II: heatmap of layer index x severity, colored by non-zero share, one per corruption, linked to the Phase I divergences it should explain.

Group decision on the exact stack. The `akida` package has a software simulator, so members without the card can regenerate small samples. Label simulator and hardware numbers separately.

## Risks and checks

- **Hardware check:** done 05.11. AkidaNet maps to the AKD1000, `predict` returns class scores, and `model.statistics` reports energy per frame (about 84-85 mJ/frame including floor power on the ImageNet-C runs). See `Model prep/akidanet-model-prep.ipynb` and `Phase 1 - No AI/run_card.log`.
- **Energy:** measured for the converted model only. The quantized Keras baseline runs on the CPU, so "on-chip energy for both" is not possible. If the baseline gets an energy number, label the method and do not compare the two as if measured alike. The near-constant mJ/frame across conditions is mostly floor power; check whether the dynamic part varies before reading anything into it.
- **Card vs Keras agreement is about 0.75-0.8 on fog 1.** Disagreements are conversion rounding on images where Keras's top two classes are nearly tied. Top-1 accuracy matches to within about half a point. This divergence is the Phase I subject, not a bug.
- **"SNN-converted":** MetaTF converts a quantized CNN into an event-based Akida model with low-bit integer activations. It is not a classic spiking network. Use the phrase as the course framing, but describe precisely what the chip does.
- **Environment:** the card runs used Python 3.11, TensorFlow 2.19.1, akida/cnn2snn 2.19.3, quantizeml 1.2.5, akida_models 1.14.2, with `TF_USE_LEGACY_KERAS=1`. Keep `requirements-akida.txt` pinned to these.
- **Single machine:** the chip is in one server. Export data early so the group does not wait on it.
- **Rasters:** not promised. Only attempted if easy.
- **Public repo vs self-hosted runner:** the midpoint needs a public GitHub repo, but `docs/ci.md` advises against self-hosted runners on public repos. Before making it public, remove the `hw.yml` runner or restrict it (environment approval, no fork triggers), and keep the course PDFs and transcripts out. `Research/` holds those and is not tracked.
- **Phase I depth:** a table of scores is not enough. Each pattern needs an explanation (why a cluster is dense, whether it has sub-clusters, what distinguishes it). Plan the questions before plotting.

## Plan to the final

| By | Step |
|---|---|
| 05.11 | Hardware check: one image through the card, check what SDK stats and energy it reports. **Done.** |
| 05.11 | Confirm the Moodle entry exists and matches the description below. Update it, do not duplicate. |
| 05.11 | All four: agree the Phase I common direction (questions, data, starting view). Split into AI-aided and human-only pair. |
| 07.11 | Network chosen, quantized and converted, clean accuracy for both. **AkidaNet chosen and converted (05.11); clean accuracy still open.** |
| 09.11 | Fog/noise/low-contrast run exported (`accuracy.csv`, `energy.csv`). **Done on card and simulator (05.11).** Both pairs' views still need to switch to these files. |
| 10.11 | Pairs compare Phase I process and findings. Public repo ready with AI record and GenAI reflection. Video recorded. |
| 11.11 | Midpoint submission (repo, AI record, reflection, 5-min video). Pairs then switch roles. |
| 12.11 | Lab session 13:45-15:15 (S3 055 + online): Phase II practical examples and final submission/presentation details. Then all four agree the Phase II direction and which Phase I findings to follow up |
| 25.11 | Phase II data and views with at least two techniques per pair, claim tested |
| 02.12 | Freeze results. AI4VIS records and reflection summarized. Slides drafted. |
| 09.12 | Final submission: code from both pairs, AI records, 10-min video, slide deck |

## Description to hand in

**Task/model:** quantized AkidaNet (pretrained, ImageNet) vs. its SNN-converted counterpart on BrainChip Akida AKD1000 via MetaTF, on ImageNet-C

**Phase I:** investigate where CNN and SNN behavior diverges under fog/noise/low-contrast at 5 severities, using confusion-matrix and projection views of agreement and confidence, and measured on-chip energy per inference of the converted model

**Phase II:** explain the divergence with layer-wise activity/sparsity as spike-death proxy, combined with a second technique (for example saliency maps or k-nearest neighbors); full spike rasters only as stretch/limitation

**AI4VIS:** use ChatGPT/Claude while building the Phase I/II views; log prompts and accepted/rejected/refined decisions; report failure patterns and best practices

**References:** BrainChip MetaTF and Akida 1.0 model zoo (AkidaNet), ImageNet-C robustness benchmark

**Note:** group informed, only one entry, update existing entry if already registered

The Moodle entry still names MobileNetV2/ResNet-18 on CIFAR-10-C. Update the existing entry to the text above before 11.11; do not create a second one.

## Message for the group chat

> Hi all! Model change: MobileNetV2 does not convert for the AKD1000 (Akida 1.0), so we use BrainChip's pretrained quantized AkidaNet on ImageNet-C instead of CIFAR-10-C. No training needed. We compare the quantized Keras AkidaNet with its converted version on the card, under fog, noise and contrast at 5 severities. Phase I: find where the two models diverge (confusion matrices, projections, accuracy and confidence, on-chip energy per inference). Phase II: explain it with layer-wise activity as a proxy for "spike death". AI4VIS: we log prompts and decisions while ChatGPT/Claude help us build the views, and report failure patterns and best practices. **You need no hardware.** Card and simulator results are in `Phase 1 - No AI/results_card/` and `results_sim/` on the `yoshi-phase-1` branch. Please point your views at those files. I will update the Moodle entry.

## Sources and gaps

- Course PDF "2026 AI VIS Lab 02 Assignment Introduction" (attached to CEL-1192).
- BrainChip docs: https://doc.brainchipinc.com (Akida 1.0 model zoo and API reference).
- The chip is an AKD1000 PCIe card (PCI id 0x1e7c:0xbca1, from the lab rig config).
- Verified 05.11: AkidaNet converts for Akida 1.0 and runs on the card; the card reports energy per frame; tool versions (see Risks).
- Not verified: clean ImageNet accuracy on our setup, an energy method for the Keras baseline, and per-layer activity counts on hardware (`results_sim/layers.csv` currently holds layer metadata, not `nonzero_count` per image).
