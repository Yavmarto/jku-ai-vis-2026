# AI and Visualization Lab: Akida project proposal

Issue: CEL-1192 · Updated 2026-11-03 · Replaces `akida-ai-vis-proposal-v2.md` (deleted). `akida-ai-vis-course-feasibility.md` stays for reference.

## The idea in one paragraph

We compare a **quantized CNN baseline** with its **SNN-converted counterpart** running on a real BrainChip AKD1000, using clean images plus three corruptions: fog, noise and low contrast. Phase I investigates where and for which inputs the two models diverge, using accuracy degradation, confidence and on-chip energy per inference for both models. Phase II explains that divergence by looking inside the converted model, using layer-wise activity/sparsity as a proxy for "spike death" under corruption. AI4VIS logs how ChatGPT/Claude help us build these visualizations and reports the failure patterns and best practices we find.

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
| 11.11 | Midpoint deadline | Next. The submission format and the two-pair setup were announced in the 29.10 session. That material is not in our files, so confirm it on Moodle or in the recording. |
| 09.12 | Final presentations | |

Final presentation (09.12): attendance is compulsory for the final live presentation session (Lecture 01). The presentation is 5 minutes, everyone submits a video, and a random subset presents live and is graded on it. All four members should be present.

## Task and models

- **Task/model:** quantized MobileNetV2 or ResNet-18 vs its SNN counterpart on the BrainChip AKD1000 via MetaTF. Pick one network by the first conversion check (below).
- **Baseline:** the quantized CNN.
- **Converted model:** the same network converted with MetaTF (`quantizeml` / `cnn2snn` / `akida`) and mapped to the chip.
- **Data:** corrupted-image robustness benchmark: CIFAR-10-C as the primary set (small, realistic for the chip), an ImageNet-C subset only if the network and time allow. Both benchmarks include fog, Gaussian noise and contrast corruptions with 5 severity levels.
- **Fallback:** if conversion of the chosen network fails, use a pretrained model from the Akida 1.0 zoo (for example AkidaNet) with the same plan.

## Phase I: accuracy degradation and energy

1. Evaluate both models on clean test images. Record top-1 accuracy.
2. Evaluate both models on fog, noise and low-contrast at severities 1 to 5.
3. Report accuracy per corruption and severity, and the drop relative to clean accuracy.
4. Report measured on-chip energy per inference for both models, from the SDK's power/energy output, on clean and on corrupted inputs.

Outputs: `accuracy.csv` (`model, corruption, severity, n_images, top1`) and `energy.csv` (`model, corruption, severity, energy_per_inference, source`).

## Phase II: layer-wise activity as a spike-death proxy

- For the converted model, record per layer the share of non-zero activations (sparsity) for each corruption and severity, for the same images used in Phase I.
- Visualize activity per layer against severity. The question is whether activity fades toward the last layers as corruption grows ("spike death"), or whether it behaves differently per corruption.
- State a testable claim up front, for example "low contrast makes activity die out before the last layers", and report it even if it turns out false.
- **Spike rasters:** attempt only if MetaTF/Akida exposes per-timestep or per-event data without much effort. Otherwise list it as a stretch goal and limitation.

Wording note: the AKD1000 computes only on non-zero activations, but the values are small integers, not single-bit spikes. Phase II measures "sparse activity", and we call it a proxy for spike death in the write-up.

Output: `layers.csv` (`image_id, corruption, severity, layer_index, layer_name, nonzero_count, total_units`).

## AI4VIS: LLM support for visualization design and analysis

The course asks us to record how generative AI supports and influences the design and analysis process, across Phase I, Phase II and the presentation.

- Keep a prompt log from day one: important prompts, tool (ChatGPT or Claude), result, accepted/rejected/refined, and why.
- Use the LLMs for sketching views, writing plotting code, annotating results and transforming data, including the Phase I and Phase II views above.
- Report failure patterns (wrong aggregation, misleading color scales, invented API calls, charts that hide the pattern) and the best practices we arrive at for using AI effectively.
- Stretch: ask the LLMs to propose and critique visualizations specifically for event-based, per-layer sparsity data, and report qualitatively how well that works. Layer activity is unusual data, so there is little for the model to lean on.

## Visualization

Phase I views are investigative tools with the black box closed. The course marks a chart that only shows a performance metric as the wrong approach, so accuracy numbers go into the views below and are not plotted alone. Plain Python notebooks (Plotly/Altair or matplotlib) reading the three CSV files. Suggested views:

- Confusion matrices per corruption and severity for both models, plus a difference matrix. Reorder classes by error similarity to reveal blocks.
- Projection (UMAP or t-SNE) of the test images or model embeddings, colored by whether CNN and SNN agree and by confidence, with drill-down to the images. Report the parameters (perplexity or neighbors) and do not read cluster sizes as meaningful (Lecture 03).
- Small multiples by corruption and severity, so patterns can be compared across cases.
- Energy per inference shown as an attribute of these views, not as a separate metric chart.
- Hypotheses from the training data and class similarity (for example cat/dog or car/truck), tested against the confusion matrices.
- Phase II: heatmap of layer index x severity, colored by non-zero share, one per corruption, linked to the Phase I divergences it should explain.

Group decision on the exact stack. The `akida` package has a software simulator, so members without the card can regenerate small samples. Label simulator and hardware numbers separately.

## Risks and checks

- **Hardware check first:** run one image through a small model on the card. Confirm it maps to the device, returns a prediction, and prints SDK statistics. I have not run the card for this project, so nothing here is verified yet.
- **Energy is the biggest unknown.** Whether the AKD1000 SDK reports usable power/energy is not confirmed. If it does not, report energy as unavailable or estimated and say so. Do not present estimates as measurements.
- **Open question: how does the baseline get an energy number?** The converted model runs on the chip. The quantized CNN baseline may only run on a CPU/GPU or the simulator, so "on-chip energy for both" may not be possible. Resolve in the hardware check. If not possible, report the baseline's energy with a different method, labeled as such, and do not compare the two as if measured alike.
- **"SNN-converted":** MetaTF converts a quantized CNN into an event-based Akida model with low-bit integer activations. It is not a classic spiking network. Use the phrase as the course framing, but describe precisely what the chip does.
- **Environment:** the Akida tools need specific Python and TensorFlow versions. Pin them in one environment file early.
- **Single machine:** the chip is in one server. Export data early so the group does not wait on it.
- **Rasters:** not promised. Only attempted if easy.

## Plan to the final

| By | Step |
|---|---|
| 05.11 | Hardware check: one image through the card, check what SDK stats and energy it reports |
| 05.11 | Confirm the Moodle entry exists and matches the description below. Update it, do not duplicate. |
| 07.11 | Network chosen, quantized and converted, clean accuracy for both |
| 09.11 | Fog/noise/low-contrast run exported (`accuracy.csv`, `energy.csv`) |
| 11.11 | Midpoint: Phase I prototype plots |
| 25.11 | Phase II layer-activity data and views, claim tested |
| 02.12 | Freeze results. AI4VIS log summarized. Slides drafted. |
| 09.12 | Final presentation |

## Description to hand in

**Task/model:** quantized MobileNetV2/ResNet-18 vs. SNN counterpart on BrainChip Akida AKD1000 via MetaTF, on CIFAR-10-C

**Phase I:** investigate where CNN and SNN behavior diverges under fog/noise/low-contrast at 5 severities, using confusion-matrix and projection views of agreement and confidence, and measured on-chip energy per inference

**Phase II:** explain the divergence with layer-wise activity/sparsity as spike-death proxy; full spike rasters only as stretch/limitation

**AI4VIS:** use ChatGPT/Claude while building the Phase I/II views; log prompts and accepted/rejected/refined decisions; report failure patterns and best practices

**References:** BrainChip MetaTF, ImageNet-C/CIFAR-C robustness benchmark

**Note:** group informed, only one entry, update existing entry if already registered

Check the "measured on-chip energy" wording after the hardware check. If the SDK reports no usable energy numbers, or the baseline cannot be measured the same way, change the Phase I line before the midpoint.

## Message for the group chat

> Hi all! Update on our Moodle entry: we compare a quantized CNN baseline with its SNN-converted version on the BrainChip AKD1000, on clean images plus fog, noise and low-contrast corruptions (CIFAR-C style). Phase I: find where the two models diverge (confusion matrices, projections, accuracy and confidence, on-chip energy per inference). Phase II: explain it with layer-wise activity as a proxy for "spike death". AI4VIS: we log prompts and decisions while ChatGPT/Claude help us build the views, and report failure patterns and best practices. **You need no hardware.** I run the images on the chip and share CSV files; the `akida` package also has a simulator. Please check the entry on Moodle matches, and tell me if you want a different network or dataset before 11.11.

## Sources and gaps

- Course PDF "2026 AI VIS Lab 02 Assignment Introduction" (attached to CEL-1192).
- BrainChip docs: https://doc.brainchipinc.com (Akida 1.0 model zoo and API reference).
- The chip is an AKD1000 PCIe card (PCI id 0x1e7c:0xbca1, from the lab rig config).
- Not verified: that the card returns power/energy data, how the baseline energy is measured, exact tool and Python versions for the chosen network, per-layer counts on hardware, and whether MobileNetV2/ResNet-18 convert cleanly for Akida 1.0.
