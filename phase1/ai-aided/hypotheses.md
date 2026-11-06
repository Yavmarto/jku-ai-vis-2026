# Hypotheses (AkidaNet / ImageNet-C)

Setup: quantized Keras AkidaNet (CNN) vs the same weights converted and run on the AKD1000 (SNN),
2000 ImageNet-C images (2 per class), fog / Gaussian noise / contrast, severities 1–5
(Yoshi's `results_card/`). No clean condition and no confidence scores are exported.

**Honesty note:** these replace the CIFAR-10 hypotheses after the switch to AkidaNet. They were
written after a first look at the aggregate tables (accuracy, agreement, most-predicted class),
not blind, so treat H1 and H4 as descriptions rather than predictions (AI record row 4).
The **expectation** is what was tested. The **suspected mechanism** is a guess and was *not*
tested in Phase I.

| ID | Expectation (tested) | Suspected mechanism (guess, untested) | Tested in | Verdict |
|---|---|---|---|---|
| H1 | Accuracy of CNN and SNN stays within ±1 pp in every condition. | The conversion keeps the quantized weights exactly, so only rounding of potentials differs. | 01 accuracy/agreement | **Supported.** The largest gap is +1.2 pp [+0.3, +2.2] (contrast 3, SNN better). Every other interval includes 0 (01 §1) |
| H2 | Agreement falls with severity, fastest under contrast. | Low contrast leaves only small activation differences, and 4-bit outputs round them into ties. | 01 | **Partly supported.** Contrast falls monotonically (0.79 → 0.63). Fog flattens at ~0.71. Noise is U-shaped (0.74 → 0.66 → 0.73), because both models fall into the same sink (01 §2–3) |
| H3 | When CNN and SNN disagree, their answers are semantically closer than random classes, and drift apart with severity. | Disagreements come from near-ties between related classes. | 01 WordNet distance | **Supported.** Median distance 8–10 edges vs 14 for random pairs; same-group share 0.43–0.46 at severity 1 vs 0.10 by chance, falling to 0.21–0.27 at fog/contrast 5 (01 §4) |
| H4 | Under heavy corruption both models collapse onto the *same* few classes. | Corrupted images share low-level statistics (texture, colour) that one class "owns". | 01 sinks | **Supported.** Distinct predicted classes drop from ~770 to ~50 (noise 5). Both models' top class is the same in 13 of 15 conditions; the two exceptions are at severity 1, where no class takes more than 1.6 % (01 §3) |
| H5 | Fine-grained animal groups (dog, bird) lose agreement faster than object groups. | Fine-grained classes depend on texture that corruption removes first. | 02 group purity, 03 heatmaps | **Not supported.** No group stands out beyond its interval. Behaviour profiles carry almost no group information (purity ≈ chance except cat, n = 26) (02 §3, 03 §2) |
| H6 | Per image, CNN and SNN break at the same severity. | Same weights, so the same images are hard. | 02 breaking severity | **Mostly supported.** 58–66 % break at the same step. The rest is symmetric (sign tests p = 0.09–0.42) (02 §1) |
| H7 | SNN energy per inference drops as severity grows. | Sparser input gives fewer events, so less dynamic power and faster frames. | 03 energy | **Direction supported, effect tiny.** Spearman −0.9 within every corruption (n = 5 each), but the total spread is 1.4 % (83.9–85.1 mJ), mostly floor power, one measurement per condition (03 §3) |
