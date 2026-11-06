# Hypotheses

Written before looking at results (AI record #2). The **expectation** is what we test. The
**suspected mechanism** is a guess. Unless the Verdict column says otherwise, it was *not* tested
in Phase I (black box). Phase II (layer activity) is where mechanisms can be checked.

Verdicts are for the verification run described in `README.md` (float CNN trained here,
`snn-sim`, regenerated CIFAR-10-C). Re-check them on the official data and the final models.

| ID | Expectation (tested) | Suspected mechanism (guess, untested) | Tested in | Verdict |
|---|---|---|---|---|
| H1 | Corruption mainly increases confusion within similar animal pairs (cat↔dog, deer↔horse), not random errors. | Corruptions remove fine texture first, and texture is what separates these pairs. | 01 pair plot, sink view | **Refuted.** cat→dog and deer→horse *fall* with severity. Errors go to sink classes instead (01 §2, §4) |
| H2 | automobile↔truck and airplane↔ship grow with severity. | Fog and low contrast wash out the sky/sea background that separates airplane from ship. | 01 | **Partly supported.** airplane→ship grows under fog and contrast (≈0.5 at contrast 5), automobile↔truck only under noise, ship→airplane not at all (01 §5). Background mechanism untested |
| H3 | Low contrast hurts the SNN more than the CNN. | 4-bit activations lose small differences when the input range shrinks, so more activations round to zero. | 01 Δacc, 03 gap plot | **Weakly supported.** Gap −6.3 pp at contrast 4 vs −4.3 clean, but −3.4 at 5. The clear SNN-specific effect is errors moving into deer (03 §3). 4-bit mechanism untested |
| H4 | Gaussian noise hurts both models about equally. | Quantization already acts like noise, so the SNN is not specially fragile to added noise. | 01, 03 | **Refuted.** SNN degrades *less*: the gap closes from −4.3 pp to 0.0 [−0.7, +0.7] at noise 5 (03 §4) |
| H5 | Disagreements sit at class borders in CNN feature space and grow with drift from the clean features. | Near a border, small quantization errors are enough to flip the decision. | 02 purity and drift tests | **Supported.** Disagreement 28.4 % at purity ≤ 0.4 vs 0.5 % at purity 1, and rises with drift within a severity (02 §2–3). Mechanism untested |
| H6 | When wrong, the SNN's confidence separates right from wrong no better than the CNN's (AUROC not higher). | Conversion keeps the ranking of the outputs but loses resolution. | 03 AUROC | **Holds on clean, fog, contrast; fails under strong noise** (SNN AUROC higher at noise 5: 0.687 vs 0.653) (03 §6) |
| H7 | SNN energy per inference drops as contrast/fog severity grows. | Sparser input gives fewer events. | 03 energy | not testable: no energy data (simulator, `energy.csv` not produced) |
