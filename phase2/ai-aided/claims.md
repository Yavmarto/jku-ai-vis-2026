# Phase II claims (AI-aided pair: Yoshi, Gabriel)

Phase I described *where* the quantized Keras AkidaNet (CNN) and its Akida conversion (SNN)
diverge. Phase II tries to explain it. Each claim below tests a **suspected mechanism** that
Phase I wrote down but did not test (`phase1/ai-aided/hypotheses.md`).

**Honesty note:** C1–C3 were fixed after a short prototype over the per-condition AUROCs (AI
record, Phase II row 3). The mechanisms themselves come from Phase I and were written before any
Phase II data. Verdicts use the decision rules written here, not ones chosen after the run.

Techniques (Moodle: at least two per pair, compared or combined):

1. **kNN in embedding space** (this pair's choice, 26.11): 512-d embeddings from
   `results/akidanet_imagenetc.parquet`, cosine, k = 15, neighbours searched within one
   corruption x severity.
2. **Confidence of the top answer** (`cnn_conf`, softmax) as a decision-level signal, combined
   with technique 1 in C3.
3. **Layer-wise sparsity** (the shared Phase II technique): **blocked**, the per-image layer
   export has not been run. See `TODO-yoshi.md`. Claim L1 is stated now so it cannot drift.

| ID | Claim | From Phase I | Test and decision rule | Verdict |
|---|---|---|---|---|
| C1 | Disagreements are near-ties: they happen where the CNN is unsure, and the SNN's answer is a class the CNN's own neighbourhood already votes for. | H3 mechanism | (a) AUROC of −`cnn_conf` for disagreement > 0.5 with the 95 % interval above 0.5 in every condition. (b) `snn_support` on disagreeing images > frequency-matched `rand_support`, paired interval above 0, in every condition. | see `findings/01.md` |
| C2 | Corruption collapses the embedding, and sinks are dense regions of it. | H4 mechanism | (a) Embedding spread falls monotonically with severity in each corruption. (b) Within each condition that has a sink, sink-predicted images have a smaller mean kNN distance than the rest (Mann-Whitney p < 0.01). | see `findings/01.md` |
| C3 | The two signals trade places: at low severity, disagreement is a decision-level near-tie (confidence explains it best); at high severity it is a neighbourhood effect (consensus explains it best). Combining them beats either. | new | (a) AUROC(−consensus) − AUROC(−conf) rises with severity in each corruption (Spearman > 0) and its paired interval is above 0 in at least one severity-5 condition. (b) Cross-validated AUROC of a logistic model on both is ≥ the better single one in every condition. | see `findings/01.md` |
| L1 | Activity dies toward the last layers as severity grows, most under contrast; within a condition, disagreeing images have lower last-layer activity than agreeing ones. | H2 and H7 mechanisms | Per-layer non-zero share from the simulator export. **Not run.** | open |
