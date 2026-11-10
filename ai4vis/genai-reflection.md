# Reflection on GenAI usage

Failure patterns observed:

- **It builds beside the contract instead of on it.** The first version invented its own schema,
  model names and folder layout. The shared `lab/` was not in the clone it was working from, and
  it built anyway instead of asking. That cost one full rejected version.
- **Works on stale context.** The second version was built on `main` and never looked at the
  other branches, so it missed the team's switch to AkidaNet / ImageNet-C. A teammate caught it
  in review. The AI does not know what the team decided outside the files it was shown.
- **Plausible API from memory.** Akida energy via `statistics.energy` (does not exist on the
  simulator), `cnn2snn.quantize` for MobileNetV2 (fails), "MobileNetV2 runs on the AKD1000"
  (it cannot: Akida 1.0 has no residual adds). The AI-assisted shared code had the same problem.
  Nothing about the toolchain held until it was run.
- **Demo data that looks like evidence.** Synthetic data with injected patterns produced
  convincing views that showed only what had been put in. Labels and images did not even match.
- **Carrying a design over to data where it no longer fits.** Class-level confusion matrices
  (fine for CIFAR-10) would be pure noise with 1000 classes and 2 images each. Softmax
  probability and raw Akida potential would have been plotted on one scale.
- **Views without uncertainty.** The first views showed point estimates only. With about 1000
  images per class per condition, many visible CNN−SNN differences are within noise.
- **Overclaiming status.** "Notebooks run" meant "run on synthetic data". A hypothesis's mechanism
  was written as if it were part of the claim.

Emerging best practices:

- Give the AI the shared contract (results layout, which branch is current) first, and make it
  read the code *and list the branches* before it generates anything.
- Every AI claim about an external API gets a run, not a doc check. Log the check and its result
  next to the claim (see the checks table in the record).
- No synthetic data in a view. If the views must exist before the data, run them on a small real
  run instead.
- Intervals on every rate and paired intervals on every CNN−SNN difference. Mark only what
  survives them.
- Separate what was tested from the suspected mechanism in every hypothesis.
- Review the AI's output against a checklist (schema, names, scales, uncertainty, provenance).
  That review is what caught the problems here, not the AI.

## Phase I comparison of the two pairs

### Shared question and different approaches

Both pairs asked how model behavior changes as inputs get harder to recognize and where the
quantized Keras AkidaNet and its converted AKD1000 counterpart give different answers. We used the
same card-run predictions for fog, Gaussian noise and contrast, each at five severities, with
2,000 images per condition and two images per class. Severity 1 was the mild-corruption reference.
The shared benchmark made it possible to compare what each analysis uncovered. [Card-run results](../Phase%201%20-%20No%20AI/results_card/)

The human-only pair used a hardware deviation matrix to trace individual label changes, then
used bootstrap intervals to test two specific hypotheses about groups of images. The AI-aided
pair explored a wider set of views, including prediction frequencies by condition, WordNet group
confusions, small multiples with uncertainty bands, and a projection of per-image prediction
outcomes across severities. The matrix made particular Keras-to-card label changes easy to
inspect. The condition-by-condition views made trends and reversals across severities easier to
see. [Human-only deviation matrix](../Phase%201%20-%20No%20AI/view_confusion.ipynb) ·
[AI-aided notebooks](../phase1/ai-aided/README.md)

### What each pair uncovered

The human-only pair found that dogs had a higher disagreement rate than devices and instruments
under its interval rule in only 3 of 15 conditions. Gaussian noise met the rule for fewer
cross-superclass disagreements on animal images in 0 of the 3 severities tested. The score
follow-up gave a more specific result. For dog images whose two leading Keras labels were both
breeds, the median top-two score gap was 0.212 on disagreements and 0.726 on agreements. The gap
was also smaller on device disagreements. This makes close scores a useful lead to investigate
in both groups. The AI-aided pair's broader group analysis also found little evidence that
fine-grained animal groups lost agreement faster, although it used different group definitions
and asked a different question. [Human-only hypothesis notebook](../Phase%201%20-%20No%20AI/view_hypotheses.ipynb) ·
[AI-aided group findings](../phase1/ai-aided/findings/03.md)

The AI-aided pair followed a reversal under Gaussian noise. Agreement rose from 65.9% at severity 3
to 72.8% at severity 5, even though both models were below 1% accuracy at severity 5. The label
frequencies showed that both models increasingly chose prayer rug, poncho and doormat. This
explains how they could agree more often as their predictions got worse. In the human-only
analysis, both models were wrong in 86.5% of disagreements across all conditions. Together these
results show why accuracy and agreement answer different questions. The score follow-up explains
where to look for closely ranked alternatives, and the frequency plots show how common wrong
labels can dominate a condition. [AI-aided divergence findings](../phase1/ai-aided/findings/01.md) ·
[Human-only hypothesis notebook](../Phase%201%20-%20No%20AI/view_hypotheses.ipynb)

### What AI changed and what verification cost

Claude helped the AI-aided pair explore visualization options and draft plotting code. The pair
discarded the first version because it used an invented data schema and synthetic examples with
patterns built into them. A teammate caught that a later version had missed the switch to
AkidaNet. The usable views came after rebuilding around the shared card CSVs and checking the
paired counts, labels and uncertainty. The human-only pair also spent time revising its approach.
It reduced an unwieldy 1,000-class matrix to a focused comparison and reran Keras to recover
the leading scores. The logs show different sources of rework, and they lack a common measure
of total time. The benefit we can point to is the breadth of the verified AI-aided views. The
clearest practice to carry forward is to give the tool the current model and data schema first,
then verify generated findings on the shared data before reporting them. [AI interaction record](ai-interaction-record.md) ·
[Human-only process log](../Phase%201%20-%20No%20AI/process-log.md)

### Confidence and Phase II

Direct counts support the shared-label pattern on this benchmark. The hypothesis verdicts
depend on the specified interval rules, and the small number of images per class limits
class-level conclusions. The top-two gaps are raw scores from a Keras rerun and establish
an association with disagreement. To test possible explanations in Phase II, we can compare
per-layer activity on matched examples where the models disagree and where both choose the
same wrong label, using the same corruption and severity within each comparison.

*Codex drafted this comparison from both pairs' notebooks and logs for review by both pairs.*
