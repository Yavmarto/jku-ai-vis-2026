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

Comparison with the other pair (insights found or missed, design choices, breadth/depth, speed vs verification, confidence):
