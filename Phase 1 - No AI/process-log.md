# Human-only pair: process log

Fill in as you work, not afterwards. This is the evidence for the pair comparison.
Light AI use that is allowed (code completion, docs lookup): note it here too.

| Date | Who | Step | Time spent | Stuck on / dead ends | Decision and why |
| 03.11.2026 | Gabriel | Git init and debug | 1h | Cloned git and had to debug git (clicked some buttons accidentally that destroyed my local changes). | Reset git to HEAD because it was too tedious. |
| 03.11.2026 | Gabriel | Corruptions | 1h | Stuck on values to take for corruptions and what corruptions to take. | Research for CIFAR-10-C, found a paper and recreated the values |
| 03.11.2026 | Gabriel | Pipeline scaffolding & placeholder model | ~2.2h | Teammate is training MobileNetV2 on 2080 Ti; unable to evaluate final weights immediately. | Built placeholder loader in `Phase 1 - No AI/mobilenetv2_cifar10_utils.py` and decoupled inference into `view_eval_pipeline.ipynb` to unblock matrix and UMAP views. |
| 03.11.2026 | Gabriel | View Development | 1.5h | Needed standardized Parquet outputs to bridge pipeline to notebooks. | Implemented `view_confusion.ipynb` and `view_projections.ipynb` directly consuming Parquet results. |
| 04.11.2026 | Gabriel | View Development & unlearned model tryout | 3h | Making plots work the way intended. Tedious model shape mismatch resolvements. | Built placeholder pipeline, Debggung and retry.  |
