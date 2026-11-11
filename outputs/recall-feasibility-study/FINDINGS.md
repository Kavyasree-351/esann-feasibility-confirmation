# Prospective diagnostic extension: findings

All ten dataset/index-seed pilots retained: GloVe-25 and COCO CLIP text-to-image retrieval. 600 cap/calibration-size/method conditions, 2400 group rows, 200 floor rows. Query pools are shared across seeds; these counts are not independent external replications.

| Dataset | Seed | Occupancy Brier | Geometry Brier | Relative gain | False admissions | H1 pass | Work undercount |
|---|---:|---:|---:|---:|---:|---|---:|
|glove|0|0.0608|0.0501|17.6%|0/12|True|11.3%|
|glove|1|0.0643|0.0510|20.6%|0/12|True|0.9%|
|glove|2|0.0600|0.0498|17.0%|0/12|True|1.6%|
|glove|3|0.0591|0.0485|17.9%|0/12|True|11.4%|
|glove|4|0.0637|0.0526|17.4%|0/12|True|11.8%|
|coco_clip|0|0.0940|0.0852|9.4%|0/9|False|6.4%|
|coco_clip|1|0.0961|0.0838|12.7%|0/8|True|6.0%|
|coco_clip|2|0.0926|0.0794|14.3%|0/9|True|6.9%|
|coco_clip|3|0.0926|0.0792|14.5%|0/9|True|7.0%|
|coco_clip|4|0.1034|0.0837|19.0%|0/8|True|5.5%|

Geometry diagnostic gate: **True**. Native replay/lookahead pilot gate: **False**. Full author engine verified: **False**.
Fresh confirmation permitted: **True**. No confirmation rows or previously reserved final queries have been evaluated.

## Interpretation
- The geometry model uses design labels and is a predictor, not a label-free or distribution-free infeasibility certificate.
- Counting returned prefixes can omit lookahead work in this source-level replay. The unmodified full ConANN/Faiss engine could not be built locally because BLAS/MKL was missing. Its published performance claims have not been reproduced or refuted.
- Author-score adapters differ from the original GSL root solver and full engine. Marginal CRC references do not certify simultaneous tail/group risk; LTT adapters share the stronger objective.
- The actual author scanner can truncate lists with max_codes; whole-list prefix results do not cover arbitrary partial-list operation.
- Do not promote this pilot into independent confirmation, a universal calibration impossibility, or a proven new method.

## Next action
If the gates are closed, keep confirmation sealed and report the negative extension. Do not change thresholds or keep searching the same pilot for a success. If H1 passes, freeze the current hashes and use the gated confirmation command. If only the kernel-scope H2 passes, validate the full author engine on a supported Linux/BLAS environment first.
