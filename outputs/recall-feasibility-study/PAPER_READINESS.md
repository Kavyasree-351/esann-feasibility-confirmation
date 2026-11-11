# Publication-readiness assessment after fresh confirmation

## What changed since the original diagnostic draft

We now have a **confirmed empirical finding**, not just a reinterpretation of a failed candidate: a matched geometry-augmented predictor reduces severity Brier error by18.8% on GloVe and15.2% on COCO CLIP, with the original combined criterion passing5/5 and4/5 index seeds on both pilot and fresh confirmation. The models and thresholds were frozen before the latter outcomes. Confirmation vectors were deduplicated against every earlier role before labeling.

On COCO, geometry reduces observed false admissions from10/52 to1/43 without adding false refusals among the evaluated regimes. On GloVe, it changes no decisions because the occupancy baseline already classifies the regimes correctly. These cases share groups, caps and query pools; the counts are descriptive dependent cases, not independent Bernoulli trials or a population safety guarantee.

This is materially stronger evidence for an ESANN diagnostic submission candidate than the previous paper. It does **not** establish automatic publishability, an A* algorithmic contribution or guaranteed acceptance.

## Claims the evidence supports

1. Under the declared prefix-IVF setup and workload distributions, the geometry feature bundle improves prediction of maximum-prefix severity losses over a matched cap/occupancy learner on separate held-out queries.
2. Prediction improvement can change conservative workload-admission decisions, but this benefit is workload-dependent and does not imply a formal risk guarantee.
3. More calibration observations can distinguish unknown from infeasible rather than always produce an issued policy; capacity and statistical evidence remain different failure mechanisms.
4. Source-level returned-prefix counts can omit executed lookahead work in a replay. Material overhead did not replicate across the prespecified pilot gate, and the author's native ndis counter already includes scans. No published ConANN performance claim was refuted.

## Remaining important limits

- ExtraTrees, geometry features, CRC, LTT and feasibility analysis already exist. The paper's candidate contribution is the scoped empirical diagnosis and confirmation, not a newly invented generic method. An expert assessment against closest work is still needed to judge whether this is sufficiently substantive for ESANN.
- Two public sources, a10000-vector corpus and five index realizations do not establish broad robustness across modern encoders, large production indexes or OOD query laws. The query pools are shared across seeds.
- The full unmodified ConANN/Faiss baseline is not reproduced. Isolated BLAS/LAPACK/GSL packages resolved the initial dependency gap, but actual compilation then failed at the unsupported POSIX allocation call on Windows/MinGW. This limits direct system claims; the main forecast comparison is against the matched occupancy learner, not a claimed ConANN defeat.
- No calibrated joint-risk path issued at the primary25% cap. Do not claim improved deployed recall/latency or measured calibration-label savings from the diagnostic forecasts. Missing deployment-latency measurements remain missing.
- Zero or few false admissions in dependent finite workloads do not certify a10% population bound. Additional operating thresholds must not be tuned on the already-used confirmation data.
- Component controls run after confirmation use only the OLD PILOT and are labeled exploratory; they are not independently confirmed mechanistic evidence.
- Author affiliation/postal address remain pending. The paper has not been submitted or pushed to GitHub.

## Recommended submission posture

Present this as a narrowly scoped, reproducible diagnostic study with a held-out empirical result and preserved failures. Obtain a critical external read focused on significance and closest-work overlap, and complete the author metadata. For stronger ANN-systems claims, first reproduce the full engine on a supported Linux/BLAS environment and add larger independent workloads under a new frozen protocol. Do not claim the existing confirmation pool is fresh again.
