# Forecasting Recall Feasibility Before Calibration

A held-out study of approximate nearest-neighbor (ANN) search under a hard limit on distance computations. We ask whether the geometry of a query and its index can help predict **which workloads are likely to meet a recall target before running formal risk calibration**.

[Read the six-page paper](outputs/esann-feasibility-confirmation-paper/main.pdf) &middot; [Reproduce the study](outputs/recall-feasibility-study/README.md) &middot; [See the confirmation results](outputs/recall-feasibility-study/CONFIRMATION_FINDINGS.md)

## Why this matters

An inverted-file (IVF) index groups vectors into lists. At query time it orders the lists by distance to their centroids and scans the closest ones first. Scanning more lists can recover more true neighbors, but each additional list consumes part of the search budget. Two queries with the same budget and similarly sized lists can still miss different numbers of true neighbors because those neighbors can land in different lists.

When calibration fails to certify a policy, the distinction matters. The search budget may be too small for the tested policy class, or the available calibration queries may provide too little evidence. The study separates **certified**, **evidence of infeasibility**, and **unknown**. For policies that scan whole lists in a fixed order, the longest prefix that fits the cap gives the best attainable recall for that query. This provides a reference for testing whether a smaller prefix could possibly meet the target.

## How we tested it

- **Workloads.** GloVe-25 word embeddings and COCO CLIP text-to-image retrieval, with 10,000 corpus vectors, 64 IVF lists, and five corpus/index seeds per workload. We computed recall@10 against the exact top ten embedding neighbors.
- **Search and risk.** We tested caps of 5%, 10%, 25%, 50%, and 100% of corpus distance work, with centroid routing counted separately in the cap. We measured mean false-negative rate (FNR) and the frequencies of queries missing more than 20% or 50% of their true neighbors. We evaluated all queries and three overlapping groups comprising the most ambiguous 20%, 10%, and 5% of queries, defined using design data.
- **Matched prediction models.** The occupancy model used the cap, the fraction of lists that fit, work spent, and the size of the next list. The geometry model added centroid distances, ambiguity, and list radii. Both used ExtraTrees with the same design labels and hyperparameter search budget. Exact neighbors provided training targets. Prediction features came from the query and index.
- **Held-out confirmation.** Each workload had separate design, calibration, pilot, and confirmation roles of 500, 2,000, 500, and 2,000 queries. We locked the pilot-selected settings before labeling the confirmation queries, screened confirmation inputs for exact duplicates before labeling, and kept all five seeds. Both phases retained 600 policy cases and 2,400 group-level rows across the two workloads.

The primary score was Brier error, the mean squared error of the predicted probabilities for the two severe-miss events, averaged across queries and caps. We used Learn Then Test to check three risk targets for each group under matched calibration conditions. The target was at most 10% mean FNR, at most a 10% chance of FNR above 20%, and at most a 5% chance of FNR above 50%. [Frozen protocol](outputs/recall-feasibility-study/PROTOCOL.md)

## What we achieved

**Geometry improved forecasts of severe recall loss on both held-out workloads.** The table reports the mean Brier error and standard deviation across five index seeds. The percentage improvement is the mean relative reduction across seeds.

| Workload | Occupancy Brier error | Geometry Brier error | Relative reduction | Seeds meeting the prespecified criterion |
| --- | ---: | ---: | ---: | ---: |
| GloVe-25 | 0.0578 &plusmn; 0.0024 | 0.0469 &plusmn; 0.0014 | 18.8% | 5/5 |
| COCO CLIP | 0.0949 &plusmn; 0.0034 | 0.0804 &plusmn; 0.0029 | 15.2% | 4/5 |

A passing seed required at least a 10% Brier error reduction, at least one admission, and at most 10% false admissions among its admitted cases. One COCO seed exceeded the false-admission limit and remains in the results.

**The decision benefit depended on the workload.** We called a cap and query group *admitted* when its predicted risks met all three targets. A false admission occurred when the observed risks exceeded at least one target. On COCO CLIP, false admissions fell from **10 of 52** admitted cases with occupancy features to **1 of 43** with geometry, with zero false refusals for both models. On GloVe, both models made the same decisions, with zero false admissions among 60 admissions each. This shows where better forecasts changed the decision and where the simpler model already sufficed. The cases share query pools, groups, and caps, so the counts describe the observed workloads. [Per-seed scores](outputs/esann-feasibility-confirmation-paper/generated/figure_forecasts.png) &middot; [Decision counts](outputs/recall-feasibility-study/confirmation_decisions.csv)

The contribution is an empirical diagnosis. Query geometry adds useful information beyond how much of the index fits under a cap, and that information can change which workloads look suitable for calibration. A formal certificate still depends on the calibration evidence and the assumptions of the risk test.

## What the experiments left open

- At the primary 25% cap, none of the tested joint-risk search policies received a certificate. The forecast comparison therefore supports a decision about which settings to investigate, with deployment latency and labeling savings still unmeasured.
- The source-level ConANN lookahead audit missed its prespecified cross-workload criterion. The native distance counter already includes executed scans. The Windows build reached a platform error in the full engine, so the verified code audit covers the scalar stopping logic. [Source audit](outputs/recall-feasibility-study/SOURCE_AUDIT.md)
- The evidence covers two workloads with 10,000-vector indexes and recall against exact embedding neighbors. Transfer to larger indexes, shifted query populations, or downstream answer quality remains open. [Publication assessment](outputs/recall-feasibility-study/PAPER_READINESS.md)

## Explore or reproduce the study

- [Saved results and rerun instructions](outputs/recall-feasibility-study/README.md) cover dataset downloads, smoke tests, validation, figure generation, and full experiment reruns. The saved results can be inspected before downloading either dataset.
- [Study code](outputs/recall-feasibility-study/) includes the IVF engine, risk tests, split and confirmation checks, and pilot and confirmation traces.
- [Paper source](outputs/esann-feasibility-confirmation-paper/main.tex) and [compilation instructions](outputs/esann-feasibility-confirmation-paper/README.md) are included alongside the PDF, tables, and figures.
- [Package manifest](PACKAGE_MANIFEST.json) records checksums for the archived artifacts. The accompanying research notes preserve the status recorded when the archive was assembled.
