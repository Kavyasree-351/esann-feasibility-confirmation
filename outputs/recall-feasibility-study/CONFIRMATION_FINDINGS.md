# Fresh confirmation: unchanged diagnostic criterion

All pilot-selected tree settings and regularization parameters matched the lock. Original 600 policy cases and 2400 group rows were repeated on 2000 fresh, duplicate-excluded queries per dataset, across all five index seeds. Query roles never crossed; no previous official final TEST queries were opened.

| Dataset | Occupancy Brier mean ± SD | Geometry Brier mean ± SD | Relative gain mean ± SD | H1 seeds | False admissions / admissions |
|---|---:|---:|---:|---:|---:|
|glove|0.0578 ± 0.0024|0.0469 ± 0.0014|18.8% ± 1.9%|5/5|0/60|
|coco_clip|0.0949 ± 0.0034|0.0804 ± 0.0029|15.2% ± 2.6%|4/5|1/43|

Replicated H1 criterion: **True**.
Counts of admitted groups/caps across seeds are dependent and must not be interpreted as independent Bernoulli trials. A zero observed false-admission count does not prove a 10% population false-admission bound.

## Publication assessment
This confirmation strengthens only the scoped empirical diagnostic finding if the criterion passes. ExtraTrees, geometric features, CRC and LTT are established tools; none is claimed as a new statistical algorithm. Operational utility relative to the occupancy model is reported separately in confirmation_decisions.csv. The full ConANN engine remains unverified, and no conclusion about its published latency or native ndis measurements is supported. Publication readiness still requires a clear comparison with prior feasibility-diagnostic work and substantive engineering value beyond average prediction-score gains.
