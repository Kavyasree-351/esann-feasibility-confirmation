# Additional closest-work check — 30 October 2026

This audit does not change frozen hypotheses or gates. It limits what a positive result could legitimately claim.

- [ConANN, PVLDB 2025](https://www.vldb.org/pvldb/vol19/p29-horchidan.pdf) and its [author code](https://github.com/soniahorchidan/conann): adaptive IVF recall calibration is established. Our author-score adaptation and scalar replay do not constitute its full system reproduction.
- [Efficiently Controlling Multiple Risks with Pareto Testing, ICLR 2023](https://arxiv.org/abs/2210.07913): existing multi-objective optimization followed by risk-controlling testing, including compute/quality trade-offs. Learning a path and calibrating it with LTT is not a new general methodology.
- [When Can Conformal Risk Control Certify LLM Outputs? Bounds, Impossibility, and Adaptation for Structured Generation, 2026 preprint](https://arxiv.org/abs/2606.29054): directly related feasibility/certification-boundary analysis in a different application. It already discusses feasibility, evidence strength and transfer checks. We have not independently validated all its theoretical or empirical assertions. A generic “check feasibility before calibration” narrative is therefore insufficient novelty.
- [Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees, 2026 preprint](https://arxiv.org/abs/2603.14623): learned input gates, risk calibration and feasibility conditions are existing work. A design-trained geometry gate is not automatically a novel risk-control method.

## Remaining possible contribution, conditional on evidence

A carefully reproduced ANN-specific empirical result about which geometry/evidence/work-accounting factors change an operational decision could still be informative. It requires cross-workload confirmation, strong matched baselines and a clear distinction from known risk-control tools. Native scalar-expression agreement establishes only implementation consistency. Returned-prefix versus executed-work distinction does not prove the authors misreported their runtime or native distance counters.

No exhaustive novelty claim, original general theorem, or publication guarantee is made.
