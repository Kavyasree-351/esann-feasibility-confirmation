# Prospective diagnostic extension — frozen before new workload outcomes

Date: 30 October 2026. This protocol supersedes no previous superiority gate. The earlier exploratory diagnostic paper remains unchanged until new evidence is available. Predictions are hypotheses, not promised novelty.

## Questions and decision rules

**H1: Design-only diagnosis.** Given the same 500 design labels already required for policy learning, can centroid geometry improve predictions of the *maximum-cap-prefix loss* beyond a strong cap-and-occupancy-only predictor? This is not label-free or a distribution-free certificate. Targets are loss, `1(loss > .20)`, `1(loss > .50)`. Primary objective: mean squared probability error averaged over the two severity events and all fixed cap fractions. Primary comparator: matched-tuned occupancy predictor, not a constant strawman. Pass: >=10% reduction in this score in >=4/5 seeds on BOTH datasets; when used to admit a workload/group regime (all three predicted risks under their targets), false-admission fraction among empirically infeasible admitted regimes <=.10, with every count reported. A run with zero admissions has no demonstrated diagnostic utility and does not pass. The finite pilot workload is the target of this exploratory prediction check, not a population impossibility claim.

**H2: Executed versus returned work.** Author ConANN search scans a lookahead list, then may restore the previous result. Audit whether counting only the returned prefix undercounts executed distance work. Pass: exact native replay/portable differential controls, positive overhead and >=5% mean distance undercount in >=4/5 seeds on BOTH datasets at the primary 25% cap. Under-count is `(executed - returned)/executed`; costs include centroid routing. This is an implementation-scope audit, not a statistical theorem failure. A native kernel replay is not a full author Faiss-engine reproduction. Full-engine validation is required before a strong published ConANN claim.

If neither H1 nor H2 passes, stop before confirmation: report a negative extension, preserve data blindness, and do not improve the paper's claims. If H1 passes, freeze source/model/config hashes before fresh confirmation. If only H2 passes, confirmation additionally requires full-engine validation; otherwise report it as provisional. No parameter/gate changes after inspecting pilot outcomes. No dataset is omitted for an inconvenient result.

## Data and separation

- GloVe-25 public normalized embeddings: fresh random TRAIN-bank queries, excluding a common corpus-candidate bank. Never read any of the previous reserved official TEST query IDs. Corpus size 10000, five random corpus/index seeds 0..4.
- COCO text-to-image CLIP ViT-B/16 512-dimensional embeddings: official author release `fabiocarrara/str-encoders`, tag v0.1.3, `coco-t2i-512-angular.hdf5` (142254080 bytes). Uses random ANN-benchmarks split: image corpus, caption queries. Normalize both; report cross-modal source law explicitly. This is a new workload, not a claim that CLIP represents every current encoder.
- Split query roles globally and identically across index seeds using a fixed permutation seed 73001: design 500, calibration 2000, pilot 500, reserved confirmation 2000. Remaining public queries stay unused. GloVe uses these role sizes from its independent TRAIN query bank. Corpus sampling RNG is 81000+seed. Splits are saved and hashed BEFORE reading query vector values or labels. Query IDs never change role across seeds.
- Calibration-size ablation uses prefixes 100, 300, 1000, 2000 of the same calibration pool. No cherry-picking counts or outcomes. Pilot queries and the new narrative remain exploratory; only gated, frozen confirmation may support an independent confirmation claim.
- Budget fractions `.05,.10,.25,.50,1.00`, total cap `64 + floor(b*N)`. k=10, M=64 lists. Whole-list prefixes, stable distance/ID ties, float64 exact truth. Conditional groups: all plus ambiguity upper 20%, 10%, 5%, thresholds learned on design only.

## Methods and matched conditions

- Uniform LTT: fixed probe counts 0..64 with cap clipping.
- Cell-shift LTT: ten design ambiguity cells, CRC-derived design templates plus common offsets.
- ConANN-score LTT adaptation: author score family using prefix k-th squared distance, rank regularization and normalization, tune regularization on design only, then apply the SAME LTT risk procedure. Label this an adaptation, not an author-engine reproduction. Account for lookahead scans; compare the uncapped author CRC reference separately with its weaker marginal objective.
- Author-score marginal CRC reference: author score construction and regularization grid `{0,.001,.01,.1}`, with discrete conservative CRC selection rather than GSL approximate root solving. Record that numerical/implementation deviation explicitly; do not call it the actual author engine.
- Original full-engine build attempt and native extracted early-stop kernel audit are logged separately. Keep upstream hash/license and all local build failures.
- Every LTT method: conditional mean FNR<=.10, Pr(FNR>.20)<=.10, Pr(FNR>.50)<=.05; 4 overlapping groups, 12 risks; fixed-sequence maximum-p test at .025 and reverse witness at .025/12. No simultaneous all-caps/all-method guarantee. Zero-prefix queries have loss1. Unknown is not infeasible.
- Two diagnostic models use identical ExtraTrees hyperparameter grids: depth3/6 x leaf10/30, 64 trees, fixed RNG. Design split first300 train/last200 validation, split by query BEFORE expanding caps. Features for occupancy control: cap fraction, feasible prefix fraction, used corpus-work fraction, next-list fraction; geometry extension adds centroid ambiguity, nearest distance scaled by design median, centroid distance spread, visited/unvisited radius ratios. No exact-neighbor feature at query time. Retrain chosen setting on all design queries only. All tuning outcomes saved.

## Reports

Five seeds retained; mean/SD; paired query-bootstrap intervals (2000 replicates) for prediction-score differences; all status counts, effective group sizes, cap maxima, prefix floors, false admissions, lookahead overhead. Query resamples match models and preserve within-query cap dependence. Across seeds, datasets/query pools are shared; do not claim independent external studies.

Time actual index construction, exact-label generation, diagnostic learning, calibration selection and bounded retrieval separately. Record peak process RSS. Measure retrieval latency on a fixed, prespecified pilot subset, 3 repeats in a balanced method order, same threads, without exact-label computations inside timed regions. These measurements characterize this loaded Windows host only; no hard latency guarantee or GPU claim.

## Stop/confirmation discipline

Smoke controls precede pilots. Pilot is 2 datasets x 5 seeds with the full factorial cap/calibration-size/group setup. Do not read confirmation rows while diagnosing/tuning. A confirmation command must require a stored passing gate and matching source/protocol/config hashes. Failure of that check blocks evaluation by design. Any necessary bug correction is logged and invalidates/repeats affected pilots, not hidden.
