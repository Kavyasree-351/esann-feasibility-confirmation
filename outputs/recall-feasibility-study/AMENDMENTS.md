# Attempt and amendment ledger

All entries below precede the first new pilot result.

1. Inspected and pinned author ConANN revision `aa76fbc2e2aaec822ddca525baf4916da46a653b`. First build configuration lacked make/Ninja. Installed Ninja into workspace-local build tooling, retried with the available GCC toolchain, and reached a substantive missing BLAS/MKL dependency. Retained `conann-build-attempt.log`. No author-engine result is claimed.
2. Compiled the exact scalar score/early-stop expressions from the pinned author source in an isolated harness. Passed 10,000 differential cases against the portable expression and a rollback/lookahead positive control. This validates that expression, not the complete scanner or author's experiments.
3. Selected the author-released COCO CLIP text-to-image HDF5 benchmark after checking provenance and download size. Rejected poorly documented AG News embeddings as the primary new workload, and rejected the 3080-query-only Banking77 embedding export as too small for the intended disjoint roles. No retrieval outcomes from either rejected dataset were computed.
4. Before reading new query vectors, corrected the proposed GloVe split selection to exclude the union of **24,771 previously inspected TRAIN calibration query IDs** from the earlier cap-tail study. Archived the initial metadata-only split ledger under workspace scratch. No new query values or labels had been read. New roles were then saved and hashed. Old official reserved TEST queries are never loaded.
5. Six synthetic smoke controls passed, including direct exact-reference agreement, monotone prefix floors, empty-group rejection, the 72-sample threshold, live search versus trace recall, and **executed** distance counters. These controls precede the pilot and use no public evaluation queries.

After freezing, changes to source, protocol, split IDs or native evidence require an explicit amendment and affected pilot reruns. No silent mutation of gates or evaluation roles is permitted.
