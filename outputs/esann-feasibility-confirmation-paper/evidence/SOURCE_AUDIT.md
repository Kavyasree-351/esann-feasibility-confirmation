# ConANN author-code audit (not a performance reproduction)

Upstream: https://github.com/soniahorchidan/conann

Revision: `aa76fbc2e2aaec822ddca525baf4916da46a653b`

Relevant file: `conann/faiss/IndexIVF.cpp` in that revision.

- `calibrate` fixes kreg=1 and tunes rank regularization over `{0,.001,.01,.1}`. `optimization` uses GSL root finding for a marginal FNR target. It is not a joint group/tail LTT procedure.
- `search_preassigned_with_error_quantification` scans a list, computes the current kth distance and the regularized score, and on an exceeded threshold copies `prev_idxi`/`prev_simi` back before breaking. The already executed distance work is still added to `ndis`. Consequently an offline count of returned clusters need not equal the number of clusters actually visited.
- This source-level fact does **not** imply the authors' measured latency is wrong; their actual `ndis` counter already includes scans. The experiment asks whether using returned-prefix counts as a downstream budget proxy would understate work.
- The scanner also uses `list_size_max`/`max_codes` and can truncate a list. Whole-list prefix impossibility statements do not apply to arbitrary partial-list configurations of the author engine.
- `NATIVE_AUDIT.json` records the full upstream source hash and the extracted scalar-expression hash. `author_stop_excerpt.cpp` contains the exact expressions with a minimal ABI wrapper; license notices are retained in `CONANN_LICENSE.txt` and `FAISS_LICENSE.txt`.
- The portable baseline is explicitly an **author-score adaptation**: normalized-vector squared-distance upper bound4, discrete conservative CRC threshold selection instead of GSL, deterministic prefix tie handling, locally implemented IVF, and optional matched LTT group/tail selection. It must never be described as a full author-code replication or as ConANN's published result.

## Supported-engine reproduction still needed

**Later isolated-dependency attempt:** obtained checksum-verified OpenBLAS0.3.34 and GSL2.8 MSYS2 packages under workspace scratch, without modifying system installations. CMake then found BLAS/LAPACK/GSL and configured successfully. The actual unmodified-source compilation failed in `faiss/utils/AlignedTable.h:50` because `posix_memalign` is unavailable on the Windows/MinGW toolchain. Logs: `conann-isolated-dependency-attempt.log` and `conann-native-build.log`. The original frozen NATIVE_AUDIT.json records the earlier pre-pilot state; this later attempt does not change the frozen method or upgrade the full-engine reproduction status.

On a Linux environment with GCC, CMake, OpenMP, BLAS/LAPACK and GSL, obtain the pinned upstream source and follow its documented commands:

```bash
git clone https://github.com/soniahorchidan/conann.git
cd conann
git checkout aa76fbc2e2aaec822ddca525baf4916da46a653b
cd conann
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF
cmake --build build --target error latency -j 1
```

Then use the upstream data conversion/execution utilities with the SAME data/split/seed metadata. A successful build alone is not experimental equivalence; compare live returned IDs, ndis, score thresholds and timing with the replay before promoting an implementation-specific claim.
