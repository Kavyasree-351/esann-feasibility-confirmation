# Recall-feasibility diagnostic: frozen pilots and held-out confirmation

This is a prospective diagnostic extension, not a claim of a novel conformal algorithm. It preserves the previous projects and paper. Read **CONFIRMATION_FINDINGS.md**, **PROTOCOL.md**, **SOURCE_AUDIT.md** and **LITERATURE_AUDIT.md** together.

## What actually ran

- Two workloads: fresh GloVe-25 TRAIN-bank queries and public **COCO CLIP ViT-B/16 text-to-image retrieval**. Corpus10000, IVF64 lists, k10, seeds0..4.
- Frozen design/calibration/pilot/confirmation roles:500/2000/500/2000 queries. Previously inspected GloVe TRAIN query IDs were excluded. No previous official reserved TEST query was loaded.
- Six correctness controls, including actual list scans versus reference traces;10,000 native differential checks of the author's scalar stop expressions.
- All ten pilot runs:600 policy cases and2400 group rows. H1 passed5/5 GloVe and4/5 COCO index seeds. H2 failed across workloads, and remains a negative/mixed finding.
- Pilot settings/output hashes were locked. Before any confirmation labels, six exact repeated COCO input vectors were replaced with unused query rows under a fixed input-only rule. No model/gate was changed. The original pilot's three repeated evaluation inputs are disclosed with a score sensitivity.
- All ten confirmation runs, each with2000 fresh duplicate-excluded queries: another600 policy cases and2400 group rows. H1 again passed5/5 and4/5. Mean severity Brier improvements:18.8% and15.2%; these are means over seeds, with SDs in the tables.
- COCO decision counts: occupancy10 false admissions out of52 admissions versus geometry1/43. GloVe: both0/60, so better scores changed no decisions. The caps/groups/seeds are dependent; these fractions are not population guarantees or100 independent trials.

**Current data-access state is DATA_ACCESS_LEDGER.json.** The split/lock files are immutable pre-label snapshots; their zero-touch fields describe the time they were frozen. Fresh confirmation has now been evaluated. Previous official final-query data remain untouched;4994 official COCO query rows remain entirely unused by this extension.

## Reproduce from the supplied workspace-shaped archive (Bash)

The archive keeps `outputs/recall-feasibility-study` and `outputs/esann-feasibility-confirmation-paper` as sibling folders. Run from its root. Python3.12 was used.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r outputs/recall-feasibility-study/requirements.txt
mkdir -p work/adaptive-retrieval work/diagnostic-data
curl -L --fail 'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/0bc1ed6/glove-25-angular.hdf5' \
  -o work/adaptive-retrieval/glove-25-angular.hdf5
curl -L --fail 'https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-t2i-512-angular.hdf5' \
  -o work/diagnostic-data/coco-t2i-512-angular.hdf5
python -B outputs/recall-feasibility-study/smoke.py
python -B outputs/recall-feasibility-study/study.py pilot --resume
python -B outputs/recall-feasibility-study/report.py
python -B outputs/recall-feasibility-study/validate_results.py --stage pilot
python -B outputs/recall-feasibility-study/confirmation.py run
python -B outputs/recall-feasibility-study/confirmation_report.py
python -B outputs/recall-feasibility-study/validate_results.py --stage confirmation
python -B outputs/recall-feasibility-study/build_paper_assets.py
```

Dataset SHA256 values are pinned in splits.json and checked before use. The archive already includes all saved outcomes, split roles, source locks, native-audit metadata and the confirmation driver, so `--resume`/confirmation `run` skip completed runs. To regenerate compute rather than inspect the existing evidence, preserve the original results elsewhere and use a separate extraction/workspace; do not mix fresh reruns into the frozen lock or overwrite the original evidence. Models are reselected only from the exact same DESIGN queries and fixed grid; confirmation_report checks the resulting selected settings against the original lock.

For a complete COMPUTATIONAL rerun after the downloads above, this command creates a new sibling folder, reruns every pilot and confirmation seed, verifies the selected design settings, refreshes timing-dependent hashes only in that new folder, and checks Brier-score agreement with the recorded results:

```bash
python -B outputs/recall-feasibility-study/computational_reproduction.py
```

It reuses already-known query roles and therefore is **not** new independent scientific confirmation. The original frozen evidence remains unchanged. A failed agreement check is retained for investigation rather than forced to pass.

`prepare.py splits` and `prepare.py freeze` intentionally refuse to overwrite existing ledgers. `confirmation.py prepare` intentionally refuses to overwrite its input-only split ledger. They are provenance tools for creating a NEW study, not necessary commands to rerun this archived one. Regenerating query splits after seeing outcomes invalidates confirmation.

## Inspect results without downloading data

```bash
python -B outputs/recall-feasibility-study/report.py
python -B outputs/recall-feasibility-study/confirmation_report.py
python -B outputs/recall-feasibility-study/build_paper_assets.py
```

These commands use saved outcomes. The independent validators rebuild index/group assignments and therefore require public datasets. They do not open additional query roles. `pilot_sensitivity.py` excludes exact repeats only for a descriptive pilot score check; it never changes the original model or gate.

## Paper compilation

The updated six-page paper is in `outputs/esann-feasibility-confirmation-paper`. It uses the original official esannV2 class, corrected legacy offsets, A4,100-word abstract and an identified author (affiliation/address pending).

```bash
cd outputs/esann-feasibility-confirmation-paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# Alternatively: tectonic --keep-logs main.tex
python -m pip install -r requirements-pdf.txt
python -B verify_paper.py main.pdf
```

## Baseline and efficiency limits

The actual full ConANN/Faiss build initially reached a missing BLAS/MKL dependency. A later attempt supplied checksum-verified, workspace-local OpenBLAS/LAPACK/GSL libraries and configured successfully, then failed in unmodified upstream code at `posix_memalign` on Windows/MinGW. The successful native audit covers exact extracted scalar expressions ONLY. The portable author-score path differs from GSL threshold solving and is always labeled an adaptation. Instructions for a supported full-engine build are in SOURCE_AUDIT.md. Native ConANN `ndis` already counts executed scans: our replay does not refute its published latency.

All LTT methods share groups, risks, calibration labels and confidence settings. Marginal CRC references have a different objective and are not promoted as equivalent safety baselines. At the primary25% cap no joint-risk path issued, so the planned deployment-latency table is empty rather than populated with uncertified policies. Timing and peak RSS are recorded in per-run JSON; a large wall-clock outlier remains intact. Do not infer hard hardware deadlines or label-cost savings from these records.

ExtraTrees, the features, LTT and CRC are established tools. The confirmed contribution candidate is a scoped empirical diagnostic finding, with limitations stated in the paper; novelty/acceptance remain unestablished. No new paper or code has been pushed to GitHub or submitted. Upstream licenses are included for the extracted expressions; no new repository-wide license is granted.
