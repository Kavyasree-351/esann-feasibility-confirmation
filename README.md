# Forecasting Recall Feasibility Before Calibration

A held-out diagnostic study of capped inverted-file (IVF) vector retrieval. The study tests whether query and index geometry improves predictions of severe recall loss beyond search budget and partition occupancy.

| Workload | Reduction in severity prediction error (Brier score) | Index seeds meeting the prespecified criterion |
| --- | ---: | ---: |
| GloVe | 18.8% | 5/5 |
| COCO CLIP | 15.2% | 4/5 |

On COCO CLIP, observed false admissions fell from 10 of 52 admissions to 1 of 43. GloVe admission decisions were identical. These dependent workload cases provide descriptive evidence. Formal recall certification remains a separate step.

- [Read the six-page paper](outputs/esann-feasibility-confirmation-paper/main.pdf) or view the [LaTeX source](outputs/esann-feasibility-confirmation-paper/main.tex).
- [Reproduce the study](outputs/recall-feasibility-study/README.md) using the documented dataset downloads, experiment commands, and validation scripts.
- [Review the confirmation findings](outputs/recall-feasibility-study/CONFIRMATION_FINDINGS.md), [frozen protocol](outputs/recall-feasibility-study/PROTOCOL.md), and [publication limitations](outputs/recall-feasibility-study/PAPER_READINESS.md).

The checked-in package includes the pilot and confirmation traces, negative findings, figure and table assets, and a [file checksum manifest](PACKAGE_MANIFEST.json). Raw datasets are available from the sources linked in the reproduction guide. The package notes describe the state at the time the research archive was assembled.
