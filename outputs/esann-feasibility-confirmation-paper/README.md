# Updated ESANN diagnostic draft with held-out confirmation

**Forecasting Recall Feasibility Before Calibration: A Held-Out Diagnostic Study of Capped IVF Retrieval**

This is a separate update; the earlier exploratory paper was not overwritten. The research code/results and complete reproduction commands are in the sibling `recall-feasibility-study` folder. The `evidence` folder here contains the exact aggregate/source-lock/access records used in this draft. Figures and tables regenerate with that study's `build_paper_assets.py`.

Compile from this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# Or: tectonic --keep-logs main.tex
python -m pip install -r requirements-pdf.txt
python -B verify_paper.py main.pdf
```

Formatting checks: exactly six A4 pages,100-word abstract, original official esannV2 class,122x193mm centered printing area,44/44/52/52mm margins, embedded fonts. Author identified for single-blind review. Affiliation/postal address remain explicitly pending. Visual review supplements automated glyph-bound checks.

The paper reports a confirmed empirical forecast improvement, not a new calibration theorem or guaranteed admission rule. Shared query pools make the five index seeds dependent replications of index construction, not independent external studies. Source-level lookahead accounting remains mixed; the unmodified full ConANN engine was not reproduced. The primary-cap joint-risk policies did not issue, so no deployed-policy latency improvement is claimed.

Original official final TEST queries remain untouched. **The new, gated confirmation pool has been evaluated**, as recorded in evidence/DATA_ACCESS_LEDGER.json. This extension is local and has not been submitted or pushed to GitHub. Publication suitability still requires assessment of novelty/significance and external review.
