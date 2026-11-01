# Revision analyses (response to reviewers)

These scripts produce every number added in the revision; outputs go to `figdata_rev/`.

1. Build the per-group matrices from the cached catalogs and diner pool of the
   `fair-mediator` repository (about 10 minutes):
   `python build_cache.py /path/to/fair-mediator cache.pkl`
2. Set `FM_REV_CACHE=cache.pkl`, `FM_REV_OUT=../figdata_rev`, `FM_REPO=/path/to/fair-mediator`.
3. `python check_baseline.py` reproduces the original headline numbers exactly
   (C1 gains and shares, 180/200 mediator, 187/190 veto).
4. Run `e1_utilities.py` (Sec. 5.4, Table 5, Fig. 7), `e2b.py` (Table 6),
   `e3_mediation.py` (Tables 7-8, dominance check), `e5b_repropose.py`,
   `e6_lambda.py`, `e6b.py`, `e6c.py` (Sec. 4.6, Table 4), `e7_multicriteria.py`
   (multi-criteria rows), then `gen_tables.py` to regenerate table rows and the
   frontier CSVs used by Figure 7.

`lib.py` re-implements the selectors and mediation loop in numpy; `check_baseline.py`
verifies it against the paper's harness output before anything new is run.
The LLM-parser replay uses `data/objections/parser_predictions.csv` (cached
gpt-oss-20b outputs); no API calls are made.

Compile a clean (unhighlighted) manuscript with:
`pdflatex "\def\cleanversion{}\input{main}"` (jobname e.g. `main_clean`).
