# Fair Mediator

Fairness-aware group restaurant recommendation with LLM-parsed feedback. A group
of n diners with clashing tastes, diets and allergies gets one restaurant that
is safe for everyone and fair to everyone, plus a per-diner dish. Safety is a
hard pre-filter, fairness is the objective, and the language model only turns
a diner's free-text reaction into a label; deterministic code makes every
decision.

**Paper:** *Should the Mediator Be Fair? Fairness-Aware Group Restaurant
Recommendation with LLM-Parsed Feedback and a Fail-Closed Safety Filter*

The maths lives in `nbio.py`, and every notebook cell that runs it shows real,
executed output. A notebook holds no logic of its own: it imports from
`nbio.py` and shows what each step reads and writes.

## Run in 60 seconds

No API key needed for notebooks 01 through 04.

```bash
pip install -r requirements.txt
jupyter notebook 01-taste/01-ingredients-to-dish-vectors.ipynb
```

## What each notebook shows

| Notebook | Shows | Feeds |
|---|---|---|
| `01-taste/01` | taste vectors are derived from ingredients, never guessed | taste coverage |
| `01-taste/02` | every group member has a taste vector | — |
| `01-taste/03` | building the Uber Eats catalogs, their taste spread, the hand-check scores and the loader contract | Methods |
| `02-safety/01` | choosing only from the feasible set F, which no later stage can override; a stress test with heavily restricted diners | Methods |
| `03-aggregate/01` | the satisfaction matrix S the selectors reduce; whether taste similarity predicts real ratings; C1 under other satisfaction functions | `figdata_rev/` |
| `03-aggregate/02` | the four selectors are genuinely different functions | Table 9 |
| `03-aggregate/03` | whether and when lambda moves the pick; C1 on real restaurants; the propositions checked on real diners; why lambda = 5 is safe | Figure 6, `figdata_rev/` |
| `04-fairness/01` | mean alone hides an abandoned member; MinSat, Jain, Gini and MaxEnvy do not | Table 9 |
| `04-fairness/02` | per-diner dish compensation, and the experienced dish-level outcome | Table 14, `figdata_rev/` |
| `05-mediate/01` | the closed-loop mediator and the veto token, including non-convergence; mediation quality and the proxy-truth gap; the loop with a live AI diner (gpt-4o-mini writes, the parser reads) | Tables 11, 13, `figdata_rev/` |
| `05-mediate/02` | free text to reaction label, measured on 100 labelled cases | Table 15 |
| `05-mediate/03` | a whole conversation end to end, the round-cap check, and rounds to agreement on real diners | Table 18 |
| `06-evaluate/01` | the whole pipeline swept over synthetic groups, with the ceiling control; reproduces the headline numbers for the revision analyses | Tables 9-14 |
| `06-evaluate/02` | the paper's figures, from figdata CSVs | Figures 2-6 |
| `07-judge/01` | an LLM-judge **proxy** for perceived fairness (not a human study), including the Houston run and the live mediator's picks | Tables 16, 17 |

The analyses added in the journal revision (alternative utility models, the
order-aware utility, mediation under a utility the system does not know, the
scale of lambda) are the "Revision" sections at the end of `03-aggregate/01`,
`03-aggregate/03`, `04-fairness/02` and `05-mediate/01`; their outputs, including
the LaTeX table rows in `figdata_rev/tex/`, are in `figdata_rev/`. They run on the
per-group matrices in `data/shared/cache/revision_matrices.pkl`, which
`nbio.build_revision_cache()` builds on first use (about 10 minutes);
`06-evaluate/01` first checks that this numpy code reproduces the headline
numbers. No API calls are made: the LLM-parser replay uses the cached
predictions in `data/objections/parser_predictions.csv`, and the live-AI-diner arm
(E8, `figdata_rev/e8_live_llm.csv`, and the 5-group pilot in `e8_live_llm_pilot.csv`) replays its conversations from
`runs/mediator_llm.jsonl` unless `MEDIATOR_LIVE = True`.

## The one rule

A function is defined exactly once, in `nbio.py`. Every notebook imports it and
calls it; none of them redefines it.

## Data

See `data/README.md`. API keys go in a local `.env` file (see `.env.example`),
which is never committed.
