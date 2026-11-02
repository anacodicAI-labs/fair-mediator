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
| `02-safety/01` | choosing only from the feasible set F, which no later stage can override | Methods |
| `03-aggregate/01` | the satisfaction matrix S the selectors reduce | — |
| `03-aggregate/02` | the four selectors are genuinely different functions | Table 9 |
| `03-aggregate/03` | whether and when lambda moves the pick | Figure 6 |
| `04-fairness/01` | mean alone hides an abandoned member; MinSat, Jain, Gini and MaxEnvy do not | Table 9 |
| `04-fairness/02` | per-diner dish compensation | Table 14 |
| `05-mediate/01` | the closed-loop mediator and the veto token, including non-convergence | Tables 11, 13 |
| `05-mediate/02` | free text to reaction label, measured on 100 labelled cases | Table 15 |
| `05-mediate/03` | a whole conversation end to end, and the round-cap check | Table 18 |
| `06-evaluate/01` | the whole pipeline swept over synthetic groups, with the ceiling control | Tables 9-14 |
| `07-judge/01` | an LLM-judge **proxy** for perceived fairness (not a human study) | Tables 16, 17 |

`revision_code/` holds the analyses added in the journal revision (alternative
utility models, the order-aware utility, mediation under a utility the system
does not know, the scale of lambda); their outputs are in `figdata_rev/`. See
`revision_code/README_REVISION.md`.

## The one rule

A function is defined exactly once, in `nbio.py`. Every notebook imports it and
calls it; none of them redefines it.

## Data

See `data/README.md`. API keys go in a local `.env` file (see `.env.example`),
which is never committed.
