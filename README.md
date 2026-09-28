# Fair Mediator

Fairness-aware group restaurant recommendation with an LLM mediator. A group of
n diners with clashing tastes, diets and allergies gets ONE restaurant that is
safe for everyone and fair to everyone, plus a per-diner dish. Safety is a hard
pre-filter, fairness is the objective, and the LLM only reads and writes — it
never computes the decision.

**Paper (working title):** *Should the Mediator Be Fair? LLM-Mediated Group
Restaurant Recommendation under Inviolable Safety Constraints*

This repository is cookbook-style: the maths lives in `nbio.py` and every
notebook cell that runs it is real, executed output — nothing on screen is
invented or seeded. A notebook holds no logic of its own: it imports from
`nbio.py` and shows what each step reads and writes.

## Run in 60 seconds

No API key needed for notebooks 01 through 04.

```bash
pip install -r requirements.txt
jupyter notebook 01-taste/01-ingredients-to-dish-vectors.ipynb
```

## What each notebook proves

| Notebook | Proves | Feeds |
|---|---|---|
| `01-taste/01` | taste vectors are derived from ingredients, never guessed | taste coverage % (Limitations) |
| `01-taste/02` | every group member has a real taste vector (F0) | — |
| `02-safety/01` | choosing only from the feasible set F makes a safety violation structurally impossible | Methods |
| `03-aggregate/01` | the satisfaction matrix S the selectors reduce | — |
| `03-aggregate/02` | the four C1 selectors are genuinely different functions | Table 1 |
| `03-aggregate/03` | whether/when lambda moves the pick, on real and constructed data | Figure 1 |
| `04-fairness/01` | mean alone hides an abandoned member; MinSat/Jain/Gini/MaxEnvy do not | Table 1, Figure 2 |
| `04-fairness/02` | per-diner dish compensation and its scale-mismatch caveat | Table 5 (H4) |
| `05-mediate/01` | the closed-loop mediator (C2) and veto token (C5) are real code, including a genuine non-convergence finding | Table 2, Table 4 |
| `05-mediate/02` | free text -> reaction label is a measured component (100 labelled cases), and cannot fail into consent | Table 6 |
| `05-mediate/03` | a whole conversation end to end, and proof the round cap hides nothing | Table 9 |
| `06-evaluate/01` | the whole pipeline swept over synthetic groups, with the ceiling control that makes the mediation result interpretable | Figures 1-2, Tables 1-5 |
| `07-judge/01` | the selector ranking survives an independent three-family LLM-judge **proxy** (not a human study) | Table 7, Table 8 |

**Still open** (tracked in `prds/`): wiring the mediator into potluck's live
chat graph (`backend/ai/graphs/team_suggestion_graph.py`, tracker MEDIATE-2).
Every claim the paper makes is now backed by a notebook that runs.

## The one rule

A function is defined exactly once, in `nbio.py`. Every notebook imports it and
calls it; none of them redefine it. Attribution for ported code lives next to
the function it names, and again in `CONTRIBUTORS.md`.

## What is not here

No API key, no secret, no personal data. `01-taste/02` harvests the taste
*vectors* Vihaan generated in `usertasteembeddings.py` — the original file also
hardcoded a Pinecone API key, which is not reproduced anywhere in this
repository. See `AnacodicAI-Labs/docs/work-files/ai-in-food/research_fair-mediator/prds/PRD_0`.
