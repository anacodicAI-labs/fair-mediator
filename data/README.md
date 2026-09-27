# data/ — inputs the code reads

Only files a notebook or `nbio.py` reads live here. Everything the code
*writes* goes to `figdata/` (tables and figures) or `runs/` (per-call logs).

| File | Read by | Source |
|---|---|---|
| `Menu/*.json` | `01-taste/*`, `nbio.load_catalog()` | 5 restaurant menus, Hyderabad. Vasu, potluck commit `481b5f5` (2026-07-07) |
| `restaurants.csv` | `01-taste/02`, `05-mediate/03` | 100 restaurants with Google Places fields (`place_id`, `maps_url`, rating). Vasu, potluck `481b5f5`, file `data/restaurants - restaurants.csv` |
| `ingredient-flavor.csv` | `01-taste/*`, `nbio.load_catalog()` | 1,000-ingredient hand-built taste table (no bitter column). Vasu, potluck `481b5f5` |
| `personas.json` | `01-taste/02`, `02-safety`, `03-aggregate`, harness | 95 **synthetic** diners. Taste vectors from Vihaan's `usertasteembeddings.py` (potluck `f045279`); persona fixture from Vincent (potluck `3c32c76`, `86124d4`) |
| `objections/labelled_reactions.csv` | `05-mediate/02` | hand-labelled objection set (PRD-10) |
| `objections/parser_predictions.csv` | `05-mediate/02`, `05-mediate/03` | cached parser output, so the notebook replays with no API key |
| `judge/human_claims.csv` | `07-judge/01` | claims checked against the LLM-judge proxy (PRD-9) |

## Not in this repo

- **FlavorGraph** (https://github.com/lamypark/FlavorGraph, `input/nodes_191120.csv`,
  `input/edges_191120.csv`). Nothing reads it yet. It returns with the code that
  derives the bitter dimension from it.
- **Food.com** (https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).
  Download from Kaggle. Not redistributed here.
