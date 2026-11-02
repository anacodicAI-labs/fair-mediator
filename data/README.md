# data/ — inputs the code reads

Only files a notebook or `nbio.py` reads live here. Everything the code
*writes* goes to `figdata/` (tables and figures) or `runs/` (per-call logs).

| File | Read by | Contents |
|---|---|---|
| `ingredient-flavor.csv` | `01-taste/*`, `nbio.load_catalog()` | 1,000-ingredient hand-built taste table |
| `bitter-ingredients.csv` | `nbio.recipe_taste()` | hand-made bitter list (weight 0 = exclusion), spot-checked in `validation/README.md` |
| `objections/labelled_reactions.csv` | `05-mediate/02` | 100 hand-written, hand-labelled diner reactions |
| `objections/parser_predictions.csv` | `05-mediate/02`, `05-mediate/03` | cached parser output, so the notebook replays with no API key |
| `judge/human_claims.csv` | `07-judge/01` | published human-study findings checked against the LLM-judge proxy |
| `shared/` | see below | third-party and derived data |

## data/shared/

All data that comes from outside sources, or is built from them, is kept in
this one folder:

| Folder | Holds |
|---|---|
| `shared/external/` | Kaggle downloads and the Food.com recipe flavour table |
| `shared/cache/` | the Uber Eats catalogs and the Food.com diner pool built from them |
| `shared/small_catalog/` | the five-restaurant catalog (`menus/`, `restaurants.csv`) and its 95 synthetic diners (`personas.json`) |

## 1. Food.com recipe flavours (needed by `nbio.recipe_taste`)

`shared/external/recipes_with_flavour_profiles.csv`: 231,636 Food.com recipes
with the columns `id, name, ingredients, flavor_profile`. The `id` is the
Food.com recipe id, so it joins to `RAW_recipes.csv` and `RAW_interactions.csv`
from [shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).
To keep it somewhere else, set `FM_FLAVOUR_CSV=/path/to/file.csv`.

`flavor_profile` has five tastes (sweet, salty, sour, umami, spicy). For each
recipe, each taste is the mean of `ingredient-flavor.csv` over the ingredients
whose exact name is in that table (checked on 300 recipes, mean absolute error
0.0012). Bitter is not in the file; `nbio` adds it by the same rule from
`bitter-ingredients.csv`.

## 2. Uber Eats restaurants and menus (needed by `load_catalog_ubereats`)

Source: [ahmedshahriarsakib/uber-eats-usa-restaurants-menus](https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus).
Download it and unzip it so that these two files exist:

```
data/shared/external/ubereats/restaurants.csv
data/shared/external/ubereats/restaurant-menus.csv
```

To keep them somewhere else, set `FM_UBEREATS_DIR=/path/to/folder`. Then build
a city's catalog (about 1 minute each):

```bash
python scripts/build_ubereats_catalog.py --list-cities
python scripts/build_ubereats_catalog.py --city "Houston, TX"
python scripts/build_ubereats_catalog.py --city "Seattle, WA"
```

Point the harness at a city without editing any notebook. `FM_FIGDATA` keeps
each run's CSVs apart:

```bash
FM_DATASET=ubereats          FM_CITY="Houston, TX" FM_FIGDATA=figdata/ubereats-houston-tx \
  jupyter nbconvert --to notebook --execute 06-evaluate/01-run-harness.ipynb --output-dir /tmp
FM_DATASET=ubereats-shuffled FM_CITY="Houston, TX" FM_FIGDATA=figdata/ubereats-shuffled-houston-tx ...
python scripts/c1_report.py
```

`ubereats-shuffled` is the random-menu control: the same restaurants and menu
sizes, with every dish dealt to a random restaurant (`FM_SHUFFLE_SEED`,
default 7).

## 3. Food.com users as real diners (needed by `load_personas_foodcom`)

Download `RAW_interactions.csv` from the same Food.com Kaggle page into
`data/shared/external/foodcom/`. `nbio.build_personas_foodcom()` keeps every
user with at least 10 star ratings (12,035 users) and sets their taste to the
mean `recipe_taste()` of the recipes they rated 4 or 5 stars. Only user IDs,
recipe IDs and star ratings are read. The pool is cached in
`data/shared/cache/personas_foodcom.json` on first use.

**Taste is real; diet and allergies are synthetic.** Food.com does not record
them, so they are drawn with a fixed seed at approximate US adult rates
(`SYNTHETIC_DIET_RATES`, `SYNTHETIC_ALLERGY_RATES` in `nbio.py`), and every
diner carries `"diet_allergy_source": "synthetic"`.

```bash
FM_PERSONAS=foodcom FM_DATASET=ubereats FM_CITY="Houston, TX" ...
```

## Licences

The Kaggle datasets are used for non-commercial academic research under the
terms stated on their pages.
