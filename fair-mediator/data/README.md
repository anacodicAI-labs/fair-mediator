# data/

Committed here: small files we made ourselves (`ingredient-flavor.csv`, `bitter-ingredients.csv`,
`personas.json`, `objections/`, `judge/`, `flavorgraph/`) and, for now, the five Potluck menus in `Menu/`.

**No Kaggle data is committed.** Kaggle-derived files go in two gitignored folders:

| Folder | Holds |
|---|---|
| `data/external/` | raw downloads, exactly as published |
| `data/external_cache/` | catalogs built from them by `scripts/build_ubereats_catalog.py` |

## 1. Food.com recipe flavours (needed by `nbio.recipe_taste`)

`recipes_with_flavour_profiles.csv`: 231,636 Food.com recipes with the columns `id, name, ingredients, flavor_profile`.
The `id` is the Food.com recipe id, so it joins to `RAW_recipes.csv` and `RAW_interactions.csv` from
[shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

Copy it from the potluck repo:

```bash
cp ../potluck/data/recipes_with_flavour_profiles.csv data/external/
```

To keep it somewhere else, set `FM_FLAVOUR_CSV=/path/to/file.csv`.

`flavor_profile` has five tastes (sweet, salty, sour, umami, spicy). For each recipe, each taste is the mean of
`ingredient-flavor.csv` over the ingredients whose exact name is in that table. That was checked on 300
recipes, with a mean absolute error of 0.0012. Bitter is not in the file. `nbio` adds it by the same rule
from `bitter-ingredients.csv`, a hand-made list in which a weight of 0 marks an exclusion such as
"white chocolate" or "root beer".

## 2. Uber Eats restaurants and menus (needed by `load_catalog_ubereats`)

Source: [ahmedshahriarsakib/uber-eats-usa-restaurants-menus](https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus).
Download it from the Kaggle page (or with `kaggle datasets download -d ahmedshahriarsakib/uber-eats-usa-restaurants-menus`)
and unzip it so that these two files exist:

```
data/external/ubereats/restaurants.csv
data/external/ubereats/restaurant-menus.csv
```

To keep them somewhere else, set `FM_UBEREATS_DIR=/path/to/folder`.

Then build a city's catalog:

```bash
python scripts/build_ubereats_catalog.py --list-cities           # choose a city
python scripts/build_ubereats_catalog.py --city "Houston"        # writes data/external_cache/
```

Point the harness at it without editing any notebook:

```bash
FM_DATASET=ubereats FM_CITY="Houston" jupyter nbconvert --execute ...
```

## Licences

Check each Kaggle page's licence before use. The terms are recorded in `journals.txt` (Rashan).
`Menu/` and `restaurants.csv` contain Google Maps data and must be dropped or replaced before the repo goes public.
