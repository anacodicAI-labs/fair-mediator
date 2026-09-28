"""Build a real-restaurant catalog for one city: Uber Eats menus, each dish
matched to its closest Food.com recipe, dish taste = nbio.recipe_taste(recipe).

    python scripts/build_ubereats_catalog.py --list-cities
    python scripts/build_ubereats_catalog.py --city "Houston" [--n 300 --seed 7]

Writes data/external_cache/catalog_ubereats_<city>.json (gitignored — derived
from Kaggle data) and validation/match_check_<city>.csv, 100 seeded dishes for
hand-checking the matcher. nbio.load_catalog_ubereats() reads the catalog.

Matching is TF-IDF: word 1-2-grams plus character 3-5-grams, fitted on the
Food.com recipe names, cosine top-1. The dish name is matched first; the
description is used only when the name alone scores below the threshold.
Below the threshold a dish stays on the menu but unmatched: taste all zeros,
ingredients and allergens None (unknown, so a safety filter fails closed).

Original to this paper (Vincent, wangvincentx-cmd).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402

UBEREATS_DIR = Path(os.environ.get("FM_UBEREATS_DIR") or nbio.EXTERNAL / "ubereats")
VALIDATION = ROOT / "validation"

# Restaurants that do not sell meals. Matched against the comma-separated
# Uber Eats category tags, case-insensitive.
NON_FOOD_TAGS = {"convenience", "grocery", "alcohol", "liquor", "wine & spirits", "pharmacy",
                 "flowers", "pet supplies", "retail", "everyday essentials", "specialty foods"}
DRINK_TAGS = {"coffee and tea", "coffee & tea", "bubble tea", "juice and smoothies",
              "smoothies", "juice", "coffee", "tea"}

# Menu items that are not dishes. A section name that matches drops every item
# in it; an item name that matches the second pattern drops just that item.
NON_DISH_SECTION = re.compile(
    r"\b(drinks?|beverages?|sodas?|bottled|juices?|smoothies?|shakes?|coffee|teas?|cocktails?|"
    r"beers?|wines?|alcohol|liquor|spirits|sauces?|dressings?|dips?|condiments?|extras?|"
    r"add[- ]?ons?|utensils?|napkins?|merch(andise)?|gift cards?)\b", re.I)
NON_DISH_ITEM = re.compile(
    r"^(side of |extra |add |no )|"
    r"\b(coke|coca[- ]cola|pepsi|sprite|dr\.? pepper|fanta|mountain dew|7[- ]?up|gatorade|"
    r"red bull|bottled water|water|soda|lemonade|iced tea|sweet tea|milkshake|smoothie|latte|"
    r"cappuccino|americano|espresso|utensils?|silverware|napkins?|cutlery|straws?)\b", re.I)
SAUCE_ONLY = re.compile(r"^\W*(\w+\W+){0,2}(sauce|dressing|dip|ketchup|ranch|mayo|salsa)\W*$", re.I)

MIN_DISHES = 10
MATCH_THRESHOLD = 0.30
HANDCHECK_N = 100


def parse_city(address: str):
    """'5215 Kirby Dr, Houston, TX, 77098' -> 'Houston, TX'. The state is the
    two-letter part; the city is the part before it."""
    parts = [p.strip() for p in str(address or "").split(",")]
    for i in range(1, len(parts)):
        if re.fullmatch(r"[A-Z]{2}", parts[i]):
            return f"{parts[i - 1]}, {parts[i]}"
    return None


def load_restaurants() -> pd.DataFrame:
    path = UBEREATS_DIR / "restaurants.csv"
    if not path.exists():
        sys.exit(f"{path} not found — download the Uber Eats dataset first (data/README.md).")
    df = pd.read_csv(path)
    df["city"] = df["full_address"].map(parse_city)
    return df


def load_menus(restaurant_ids) -> pd.DataFrame:
    path = UBEREATS_DIR / "restaurant-menus.csv"
    ids = set(restaurant_ids)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunks.append(chunk[chunk["restaurant_id"].isin(ids)])
    return pd.concat(chunks, ignore_index=True)


def is_food_restaurant(category) -> bool:
    tags = {t.strip().lower() for t in str(category or "").split(",") if t.strip()}
    if tags & NON_FOOD_TAGS:
        return False
    return not (tags and tags <= DRINK_TAGS)


def is_dish(section, name) -> bool:
    name = str(name or "").strip()
    if not name:
        return False
    if NON_DISH_SECTION.search(str(section or "")):
        return False
    return not (NON_DISH_ITEM.search(name) or SAUCE_ONLY.match(name))


def list_cities(min_restaurants: int = 200) -> None:
    rest = load_restaurants()
    rest = rest[rest["category"].map(is_food_restaurant)]
    counts = rest.groupby("city").agg(restaurants=("id", "size"),
                                      unique_names=("name", "nunique"))
    counts = counts[counts["unique_names"] >= min_restaurants].sort_values("unique_names", ascending=False)
    print(f"cities with >= {min_restaurants} distinct food restaurants: {len(counts)}\n")
    for city, row in counts.head(40).iterrows():
        tags = (rest.loc[rest["city"] == city, "category"].fillna("").str.split(",").explode()
                .str.strip().str.lower())
        tags = tags[(tags != "") & ~tags.isin(["american", "fast food", "burgers", "sandwiches"])]
        top = ", ".join(f"{t} {n}" for t, n in tags.value_counts().head(5).items())
        print(f"{city:<28} {row.restaurants:>5} rest  {row.unique_names:>5} distinct   {top}")


def select_restaurants(city: str, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rest = load_restaurants()
    here = rest[rest["city"].str.lower() == city.lower()]
    if here.empty:
        sys.exit(f"no restaurants for city {city!r}; use --list-cities (format 'City, ST').")
    stats = {"in_city": int(len(here))}
    here = here[here["category"].map(is_food_restaurant)]
    stats["food_restaurants"] = int(len(here))
    # One branch per chain: 20 identical McDonald's menus would count 20 times.
    here = here.sort_values("id").drop_duplicates(subset="name", keep="first")
    stats["after_chain_dedupe"] = int(len(here))

    menus = load_menus(here["id"])
    stats["menu_items"] = int(len(menus))
    keep = [is_dish(s, nm) for s, nm in zip(menus["category"], menus["name"])]
    dishes = menus[keep].copy()
    stats["dropped_non_dishes"] = int(len(menus) - len(dishes))
    dishes = dishes.drop_duplicates(subset=["restaurant_id", "name"])

    per = dishes.groupby("restaurant_id").size()
    eligible = here[here["id"].isin(per[per >= MIN_DISHES].index)]
    stats[f"with_>={MIN_DISHES}_dishes"] = int(len(eligible))
    ids = sorted(eligible["id"].tolist())
    picked = sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    stats["selected"] = len(picked)
    return (eligible[eligible["id"].isin(picked)].sort_values("id"),
            dishes[dishes["restaurant_id"].isin(picked)], stats)


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def _medoid(ids):
    """Of several recipes sharing one name, the one whose taste is closest to
    their mean — a typical version, not whichever has the smallest id. Taste,
    ingredients and allergens then all come from that one coherent recipe."""
    if len(ids) == 1:
        return ids[0]
    vecs = np.array([nbio.recipe_taste(i) for i in ids])
    return ids[int(np.argmin(((vecs - vecs.mean(axis=0)) ** 2).sum(axis=1)))]


class RecipeMatcher:
    def __init__(self):
        recipes = nbio._recipes()
        groups = {}
        with open(nbio.flavour_csv_path(), newline="") as f:
            import csv
            csv.field_size_limit(1 << 24)
            for row in csv.DictReader(f):
                name, rid = _norm(row["name"]), int(row["id"])
                if name and rid in recipes:
                    groups.setdefault(name, []).append(rid)
        self.names = sorted(groups)
        self.ids = np.array([_medoid(sorted(groups[n])) for n in self.names])
        self.group_size = np.array([len(groups[n]) for n in self.names])
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True, dtype=np.float32)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                    sublinear_tf=True, dtype=np.float32)
        self.R = self._stack(self.word.fit_transform(self.names),
                             self.char.fit_transform(self.names)).T.tocsr()

    @staticmethod
    def _stack(w, c):
        # Equal weight to word and char evidence; rows re-normalised so the
        # product below is a cosine in [0, 1].
        m = sparse.hstack([w, c]).tocsr()
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        return sparse.diags(1.0 / norms).dot(m).astype(np.float32)

    def top_k(self, texts, k=3, batch=256):
        Q = self._stack(self.word.transform(texts), self.char.transform(texts))
        idx = np.zeros((len(texts), k), dtype=np.int64)
        val = np.zeros((len(texts), k), dtype=np.float32)
        for s in range(0, len(texts), batch):
            S = Q[s:s + batch].dot(self.R).toarray()
            part = np.argpartition(-S, k, axis=1)[:, :k]
            ps = np.take_along_axis(S, part, axis=1)
            order = np.argsort(-ps, axis=1, kind="stable")
            idx[s:s + batch] = np.take_along_axis(part, order, axis=1)
            val[s:s + batch] = np.take_along_axis(ps, order, axis=1)
        return idx, val


def match_dishes(dishes: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    m = RecipeMatcher()
    print(f"  TF-IDF fitted on {len(m.names):,} recipe names ({time.time() - t0:.0f}s)")

    names = dishes["name"].map(_norm)
    descs = dishes["description"].fillna("").map(_norm)
    uniq = sorted(set(names))
    ui, uv = m.top_k(uniq)
    by_name = {t: (ui[j], uv[j]) for j, t in enumerate(uniq)}

    # Fallback: name + description, only for names that miss the threshold.
    weak = sorted({f"{n} {d}".strip() for n, d in zip(names, descs)
                   if by_name[n][1][0] < MATCH_THRESHOLD and d})
    by_full = {}
    if weak:
        wi, wv = m.top_k(weak)
        by_full = {t: (wi[j], wv[j]) for j, t in enumerate(weak)}
    print(f"  matched {len(uniq):,} distinct dish names, {len(weak):,} description fallbacks "
          f"({time.time() - t0:.0f}s)")

    out = []
    for n, d in zip(names, descs):
        idx, val, used = *by_name[n], "name"
        full = f"{n} {d}".strip()
        if full in by_full and by_full[full][1][0] > val[0]:
            idx, val = by_full[full]
            used = "name+description"
        out.append({"recipe_id": int(m.ids[idx[0]]), "recipe_name": m.names[idx[0]], "same_name_recipes": int(m.group_size[idx[0]]),
                    "match_score": round(float(val[0]), 4), "match_text": used,
                    "alternatives": [m.names[j] for j in idx[1:]]})
    return pd.DataFrame(out, index=dishes.index)


def _price(p):
    try:
        return float(str(p).split()[0])
    except (ValueError, IndexError):
        return None


def build(city: str, n: int, seed: int) -> None:
    t0 = time.time()
    print(f"[1/4] selecting restaurants in {city} ...")
    rest, dishes, stats = select_restaurants(city, n, seed)
    for k, v in stats.items():
        print(f"      {k:<22} {v:,}")

    print(f"[2/4] matching {len(dishes):,} dishes to Food.com recipes ...")
    dishes = dishes.join(match_dishes(dishes))
    dishes["matched"] = dishes["match_score"] >= MATCH_THRESHOLD

    print("[3/4] dish taste + allergens ...")
    restaurants = []
    for _, r in rest.iterrows():
        menu = []
        for _, d in dishes[dishes["restaurant_id"] == r["id"]].iterrows():
            ings = nbio.recipe_ingredients(d["recipe_id"]) if d["matched"] else None
            menu.append({
                "name": str(d["name"]).strip(),
                "description": None if pd.isna(d["description"]) else str(d["description"]),
                "section": None if pd.isna(d["category"]) else str(d["category"]),
                "price": _price(d["price"]),
                "recipe_id": int(d["recipe_id"]) if d["matched"] else None,
                "recipe_name": d["recipe_name"] if d["matched"] else None,
                "match_score": d["match_score"],
                "match_text": d["match_text"],
                "same_name_recipes": int(d["same_name_recipes"]),
                "taste": nbio.recipe_taste(d["recipe_id"]) if d["matched"] else [0.0] * 6,
                "ingredients": ings,
                "allergens": nbio.dish_allergens(ings),
            })
        restaurants.append({
            "option_id": f"ue_{int(r['id'])}", "name": str(r["name"]),
            "category": None if pd.isna(r["category"]) else str(r["category"]),
            "price_range": None if pd.isna(r.get("price_range")) else str(r["price_range"]),
            "menu": menu,
        })

    stats["dishes"] = int(len(dishes))
    stats["matched_dishes"] = int(dishes["matched"].sum())
    meta = {"source": "kaggle ahmedshahriarsakib/uber-eats-usa-restaurants-menus",
            "city": city, "n_requested": n, "seed": seed, "match_threshold": MATCH_THRESHOLD,
            "min_dishes": MIN_DISHES, "bitter_list_sha": nbio.bitter_list_sha(),
            "built": time.strftime("%Y-%m-%d %H:%M"), "stats": stats}
    out = nbio.ubereats_cache_path(city)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "restaurants": restaurants}, indent=1) + "\n")
    print(f"      -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    print("[4/4] hand-check sheet + summary ...")
    write_handcheck(city, dishes, rest, seed)
    catalog = nbio.load_catalog_ubereats(city)
    sp = nbio.taste_spread(catalog)
    print(f"\n  restaurants        {len(catalog)}")
    print(f"  dishes             {stats['dishes']:,}  (matched {stats['matched_dishes']:,})")
    print(f"  taste coverage     {nbio.taste_coverage(catalog):.1%}  (contract: >= 90%)")
    print(f"  spread (mean std)  {sp['spread']:.4f}   per dim: "
          + "  ".join(f"{d} {v:.3f}" for d, v in sp["per_dim_std"].items()))
    print(f"  done in {time.time() - t0:.0f}s")


def write_handcheck(city, dishes, rest, seed) -> None:
    """100 seeded dishes drawn from ALL dishes (matched or not), so the labels
    measure accuracy and also let the threshold be tuned afterwards."""
    names = dict(zip(rest["id"], rest["name"]))
    sample = dishes.sample(n=min(HANDCHECK_N, len(dishes)), random_state=seed)
    sheet = pd.DataFrame({
        "restaurant": sample["restaurant_id"].map(names),
        "dish": sample["name"],
        "description": sample["description"].fillna(""),
        "matched_recipe": sample["recipe_name"],
        "match_score": sample["match_score"],
        "match_text": sample["match_text"],
        "alt_2": sample["alternatives"].map(lambda a: a[0] if len(a) > 0 else ""),
        "alt_3": sample["alternatives"].map(lambda a: a[1] if len(a) > 1 else ""),
        "correct": "",   # fill in: y = same dish, p = close (same family), n = wrong
        "notes": "",
    })
    VALIDATION.mkdir(exist_ok=True)
    path = VALIDATION / f"match_check_{nbio.city_slug(city)}.csv"
    if path.exists():
        print(f"      {path.relative_to(ROOT)} exists — not overwritten (labels may be in it)")
        return
    sheet.to_csv(path, index=False)
    print(f"      -> {path.relative_to(ROOT)} ({len(sheet)} dishes to hand-label)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-cities", action="store_true")
    ap.add_argument("--city", help="'City, ST' exactly as --list-cities prints it")
    ap.add_argument("--n", type=int, default=300, help="restaurants to sample (200-500)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.list_cities:
        list_cities()
    elif a.city:
        build(a.city, a.n, a.seed)
    else:
        ap.error("pass --list-cities or --city")


if __name__ == "__main__":
    main()
