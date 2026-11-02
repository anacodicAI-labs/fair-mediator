"""Safety stress test: does the safety filter bite, and do the results hold,
when diners are far more restricted than the US population?

realistic = the Food.com personas as built (synthetic diet/allergy at US rates)
stress    = every diner gets 1-3 of the FDA-9 allergens; 20% vegetarian, 10% vegan
For each city and setting: restaurants left per group, groups losing any,
one-pick vs mediator vs veto agreement, and the fair dial (lambda = 5).
Also the share of a menu that is unsafe for typical restricted profiles.
Writes figdata/ubereats-foodcom/safety_stress.csv and safety_dish_share.csv.
    FM_PERSONAS=foodcom FM_DATASET=ubereats python scripts/safety_stress.py
"""
from __future__ import annotations
import csv, os, random, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import nbio  # noqa: E402
import c1_report as c  # noqa: E402

FDA = ["shellfish", "milk", "peanuts", "tree_nuts", "fish", "egg", "wheat", "soy", "sesame"]
SEED = 7


def stressed(base):
    r = random.Random(SEED)
    out = []
    for p in base:
        d = r.random()
        if d < 0.10:
            diet = "vegan"
        elif d < 0.30:
            diet = "vegetarian"
        else:
            diet = "non-veg"
        q = dict(p)
        q["diet_type"] = diet
        q["allergies"] = r.sample(FDA, r.randint(1, 3))
        out.append(q)
    return out


def write(name, rows):
    out = ROOT / "figdata" / "ubereats-foodcom" / name
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("->", out.relative_to(ROOT))


def main():
    os.environ["FM_PERSONAS"] = "foodcom"
    os.environ["FM_DATASET"] = "ubereats"
    base = nbio.load_personas()
    real_loader = nbio.load_personas
    rows = []
    for setting, loader in (("realistic", lambda: base), ("stress", lambda: stressed(base))):
        nbio.load_personas = loader
        for city in ("Houston, TX", "Seattle, WA"):
            os.environ["FM_CITY"] = city
            cat = nbio.load_catalog()
            groups = nbio.generate_groups(200, seed=SEED)
            F = [len(nbio.build_feasible_set(cat, g["members"])) for g in groups]
            one = med = veto = n = 0
            for g in groups:
                r = nbio.run_group(g, cat, "fair", 0.5)
                if not r:
                    continue
                S, ids = r["_S"], r["_member_ids"]
                bar = nbio.consensus_bar(S, ids)
                n += 1
                one += min(S[r["pick_id"]][m] for m in ids) >= bar
                med += nbio.mediate(S, ids)["converged"]
                veto += nbio.mediate(S, ids, use_veto=True)["converged"]
            c.N_BOOT = 2000
            res, _ = c.evaluate(city, "ubereats", city)
            x = [y for y in res if y["formula"] == "whole menu" and y["lambda"] == 5][0]
            rows.append({"setting": setting, "city": city, "n_groups": len(groups),
                         "F_mean": round(st.mean(F), 1), "F_min": min(F),
                         "groups_losing_any": sum(1 for f in F if f < len(cat)), "groups_empty": F.count(0),
                         "one_pick_agree": one, "mediator_agree": med, "veto_agree": veto,
                         "fair_pct_of_achievable": x["pct_of_achievable"], "fair_minsat_gain": x["minsat_gain"]})
            print(rows[-1], flush=True)
    nbio.load_personas = real_loader
    write("safety_stress.csv", rows)
    os.environ["FM_CITY"] = "Houston, TX"
    cat = nbio.load_catalog()
    sample = random.Random(SEED).sample(cat, 60)
    prof = [("milk allergy", "non-veg", ["milk"]), ("wheat allergy", "non-veg", ["wheat"]),
            ("vegetarian", "vegetarian", []), ("vegan", "vegan", []),
            ("milk + egg + wheat allergies", "non-veg", ["milk", "egg", "wheat"]),
            ("vegan + wheat + soy allergies", "vegan", ["wheat", "soy"])]
    rows2 = []
    for label, diet, allergies in prof:
        shares = []
        for o in sample:
            safe = nbio.safe_dishes_for({"diet": diet, "allergies": allergies}, o["menu"])
            shares.append(1 - len(safe) / len(o["menu"]))
        rows2.append({"profile": label, "city": "Houston, TX", "n_restaurants": len(sample),
                      "share_of_menu_unsafe": round(st.mean(shares), 3)})
    restricted = [p for p in base if p["allergies"] or p["diet_type"] != "non-veg"]
    who = random.Random(SEED).sample(restricted, 150)
    shares = []
    for p in who:
        for o in sample:
            safe = nbio.safe_dishes_for({"diet": p["diet_type"], "allergies": p["allergies"]}, o["menu"])
            shares.append(1 - len(safe) / len(o["menu"]))
    rows2.append({"profile": "average restricted Food.com diner", "city": "Houston, TX",
                  "n_restaurants": len(sample), "share_of_menu_unsafe": round(st.mean(shares), 3)})
    for r in rows2:
        print(r)
    write("safety_dish_share.csv", rows2)


if __name__ == "__main__":
    main()
