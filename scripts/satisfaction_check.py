"""Does taste-similarity satisfaction predict what Food.com users like or choose?

Per user (>= 10 star ratings), a seeded 20% of their rated recipes is held out;
the user's taste is built from the other 80% exactly as nbio.build_personas_foodcom
does (4-5 star recipes). Two tests on the held-out recipes:
  ratings: does cosine(user, recipe) separate liked (4-5) from not liked (1-3)?
  choice:  does cosine rank a held-out recipe the user chose above a random
           recipe they never rated?
Each is an AUC (0.5 = chance), for the personal taste vector, a non-personal
baseline (the mean taste of all users) and a centred variant.
Writes figdata/ubereats-foodcom/satisfaction_check.csv.
    python scripts/satisfaction_check.py
"""
from __future__ import annotations
import csv, math, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402

SEED = 7


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def mean_vec(vs):
    out = []
    for i in range(6):
        out.append(sum(v[i] for v in vs) / len(vs))
    return out


def main():
    rng = random.Random(SEED)
    by_user = {}
    with open(nbio.FOODCOM_INTERACTIONS, newline="") as f:
        for row in csv.DictReader(f):
            s = int(row["rating"])
            if s > 0:
                by_user.setdefault(int(row["user_id"]), []).append((int(row["recipe_id"]), s))
    taste = {}

    def T(r):
        if r not in taste:
            taste[r] = nbio.recipe_taste(r)
        return taste[r]

    known = lambda r: sum(T(r)) > 0
    all_ids = [r for r in nbio._recipes().keys()]
    split, pool = {}, []
    for u, v in by_user.items():
        if len(v) < nbio.FOODCOM_MIN_RATINGS:
            continue
        v = [x for x in v if known(x[0])]
        if len(v) < 5:
            continue
        rng.shuffle(v)
        k = max(1, len(v) // 5)
        test, train = v[:k], v[k:]
        liked = [T(r) for r, s in train if s >= nbio.FOODCOM_LIKE_STARS]
        if not liked:
            continue
        split[u] = (mean_vec(liked), test)
        pool += liked
    centre = mean_vec(pool)
    sub = lambda v: [x - c for x, c in zip(v, centre)]
    wins = {}
    counts = {}
    for name in ("rating_personal", "rating_everyone", "choice_personal", "choice_everyone", "choice_centred"):
        wins[name] = 0.0
        counts[name] = 0
    liked_c, notliked_c = {"p": [], "e": []}, {"p": [], "e": []}
    for u, (tu, test) in split.items():
        seen = {r for r, _ in by_user[u]}
        for r, s in test:
            cp, ce = cos(tu, T(r)), cos(centre, T(r))
            if s >= nbio.FOODCOM_LIKE_STARS:
                liked_c["p"].append(cp); liked_c["e"].append(ce)
            else:
                notliked_c["p"].append(cp); notliked_c["e"].append(ce)
            neg = rng.choice(all_ids)
            while neg in seen or not known(neg):
                neg = rng.choice(all_ids)
            for name, a, b in (("choice_personal", cos(tu, T(r)), cos(tu, T(neg))),
                               ("choice_everyone", cos(centre, T(r)), cos(centre, T(neg))),
                               ("choice_centred", cos(sub(tu), sub(T(r))), cos(sub(tu), sub(T(neg))))):
                if a > b:
                    wins[name] += 1.0
                elif a == b:
                    wins[name] += 0.5
                counts[name] += 1
    r2 = random.Random(SEED)
    for key, name in (("p", "rating_personal"), ("e", "rating_everyone")):
        n = 200000
        for _ in range(n):
            a, b = r2.choice(liked_c[key]), r2.choice(notliked_c[key])
            if a > b:
                wins[name] += 1.0
            elif a == b:
                wins[name] += 0.5
        counts[name] = n
    rows = []
    for name in wins:
        test, who = name.split("_")
        rows.append({"test": test, "taste_vector": who, "auc": round(wins[name] / counts[name], 4),
                     "n_pairs": counts[name], "n_users": len(split)})
    out = ROOT / "figdata" / "ubereats-foodcom" / "satisfaction_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"held-out ratings: liked {len(liked_c['p']):,}, not liked {len(notliked_c['p']):,} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
