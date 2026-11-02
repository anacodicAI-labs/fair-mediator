"""Build per-group matrices for the revision experiments.

For each catalog (Houston, Seattle, Houston-shuffled, Seattle-shuffled) and each
of the 200 seeded groups (seed 7, 35% adversarial, Food.com diners), store:
  ids      option ids of F (feasible set), in catalog order
  S        |F| x n whole-menu cosine satisfaction (the paper's Eq. 3)
  Sc       |F| x n centred cosine (diner and restaurant vectors minus pool means)
  Se       |F| x n Euclidean similarity 1 - ||t_i - t_x|| / sqrt(6)
  D        |F| x n best safe dish cosine (order-aware venue utility)
  D3       |F| x n mean of the top-3 safe dishes (softer order-aware utility)
  Dsafe    |F| x n cosine to the mean of the diner's safe dishes
  nsafe    |F| x n number of safe dishes
  pmed     |F| x n median price of the diner's safe dishes (NaN if none priced)
  pfrac    dict budget -> |F| x n share of safe priced dishes <= budget (filled later)
  safe_prices list of per-option per-diner price arrays (for budget utility)
Run with the repo path as argv[1].
"""
import os, sys, pickle, math
import numpy as np

REPO = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["FM_PERSONAS"] = "foodcom"
import nbio  # noqa: E402

groups = nbio.generate_groups(n_groups=200, seed=7, adversarial_frac=0.35)
pool = nbio.load_personas()
pool_mean = np.mean([p["taste_vector"] for p in pool], axis=0)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return nbio.taste_similarity(list(a), list(b))


def build(catalog, label):
    rest_mean = np.mean([o["taste"] for o in catalog], axis=0)
    out = []
    for g in groups:
        members = g["members"]
        F = nbio.build_feasible_set(catalog, members)
        n = len(members)
        ids = [o["option_id"] for o in F]
        S = np.zeros((len(F), n)); Sc = np.zeros_like(S); Se = np.zeros_like(S)
        D = np.zeros_like(S); D3 = np.zeros_like(S); Dsafe = np.zeros_like(S)
        nsafe = np.zeros_like(S); prices = []
        T = [np.asarray(m["taste"], float) for m in members]
        for a, o in enumerate(F):
            tx = np.asarray(o["taste"], float)
            prow = []
            for i, m in enumerate(members):
                S[a, i] = cos(m["taste"], o["taste"])
                Sc[a, i] = float(unit(T[i] - pool_mean) @ unit(tx - rest_mean))
                Se[a, i] = 1 - np.linalg.norm(T[i] - tx) / math.sqrt(6)
                safe = o["safe_dishes"][m["member_id"]]
                known = [d for d in safe if sum(abs(x) for x in d["taste"]) > 0]
                sims = sorted((cos(m["taste"], d["taste"]) for d in safe), reverse=True)
                D[a, i] = sims[0]
                D3[a, i] = float(np.mean(sims[:3]))
                if known:
                    mt = np.mean([d["taste"] for d in known], axis=0)
                    Dsafe[a, i] = cos(m["taste"], mt)
                else:
                    Dsafe[a, i] = S[a, i]
                nsafe[a, i] = len(safe)
                prow.append(np.array([d.get("price") for d in safe if d.get("price")], float))
            prices.append(prow)
        out.append(dict(group_id=g["group_id"], adversarial=g["adversarial"], n=n,
                        member_ids=[m["member_id"] for m in members],
                        ids=ids, S=S, Sc=Sc, Se=Se, D=D, D3=D3, Dsafe=Dsafe,
                        nsafe=nsafe, prices=prices,
                        price_range=[o.get("price_range") for o in F],
                        category=[o.get("category") for o in F]))
    print(label, "done", file=sys.stderr)
    return out


data = {}
for city in ["Houston, TX", "Seattle, WA"]:
    cat = nbio.load_catalog_ubereats(city)
    data[city] = build(cat, city)
    data[city + " shuffled"] = build(nbio.shuffle_menus(cat, seed=7), city + " shuffled")
data["_groups"] = groups
with open(OUT, "wb") as f:
    pickle.dump(data, f)
