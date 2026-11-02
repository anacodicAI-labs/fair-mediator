import os
"""How often does the loop without memory re-propose a pick that was already objected to?"""
import sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from e3_mediation import parser_replay, graded
D = data()
rows = []
for city in CITIES:
    for kind, sd in [("exact", 0.0), ("noise sd=0.01", 0.01)]:
        rng0 = np.random.default_rng(11)
        Ss = [g["S"] for g in D[city]]
        Us = [S + (rng0.normal(0, sd, S.shape) if sd else 0) for S in Ss]
        rng = np.random.default_rng(23)
        rep, props = 0, 0
        for S, U in zip(Ss, Us):
            bar = U.min(1).max() - MARGIN
            n = S.shape[1]; w = np.full(n, 1 / n); rejected = set()
            for rnd in range(CAP):
                p = pick(S, "fair", 0.5, w)
                props += 1
                if p in rejected: rep += 1
                u = U[p]
                if u.min() >= bar: break
                labels = [parser_replay(l, rng) for l in graded(u, bar, int(np.argmin(u)), rng)]
                if any(l != "accept" for l in labels): rejected.add(p)
                w = w * np.exp(ETA * np.array([GRAD[l] for l in labels])); w /= w.sum()
        rows.append(dict(city=city, truth=kind, proposals=props, repeated_rejected=rep,
                         pct=round(100 * rep / props, 1)))
        print(rows[-1])
with open(os.environ.get("FM_REV_OUT", "figdata_rev") + "/e5b_repropose.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
