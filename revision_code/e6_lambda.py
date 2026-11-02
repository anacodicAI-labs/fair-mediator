import os
"""E6: why lambda=5 is safe on these data. Per-group exact critical lambda
(the smallest lambda at which some dominated option can outscore one of its
dominators), the effective lambda on a normalised scale, crossover lambda,
and the Pareto-filtered Fair rule."""
import sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

OUT = os.environ.get("FM_REV_OUT", "figdata_rev")
D = data()


def crit_lambda(S):
    m = S.mean(1); v = S.var(1)
    best = np.inf
    # y dominates x: all S[y] >= S[x], some >
    for x in range(len(S)):
        ge = np.all(S >= S[x], axis=1) & np.any(S > S[x], axis=1)
        ys = np.where(ge & (v > v[x]))[0]
        if len(ys):
            best = min(best, np.min((m[ys] - m[x]) / (v[ys] - v[x])))
    return best


def pareto_front(S):
    keep = np.ones(len(S), bool)
    for x in range(len(S)):
        if np.any(np.all(S >= S[x], axis=1) & np.any(S > S[x], axis=1)):
            keep[x] = False
    return keep


rows, grows = [], []
for city in CITIES:
    crits, Rs, Rtop, cross, fronts = [], [], [], [], []
    for g in D[city]:
        S = g["S"]
        lc = crit_lambda(S); crits.append(lc)
        Rs.append(S.max() - S.min())
        # range over the options Fair could pick for lambda<=5 (price-of-fairness set)
        a = pick(S, "average")
        cand = S.mean(1) >= S[a].mean() - 5 * S.var(1).max()
        Rtop.append(S[cand].max() - S[cand].min())
        a0 = pick(S, "fair", 0)
        moved = [lam for lam in np.logspace(-2, 3, 201) if pick(S, "fair", lam) != a0]
        cross.append(moved[0] if moved else np.inf)
        fr = pareto_front(S); fronts.append(fr)
        grows.append(dict(city=city, group_id=g["group_id"], n=g["n"], lambda_crit=round(lc, 3),
                          range_all=round(Rs[-1], 4), first_move=round(cross[-1], 3),
                          front_size=int(fr.sum()), F_size=len(S)))
    crits = np.array(crits); cross = np.array(cross)
    # Pareto-filtered Fair vs Fair over the sweep
    diff = {}
    for lam in LAMBDAS:
        diff[lam] = sum(pick(g["S"], "fair", lam) != pick(g["S"], "fair", lam, mask=fr)
                        for g, fr in zip(D[city], fronts))
    finite = cross[np.isfinite(cross)]
    rows.append(dict(city=city,
                     crit_min=round(crits.min(), 2), crit_p5=round(np.percentile(crits, 5), 2),
                     crit_median=round(np.median(crits), 2),
                     groups_crit_le_5=int((crits <= 5).sum()), groups_crit_le_25=int((crits <= 25).sum()),
                     range_all_median=round(np.median(Rs), 3),
                     range_competitive_median=round(np.median(Rtop), 3),
                     front_size_median=int(np.median([f.sum() for f in fronts])),
                     first_move_median=round(np.median(finite), 2), groups_never_move=int((~np.isfinite(cross)).sum()),
                     **{f"pf_differs_lam{lam}": diff[lam] for lam in LAMBDAS}))
    print(rows[-1])

for name, rr in [("e6_lambda.csv", rows), ("e6_lambda_groups.csv", grows)]:
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
