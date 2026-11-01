import os
"""E7: a multi-criteria venue utility built from real catalog attributes
(dish prices, the diner's own safe-menu size) plus taste, with synthetic
per-diner budgets and criterion weights. The aggregation rules and the
mediator are unchanged: only the utility matrix is replaced."""
import sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
from e3_mediation import parser_replay, graded, summarise  # noqa (re-runs e3; cheap)

OUT = os.environ.get("FM_REV_OUT", "figdata_rev")
D = data()
GRID = sorted(set(list(np.round(np.logspace(-2, 3, 61), 4)) + LAMBDAS))


def mc_utility(g, rng, med_price):
    n = g["n"]
    w = rng.dirichlet(8 * np.array([0.6, 0.25, 0.15]), size=n)          # taste, price, choice
    budget = np.exp(rng.normal(np.log(med_price), 0.35, size=n))
    nF = len(g["ids"])
    A = np.full((nF, n), 0.5); C = np.zeros((nF, n))
    for a in range(nF):
        for i in range(n):
            pr = g["prices"][a][i]
            if len(pr):
                A[a, i] = float(np.mean(pr <= budget[i]))
            C[a, i] = min(1.0, np.log1p(g["nsafe"][a, i]) / np.log1p(30))
    U = w[:, 0] * g["S"] + w[:, 1] * A + w[:, 2] * C
    return U, A, C


rows, front = [], []
for city in CITIES:
    allp = np.concatenate([p for g in D[city][:50] for row in g["prices"] for p in row if len(p)])
    med = float(np.median(allp))
    rng = np.random.default_rng(7)
    Us = []
    for g in D[city]:
        U, A, C = mc_utility(g, rng, med)
        Us.append(U)
    within = np.mean([U.std(axis=1).mean() for U in Us])
    best = None
    for lam in GRID:
        r = c1_summary(Us, lam)
        cs = 100 * r["cost"] / r["cost_lm"]
        front.append(dict(city=city, lam=lam, pct=round(r["pct"], 2), cost_share=round(cs, 2)))
        if cs <= 100 / 3:
            best = (lam, r, cs)
    lam_m, rm, cs_m = best
    r5 = c1_summary(Us, 5)
    sel = {}
    for rule in ["average", "least_misery", "most_pleasure"]:
        ps = [pick(U, rule) for U in Us]
        sel[rule] = (np.mean([U[p].min() for U, p in zip(Us, ps)]), np.mean([U[p].mean() for U, p in zip(Us, ps)]))
    # mediation on the multi-criteria utility, with exact and noisy reactions
    med_rows = {}
    for kind, sd in [("exact", 0.0), ("noise sd=0.02", 0.02)]:
        rng2 = np.random.default_rng(11)
        Ts = [U + (rng2.normal(0, sd, U.shape) if sd else 0) for U in Us]
        bars = [T.min(1).max() - MARGIN for T in Ts]
        lm = np.mean([T[pick(U, "least_misery")].min() >= b for U, T, b in zip(Us, Ts, bars)])
        rng3 = np.random.default_rng(23)
        res = [mediate(U, U=T, bar=b, rng=rng3, react_fn=graded, parser=parser_replay, memory=True)
               for U, T, b in zip(Us, Ts, bars)]
        med_rows[kind] = (100 * lm, 100 * np.mean([r["converged"] for r in res]),
                          np.mean([r["rounds"] for r in res if r["converged"]]))
    rows.append(dict(city=city, median_dish_price=round(med, 2), within_option_sd=round(within, 4),
                     achievable=round(rm["achievable"], 4), lam_matched=lam_m,
                     pct_matched=round(rm["pct"], 1), pct_lo=round(rm["pct_lo"], 1), pct_hi=round(rm["pct_hi"], 1),
                     cost_share_matched=round(cs_m, 1), gain_matched=round(rm["gain"], 4),
                     pct_lam5=round(r5["pct"], 1), cost_share_lam5=round(100 * r5["cost"] / r5["cost_lm"], 1),
                     avg_minsat=round(sel["average"][0], 4), lm_minsat=round(sel["least_misery"][0], 4),
                     mp_minsat=round(sel["most_pleasure"][0], 4),
                     lm_agree_exact=round(med_rows["exact"][0], 1), med_agree_exact=round(med_rows["exact"][1], 1),
                     med_rounds_exact=round(med_rows["exact"][2], 2),
                     lm_agree_noise=round(med_rows["noise sd=0.02"][0], 1),
                     med_agree_noise=round(med_rows["noise sd=0.02"][1], 1),
                     med_rounds_noise=round(med_rows["noise sd=0.02"][2], 2)))
    print(rows[-1])

for name, rr in [("e7_multicriteria.csv", rows), ("e7_frontier.csv", front)]:
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
