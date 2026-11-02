import os
"""E1/E2: robustness of C1 to the satisfaction function, and the order-aware
(dish-level) utility. Writes CSVs to OUT."""
import sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

OUT = os.environ.get("FM_REV_OUT", "figdata_rev")
import os; os.makedirs(OUT, exist_ok=True)
D = data()
UTILS = [("S", "cosine, whole menu (paper)"), ("Dsafe", "cosine, safe-subset menu"),
         ("Se", "Euclidean similarity"), ("Sc", "centred cosine"),
         ("D3", "order-aware, mean of top-3 safe dishes"), ("D", "order-aware, best safe dish")]
GRID = sorted(set(list(np.round(np.logspace(-2, 3, 61), 4)) + LAMBDAS))


def frontier(Ms):
    """% achievable vs cost share of LM, over the lambda grid."""
    rows = []
    for lam in GRID:
        r = c1_summary(Ms, lam)
        rows.append((lam, r["pct"], 100 * r["cost"] / r["cost_lm"] if r["cost_lm"] else np.nan, r))
    return rows


def at_cost_share(rows, target=100 / 3):
    """Largest lambda whose cost share does not exceed target."""
    ok = [r for r in rows if not np.isnan(r[2]) and r[2] <= target]
    return max(ok, key=lambda r: r[0])


out1, out_front = [], []
for city in CITIES:
    for key, label in UTILS:
        Ms = [g[key] for g in D[city]]
        fr = frontier(Ms)
        for lam, pct, cs, r in fr:
            out_front.append(dict(city=city, utility=key, lam=lam, pct=round(pct, 2), cost_share=round(cs, 2)))
        lam_m, pct_m, cs_m, rm = at_cost_share(fr)
        r5 = c1_summary(Ms, 5)
        # selector ordering on MinSat / Jain
        order = {}
        for rule in ["average", "least_misery", "most_pleasure", "fair"]:
            ps = [pick(M, rule, 0.5 if key != "Sc" else 0.5) for M in Ms]
            order[rule] = (np.mean([M[p].min() for M, p in zip(Ms, ps)]),
                           np.mean([M[p].mean() for M, p in zip(Ms, ps)]))
        out1.append(dict(city=city, utility=key, label=label,
                         achievable=round(rm["achievable"], 4),
                         lam_matched=lam_m, pct_matched=round(pct_m, 1),
                         pct_matched_lo=round(rm["pct_lo"], 1), pct_matched_hi=round(rm["pct_hi"], 1),
                         cost_share_matched=round(cs_m, 1),
                         gain_matched=round(rm["gain"], 4),
                         pct_lam5=round(r5["pct"], 1), cost_share_lam5=round(100 * r5["cost"] / r5["cost_lm"], 1),
                         mp_minsat=round(order["most_pleasure"][0], 4), avg_minsat=round(order["average"][0], 4),
                         lm_minsat=round(order["least_misery"][0], 4)))
        print(out1[-1])


def write(name, rows):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


write("e1_utilities.csv", out1)
write("e1_frontier.csv", out_front)

# --- E1b: picks selected on the paper's proxy S, evaluated under other utilities
out2 = []
for city in CITIES:
    for key, label in UTILS[1:]:
        Ms = [g["S"] for g in D[city]]; Es = [g[key] for g in D[city]]
        for lam in (0.5, 5):
            gains, lmg, mf = [], [], []
            for M, E in zip(Ms, Es):
                a = pick(M, "average"); f = pick(M, "fair", lam); l = pick(M, "least_misery")
                gains.append(E[f].min() - E[a].min()); lmg.append(E[l].min() - E[a].min())
                mf.append(E[f].mean() - E[a].mean())
            lo, hi = boot_ci(gains)
            # oracle achievable under E
            orc = np.mean([E.min(1).max() - E[pick(M, "average")].min() for M, E in zip(Ms, Es)])
            out2.append(dict(city=city, eval_utility=key, lam=lam, gain=round(np.mean(gains), 4),
                             lo=round(lo, 4), hi=round(hi, 4), lm_gain=round(np.mean(lmg), 4),
                             oracle_gap=round(orc, 4),
                             pct_of_oracle=round(100 * np.sum(gains) / (orc * len(gains)), 1),
                             mean_change=round(np.mean(mf), 4)))
            print(out2[-1])
write("e1b_cross_eval.csv", out2)

# --- E2: candidate-set size, whole-menu vs order-aware utility
out3 = []
rng = np.random.default_rng(7)
for city in CITIES:
    for K in (10, 30, 100, None):
        for key in ("S", "D3", "D"):
            Ms = []
            for g in D[city]:
                M = g[key]
                if K is not None:
                    idx = np.sort(np.random.default_rng(int(g["group_id"][1:]) + 7).choice(len(M), K, replace=False))
                    M = M[idx]
                Ms.append(M)
            r = c1_summary(Ms, 5)
            out3.append(dict(city=city, K=K or "all", utility=key, achievable=round(r["achievable"], 4),
                             gain_lam5=round(r["gain"], 4), lo=round(r["gain_lo"], 4), hi=round(r["gain_hi"], 4),
                             pct_lam5=round(r["pct"], 1), cost_lam5=round(r["cost"], 4), cost_lm=round(r["cost_lm"], 4)))
            print(out3[-1])
write("e2_candidate_size.csv", out3)
