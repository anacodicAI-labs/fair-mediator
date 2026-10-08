"""PRD-14: test the theory section (Propositions 1-4) on real diners.

    python scripts/pareto_check.py        # Food.com diners on Uber Eats Houston + Seattle

Same groups, feasible sets, satisfaction matrix and pick rule as c1_report.py
(whole-menu formula A; pick = first maximum of nbio.aggregate over F).

For every lambda in the harness sweep and every group it checks

  dominated     is the Fair pick Pareto-dominated inside F (some option weakly
                better for every diner, strictly for one)?            -- Prop. 1
  threshold     lambda* (n) = n / (2(n-1)), the exact bound on [0,1]^n, and
                lambda*_data = n / (2(n-1) R), R = range of the group's S. Below
                either bound Prop. 1 says 0 dominated; the counts are reported.
  loss / bound  mean(x_avg) - mean(x_fair) against lambda (Var(x_avg) - Var(x_fair)),
                lambda Var(x_avg), lambda max_F Var and lambda/4  -- Prop. 2
  minsat        MinSat(x_fair) < MinSat(x_avg)?                     -- Prop. 3
  monotone      Var and mean of the Fair pick non-increasing in lambda -- Prop. 4

plus the mediator (nbio.mediate, lambda 0.5, the paper's settings): is its final
pick dominated, and is its weighted threshold n^2 w_min / (2(n-1)) met?

Writes figdata/ubereats-foodcom/pareto_check.csv: rows per (dataset, group size,
lambda), group_size "all" first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402

SEED, N_GROUPS, ADVERSARIAL_FRAC = 7, 200, 0.35       # as in 06-evaluate
SWEEP = [0, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]       # as in 06-evaluate
CITIES = ["Houston, TX", "Seattle, WA"]
OUT = ROOT / "figdata" / "ubereats-foodcom"
TOL = 1e-12


def var(v):
    return nbio.variance(v)


def dominated(S, pick, mids):
    """True if some option in F is >= for every diner and > for at least one."""
    p = [S[pick][m] for m in mids]
    for o, row in S.items():
        if o == pick:
            continue
        q = [row[m] for m in mids]
        if all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p)):
            return True
    return False


def pick(S, method, lam=0.0):
    # max() keeps the first of equal scores, as in c1_report.py
    return max(S, key=lambda o: nbio.aggregate(list(S[o].values()), method, lam))


def per_group(catalog, groups):
    out = []
    for g in groups:
        members = g["members"]
        mids = [m["member_id"] for m in members]
        F = nbio.build_feasible_set(catalog, members)
        if not F:
            continue
        S = {o["option_id"]: {m["member_id"]: nbio.taste_similarity(m["taste"], o["taste"])
                              for m in members} for o in F}
        n = len(mids)
        vals = [s for row in S.values() for s in row.values()]
        R = max(vals) - min(vals)
        avg = pick(S, "average")
        va = [S[avg][m] for m in mids]
        rec = {
            "n": n, "F": len(S), "R": R,
            "lam_star_n": n / (2 * (n - 1)),
            "lam_star_data": n / (2 * (n - 1) * R) if R > 0 else float("inf"),
            "max_var": max(var(list(row.values())) for row in S.values()),
            "any_dominated_option": any(dominated(S, o, mids) for o in S),
            "avg": va, "fair": {},
        }
        for lam in SWEEP:
            f = pick(S, "fair", lam)
            rec["fair"][lam] = {"vals": [S[f][m] for m in mids], "dom": dominated(S, f, mids)}
        med = nbio.mediate(S, mids, lam=0.5)
        if med["pick"] is not None:
            last = med["history"][-1]
            wmin = min(last["weights"].values())
            rec["med"] = {"dom": dominated(S, last["pick"], mids),
                          "lam_star_w": n * n * wmin / (2 * (n - 1)), "wmin": wmin}
        out.append(rec)
    return out


def rows_for(label, recs, size):
    gs = recs if size == "all" else [r for r in recs if r["n"] == size]
    if not gs:
        return []
    rows = []
    prev = None
    for lam in SWEEP:
        dom = [r["fair"][lam]["dom"] for r in gs]
        below_n = [r for r in gs if lam <= r["lam_star_n"]]
        below_d = [r for r in gs if lam <= r["lam_star_data"]]
        loss, b_gap, b_avg, b_max = [], [], [], []
        viol = drop = 0
        for r in gs:
            vf, va = r["fair"][lam]["vals"], r["avg"]
            L = nbio.mean_sat(va) - nbio.mean_sat(vf)
            bg = lam * (var(va) - var(vf))
            loss.append(L); b_gap.append(bg)
            b_avg.append(lam * var(va)); b_max.append(lam * r["max_var"])
            viol += (L > bg + TOL) or (bg > lam * var(va) + TOL) or (L < -TOL)
            drop += min(vf) < min(va) - TOL
        mono_var = mono_mean = 0
        if prev is not None:
            for r in gs:
                a, b = r["fair"][prev]["vals"], r["fair"][lam]["vals"]
                mono_var += var(b) > var(a) + TOL
                mono_mean += nbio.mean_sat(b) > nbio.mean_sat(a) + TOL
        k = len(gs)
        rows.append({
            "dataset": label, "group_size": size, "lambda": lam, "n_groups": k,
            "lambda_star_n": round(gs[0]["lam_star_n"], 4) if size != "all" else "",
            "n_dominated": sum(dom), "pct_dominated": round(100 * sum(dom) / k, 1),
            "n_below_lambda_star_n": len(below_n),
            "n_dominated_below_lambda_star_n": sum(r["fair"][lam]["dom"] for r in below_n),
            "n_below_lambda_star_data": len(below_d),
            "n_dominated_below_lambda_star_data": sum(r["fair"][lam]["dom"] for r in below_d),
            "pct_groups_with_any_dominated_option": round(100 * sum(r["any_dominated_option"] for r in gs) / k, 1),
            "meansat_loss_mean": round(sum(loss) / k, 6),
            "meansat_loss_max": round(max(loss), 6),
            "bound_var_gap_mean": round(sum(b_gap) / k, 6),
            "bound_var_avg_mean": round(sum(b_avg) / k, 6),
            "bound_max_var_F_mean": round(sum(b_max) / k, 6),
            "bound_quarter": round(lam / 4, 6),
            "n_p2_violations": viol,
            "n_minsat_drop": drop, "pct_minsat_drop": round(100 * drop / k, 1),
            "n_var_increase_vs_prev_lambda": mono_var if prev is not None else "",
            "n_mean_increase_vs_prev_lambda": mono_mean if prev is not None else "",
        })
        prev = lam
    return rows


def mediator_row(label, recs):
    ms = [r["med"] for r in recs if "med" in r]
    below = [m for m in ms if 0.5 <= m["lam_star_w"]]
    return {"dataset": label, "group_size": "all", "lambda": "mediator 0.5", "n_groups": len(ms),
            "n_dominated": sum(m["dom"] for m in ms),
            "pct_dominated": round(100 * sum(m["dom"] for m in ms) / len(ms), 1),
            "n_below_lambda_star_n": len(below),
            "n_dominated_below_lambda_star_n": sum(m["dom"] for m in below),
            "lambda_star_n": round(min(m["lam_star_w"] for m in ms), 6)}


def main():
    os.environ["FM_PERSONAS"] = "foodcom"
    os.environ["FM_DATASET"] = "ubereats"
    rows, meds = [], []
    for city in CITIES:
        print(f"... {city}", flush=True)
        os.environ["FM_CITY"] = city
        catalog = nbio.load_catalog()
        groups = nbio.generate_groups(n_groups=N_GROUPS, seed=SEED, adversarial_frac=ADVERSARIAL_FRAC)
        recs = per_group(catalog, groups)
        for size in ["all", 3, 4, 5, 6]:
            rows += rows_for(city, recs, size)
        meds.append(mediator_row(city, recs))
        R = [r["R"] for r in recs]
        print(f"    groups {len(recs)}  S range per group: min {min(R):.3f} median "
              f"{sorted(R)[len(R) // 2]:.3f} max {max(R):.3f}", flush=True)
    rows += meds
    cols = list(rows[0].keys())
    rows = [{c: r.get(c, "") for c in cols} for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["FM_FIGDATA"] = str(OUT.relative_to(ROOT))
    nbio.save_figdata("pareto_check.csv", rows)
    nbio.table([r for r in rows if r["group_size"] == "all"],
               cols=["dataset", "lambda", "n_groups", "n_dominated", "pct_dominated",
                     "n_dominated_below_lambda_star_n", "meansat_loss_mean", "bound_var_avg_mean",
                     "n_p2_violations", "pct_minsat_drop"], width=14)


if __name__ == "__main__":
    main()
