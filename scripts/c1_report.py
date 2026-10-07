"""C1 on real restaurants (task doc §7.4-7.7): the numbers the harness does not
print directly, computed with the same nbio functions and the same seeded groups.

    python scripts/c1_report.py            # potluck + Houston + Seattle, real and shuffled

For every dataset it reports:
  - moved     % of groups where the pick at lambda=0 differs from lambda=5
              (and from any lambda in the harness sweep)
  - gain      per-group MinSat(fair) - MinSat(average), mean and a 95%
              bootstrap CI over groups, with the mean-satisfaction cost
  - achieved  the gain as a % of the achievable gap: least-misery's MinSat
              minus Average's, i.e. the most any option could lift the
              worst-off diner (ratio of sums, bootstrap CI over groups)
  - formula   the same under both satisfaction formulas (§7.6):
                A whole menu   s_i(x) = cos(t_i, mean taste of all dishes at x)
                B safe subset  s_i(x) = cos(t_i, mean taste of i's safe dishes at x)

Writes figdata/ubereats/c1_summary.csv and figdata/ubereats/table3_datasets.csv
(fair vs average for both cities, formula A, lambda 0.5 as in Table 1).

Groups share members drawn from one pool of 95 personas, so they are not fully
independent; the CI is over groups and is optimistic by that much.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402

SEED, N_GROUPS, ADVERSARIAL_FRAC = 7, 200, 0.35       # as in 06-evaluate
SWEEP = [0, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]       # as in 06-evaluate
LAMS = [0.5, 5]
N_BOOT = 10_000
OUT = ROOT / "figdata" / "ubereats"

DATASETS = [
    ("potluck (5 restaurants)", "potluck", None),
    ("Houston, TX", "ubereats", "Houston, TX"),
    ("Houston, TX shuffled", "ubereats-shuffled", "Houston, TX"),
    ("Seattle, WA", "ubereats", "Seattle, WA"),
    ("Seattle, WA shuffled", "ubereats-shuffled", "Seattle, WA"),
]


def boot_ci(xs, seed=SEED):
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]


def boot_ratio_ci(num, den, seed=SEED):
    """95% bootstrap CI of sum(num)/sum(den), resampling groups. A ratio of
    sums, not a mean of per-group ratios: a group whose gap is 0 has nothing to
    recover and must not divide by zero or count as 100%."""
    rng = random.Random(seed)
    n = len(num)
    ratios = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(den[i] for i in idx)
        ratios.append(sum(num[i] for i in idx) / d if d > 0 else 0.0)
    ratios.sort()
    return ratios[int(0.025 * N_BOOT)], ratios[int(0.975 * N_BOOT) - 1]


def matrices(group, catalog):
    """S_A and S_B over the feasible set, {option_id: {member_id: s}}."""
    members = group["members"]
    F = nbio.build_feasible_set(catalog, members)
    SA, SB = {}, {}
    for o in F:
        SA[o["option_id"]] = {m["member_id"]: nbio.taste_similarity(m["taste"], o["taste"]) for m in members}
        SB[o["option_id"]] = {m["member_id"]: nbio.taste_similarity(
            m["taste"], nbio._restaurant_taste(o["safe_dishes"][m["member_id"]])) for m in members}
    return SA, SB


def pick(S, method, lam=0.0):
    # max() keeps the first of equal scores, as rank_candidates' stable sort does.
    return max(S, key=lambda o: nbio.aggregate(list(S[o].values()), method, lam))


def evaluate(label, dataset, city):
    os.environ["FM_DATASET"] = dataset
    if city:
        os.environ["FM_CITY"] = city
    catalog = nbio.load_catalog()
    groups = nbio.generate_groups(n_groups=N_GROUPS, seed=SEED, adversarial_frac=ADVERSARIAL_FRAC)
    per = {"A": [], "B": []}
    for g in groups:
        SA, SB = matrices(g, catalog)
        if not SA:
            continue
        for key, S in (("A", SA), ("B", SB)):
            p = {"average": pick(S, "average"), "least_misery": pick(S, "least_misery")}
            p.update({lam: pick(S, "fair", lam) for lam in set(SWEEP) | set(LAMS)})
            v = {k: list(S[o].values()) for k, o in p.items()}
            per[key].append({"picks": p, "vals": v, "F": len(S)})

    rows = []
    for key, gs in per.items():
        n = len(gs)
        moved5 = sum(g["picks"][0] != g["picks"][5] for g in gs)
        moved_any = sum(len({g["picks"][l] for l in SWEEP}) > 1 for g in gs)
        for lam in LAMS:
            d_min = [min(g["vals"][lam]) - min(g["vals"]["average"]) for g in gs]
            d_mean = [nbio.mean_sat(g["vals"][lam]) - nbio.mean_sat(g["vals"]["average"]) for g in gs]
            d_jain = [nbio.jain(g["vals"][lam]) - nbio.jain(g["vals"]["average"]) for g in gs]
            changed = sum(g["picks"][lam] != g["picks"]["average"] for g in gs)
            lo, hi = boot_ci(d_min)
            # achievable: the most any option in F could raise the worst-off
            # diner over the Average pick -- least-misery's pick, by definition
            gap = [min(g["vals"]["least_misery"]) - min(g["vals"]["average"]) for g in gs]
            pct = 100 * sum(d_min) / sum(gap) if sum(gap) > 0 else 0.0
            plo, phi = boot_ratio_ci(d_min, gap)
            lm_cost = [nbio.mean_sat(g["vals"]["least_misery"]) - nbio.mean_sat(g["vals"]["average"]) for g in gs]
            rows.append({
                "dataset": label, "formula": {"A": "whole menu", "B": "safe subset"}[key],
                "lambda": lam, "n_groups": n, "mean_F": round(sum(g["F"] for g in gs) / n, 1),
                "pct_moved_0_vs_5": round(100 * moved5 / n, 1),
                "pct_moved_any_lambda": round(100 * moved_any / n, 1),
                "pct_pick_differs_from_avg": round(100 * changed / n, 1),
                "minsat_avg": round(sum(min(g["vals"]["average"]) for g in gs) / n, 4),
                "minsat_fair": round(sum(min(g["vals"][lam]) for g in gs) / n, 4),
                "minsat_gain": round(sum(d_min) / n, 4),
                "minsat_gain_ci_lo": round(lo, 4), "minsat_gain_ci_hi": round(hi, 4),
                "meansat_change": round(sum(d_mean) / n, 4),
                "jain_change": round(sum(d_jain) / n, 4),
                "minsat_achievable_gap": round(sum(gap) / n, 4),
                "pct_of_achievable": round(pct, 1),
                "pct_of_achievable_ci_lo": round(100 * plo, 1),
                "pct_of_achievable_ci_hi": round(100 * phi, 1),
                "meansat_change_least_misery": round(sum(lm_cost) / n, 4),
            })
    return rows, per


def table3_rows(label, per_a):
    out = []
    for sel, key in (("fair (lambda 0.5)", 0.5), ("average", "average")):
        row = {"dataset": f"Uber Eats {label}", "selector": sel, "n_groups": len(per_a)}
        for metric, fn in (("mean_sat", nbio.mean_sat), ("min_sat", min), ("jain", nbio.jain)):
            m, ci, _ = nbio.mean_ci([fn(g["vals"][key]) for g in per_a])
            row[metric], row[metric + "_ci95"] = round(m, 4), round(ci, 4)
        out.append(row)
    return out


def main():
    summary, table3 = [], []
    for label, dataset, city in DATASETS:
        print(f"... {label}", flush=True)
        rows, per = evaluate(label, dataset, city)
        summary += rows
        if dataset == "ubereats":
            table3 += table3_rows(label, per["A"])
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["FM_FIGDATA"] = str(OUT.relative_to(ROOT))
    nbio.save_figdata("c1_summary.csv", summary)
    nbio.save_figdata("table3_datasets.csv", table3)
    print()
    nbio.table(summary, cols=["dataset", "formula", "lambda", "pct_moved_0_vs_5", "minsat_gain",
                              "minsat_gain_ci_lo", "minsat_gain_ci_hi", "meansat_change"], width=16)
    print()
    nbio.table(table3, cols=["dataset", "selector", "mean_sat", "min_sat", "jain"], width=22)


if __name__ == "__main__":
    main()
