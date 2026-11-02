"""Generate LaTeX table bodies (rows only) for the revision from figdata_rev/*.csv,
so that every number in the new tables traces to a CSV."""
import pandas as pd, numpy as np, os
R = os.environ.get("FM_REV_OUT", "figdata_rev")
T = os.path.join(os.environ.get("FM_REV_OUT", "figdata_rev"), "tex")
os.makedirs(T, exist_ok=True)
H, S = "Houston, TX", "Seattle, WA"


def f(x, d=4):
    return f"{x:.{d}f}"


def lam(x):
    x = float(x)
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"


# ---------------- utilities table
u = pd.read_csv(f"{R}/e1_utilities.csv")
mc = pd.read_csv(f"{R}/e7_multicriteria.csv")
names = {"S": "Cosine, whole menu (paper)", "Dsafe": "Cosine, diner's safe dishes",
         "Se": "Euclidean similarity", "Sc": "Centred cosine",
         "D3": "Order-aware, top-3 safe dishes", "D": "Order-aware, best safe dish"}
lines = []
for key in ["S", "Dsafe", "Se", "Sc", "D3", "D"]:
    cells = [names[key]]
    for c in (H, S):
        r = u[(u.city == c) & (u.utility == key)].iloc[0]
        cells += [f(r.achievable), lam(r.lam_matched),
                  f"{r.pct_matched:.1f} [{r.pct_matched_lo:.1f}, {r.pct_matched_hi:.1f}]", f"{r.pct_lam5:.1f}"]
    lines.append(" & ".join(cells) + r" \\")
lines.append(r"\midrule")
cells = ["Multi-criteria (taste, price, choice)"]
for c in (H, S):
    r = mc[mc.city == c].iloc[0]
    cells += [f(r.achievable), lam(r.lam_matched), f"{r.pct_matched:.1f} [{r.pct_lo:.1f}, {r.pct_hi:.1f}]", f"{r.pct_lam5:.1f}"]
lines.append(" & ".join(cells) + r" \\")
open(f"{T}/tab_utilities.tex", "w").write("\n".join(lines) + "\n")

# ---------------- experienced (dish-level) outcome
e = pd.read_csv(f"{R}/e2b_experienced.csv")
lines = []
for i, p in enumerate(e[e.city == H].pipeline):
    if i == 4:
        lines.append(r"\midrule")
    cells = [p.replace("lambda=", r"$\lambda{=}").replace(")", "$)") if "lambda" in p else p]
    for c in (H, S):
        r = e[(e.city == c) & (e.pipeline == p)].iloc[0]
        cells += [f"{r.exp_minsat:.4f} [{r.lo:.4f}, {r.hi:.4f}]", f(r.exp_mean)]
    lines.append(" & ".join(cells) + r" \\")
open(f"{T}/tab_experienced.tex", "w").write("\n".join(lines) + "\n")

# ---------------- mediation quality + veto ablation
q = pd.read_csv(f"{R}/e3_mediation_quality.csv")
order = ["one-shot Average", "one-shot Fair (lambda=0.5)", "one-shot Fair (lambda=5)", "one-shot Least-Misery",
         "oracle: best mean subject to the bar", "mediator (paper)", "mediator + veto (paper)",
         "removal only (g=0.5)", "larger gradient only (g=1.5, no removal)",
         "memory: never re-propose an objected pick"]
pretty = {"one-shot Average": "One-shot Average", "one-shot Fair (lambda=0.5)": r"One-shot Fair ($\lambda{=}0.5$)",
          "one-shot Fair (lambda=5)": r"One-shot Fair ($\lambda{=}5$)", "one-shot Least-Misery": "One-shot Least-Misery",
          "oracle: best mean subject to the bar": "Oracle: best mean that clears the bar",
          "mediator (paper)": r"Mediator ($g{=}0.5$, no removal)",
          "mediator + veto (paper)": r"Mediator + veto ($g{=}1.5$, removal)",
          "removal only (g=0.5)": r"Removal only ($g{=}0.5$)",
          "larger gradient only (g=1.5, no removal)": r"Larger gradient only ($g{=}1.5$)",
          "memory: never re-propose an objected pick": "Memory: objected picks never re-proposed"}
lines = []
for i, a in enumerate(order):
    if i in (5, 7):
        lines.append(r"\midrule")
    cells = [pretty[a]]
    for c in (H, S):
        r = q[(q.city == c) & (q.arm == a)].iloc[0]
        ra = r.rounds_agree
        cells += [f"{r.agree} ({r.pct_agree:.1f}\\%)", f"{float(ra):.2f}", f(r.final_minsat), f(r.final_mean)]
    lines.append(" & ".join(cells) + r" \\")
open(f"{T}/tab_medquality.tex", "w").write("\n".join(lines) + "\n")

# ---------------- proxy gap
p = pd.read_csv(f"{R}/e5_proxy_gap.csv")
truths = ["proxy is exact", "noise sd=0.01", "noise sd=0.02", "order-aware (top-3 dishes)"]
arms = list(dict.fromkeys(p.arm))
apretty = {"one-shot Average": "One-shot Average", "one-shot Fair (lambda=5)": r"One-shot Fair ($\lambda{=}5$)",
           "one-shot Least-Misery": "One-shot Least-Misery",
           "mediator, worst-off objects (paper rule)": "Loop, worst-off diner objects (original rule)",
           "mediator, graded multi-diner reactions": "Loop, graded reactions from every diner",
           "  + LLM parser replay": r"\quad + LLM-parser replay",
           "  + LLM parser replay + memory": r"\quad + LLM-parser replay + memory",
           "  + LLM parser replay + veto token": r"\quad + LLM-parser replay + veto token"}
lines = []
for i, a in enumerate(arms):
    if i == 3:
        lines.append(r"\midrule")
    cells = [apretty[a]]
    for t in truths:
        hh = p[(p.city == H) & (p.arm == a) & (p.truth == t)].iloc[0]
        ss = p[(p.city == S) & (p.arm == a) & (p.truth == t)].iloc[0]
        cells.append(f"{hh.pct_agree:.1f} / {ss.pct_agree:.1f}")
    lines.append(" & ".join(cells) + r" \\")
open(f"{T}/tab_proxygap.tex", "w").write("\n".join(lines) + "\n")

# ---------------- lambda table
l = pd.read_csv(f"{R}/e6_lambda.csv"); fd = pd.read_csv(f"{R}/e6_first_dominated.csv")
lines = []
for c in (H, S):
    r = l[l.city == c].iloc[0]; d = fd[fd.city == c].iloc[0]
    lines.append(" & ".join([c.split(",")[0], f"{r.first_move_median:.2f}", f"{r.range_all_median:.2f}",
                             f"{r.range_competitive_median:.2f}", f"{d.first_dominated_min:.1f}",
                             f"{int(r.pf_differs_lam25)}", f"{int(r.pf_differs_lam50)}", f"{int(r.pf_differs_lam100)}"]) + r" \\")
open(f"{T}/tab_lambda.tex", "w").write("\n".join(lines) + "\n")

# ---------------- frontier per-utility CSVs for pgfplots
fr = pd.read_csv(f"{R}/e1_frontier.csv"); f7 = pd.read_csv(f"{R}/e7_frontier.csv")
for c, tag in ((H, "hou"), (S, "sea")):
    for key in ["S", "Se", "Sc", "D3", "D"]:
        d = fr[(fr.city == c) & (fr.utility == key)].sort_values("lam")
        d = d[(d.cost_share.notna()) & (d.cost_share <= 100.5)]
        d[["cost_share", "pct"]].to_csv(f"{T}/front_{tag}_{key}.csv", index=False)
    d = f7[(f7.city == c)].sort_values("lam"); d = d[(d.cost_share.notna()) & (d.cost_share <= 100.5)]
    d[["cost_share", "pct"]].to_csv(f"{T}/front_{tag}_MC.csv", index=False)
print("ok")
for fn in sorted(os.listdir(T)):
    if fn.endswith(".tex"):
        print("==", fn); print(open(f"{T}/{fn}").read())
