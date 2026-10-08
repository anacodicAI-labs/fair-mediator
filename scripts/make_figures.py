"""Figures 2-5 of "Should the Mediator Be Fair?" for general readers (PRD-13).

    python scripts/make_figures.py                       # writes into the paper folder
    python scripts/make_figures.py --out some/dir        # anywhere else
    python scripts/make_figures.py --plot-only           # skip step 1, plot from the CSVs

Two steps, so every plotted number lives in a CSV first (PRD-13 section 3):

  1. COMPUTE (needs the gitignored Uber Eats + Food.com caches in data/external_cache/)
     writes three intermediates under figdata/ubereats-foodcom/:
       fig_taste_map.csv       restaurants and the example group's diners on a 2-D
                               taste map (PCA of the 6-D taste vectors, unit-normalised)
       fig_taste_axes.csv      the PCA loadings and explained variance, from which the
                               plain-word axis labels ("sweeter ->") are derived
       fig_example_group.csv   the one example group: each diner's satisfaction at the
                               Average, Fair (lambda=5) and Least-Misery picks
     Houston, Food.com diners, 200 groups, seed 7, 35% adversarial, whole-menu
     formula -- the same groups and picks as scripts/c1_report.py.

  2. PLOT reads only figdata/ CSVs:
       Fig 2  fig2_taste_map        fig_taste_map.csv, fig_taste_axes.csv
       Fig 3  fig3_left_behind      fig_example_group.csv
       Fig 4  fig4_price_of_fairness  c1_summary.csv
       Fig 5  fig5_own_dish         ubereats-{houston-tx,seattle-wa}-foodcom/table5_compensation.csv
       Fig 6  fig6_lambda_sweep     ubereats-{houston-tx,seattle-wa}-foodcom/fig1_lambda_sweep.csv

Example-group rule (Fig 3): among the 200 Houston groups, keep those whose Fair
(lambda=5) pick differs from the Average pick; sort them by the worst-off diner's
gain (MinSat(Fair) - MinSat(Average)), ties by group id; take the median one
(index n // 2). A typical changed group, not the most dramatic.

Outputs are byte-for-byte reproducible: PDF/PNG metadata dates are stripped and
nothing random is drawn.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGDATA = ROOT / "figdata"
INTER = FIGDATA / "ubereats-foodcom"
DEFAULT_OUT = (ROOT.parents[3] / "docs" / "ai-in-food" / "research_fair-mediator"
               / "papers" / "paper 2" / "figures")

SEED, N_GROUPS, ADVERSARIAL_FRAC, LAM = 7, 200, 0.35, 5
DIMS = ["sweet", "salty", "sour", "bitter", "umami", "spicy"]
# plain words for each end of a taste dimension
WORD = {"sweet": "sweeter", "salty": "saltier", "sour": "more sour",
        "bitter": "more bitter", "umami": "more savoury (umami)", "spicy": "spicier"}

NOTE = {"sweet": "sweet tooth", "salty": "loves salty", "sour": "loves sour",
        "bitter": "likes bitter", "umami": "loves savoury", "spicy": "loves spicy"}

# Cuisine groups for the taste map: first category token that matches wins.
CUISINES = [
    ("Sweets, bakery & breakfast", ["dessert", "bakery", "ice cream", "donut", "breakfast",
                                    "coffee", "juice", "smoothie", "crepe", "waffle",
                                    "pancake", "cake", "boba", "bubble tea"]),
    ("Mexican & Latin", ["mexican", "latin", "tex mex", "taco", "venezuelan", "colombian",
                         "salvadoran", "cuban", "peruvian", "honduran", "burrito"]),
    ("Asian & Indian", ["asian", "chinese", "japanese", "sushi", "vietnamese", "thai",
                        "korean", "indian", "pakistani", "noodle", "pho", "ramen",
                        "filipino", "malaysian", "poke"]),
    ("Burgers, chicken & sandwiches", ["burger", "sandwich", "wings", "chicken",
                                       "fast food", "bbq", "hot dog", "deli"]),
]
OTHER = "Other (pizza, seafood, Cajun, ...)"


# ════════════════════════════════════════════════════════════════════════════
# 1. COMPUTE -- writes the intermediate CSVs
# ════════════════════════════════════════════════════════════════════════════

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def _cuisine(category: str) -> str:
    for tok in (t.strip().lower() for t in (category or "").split(",")):
        for name, kws in CUISINES:
            if any(k in tok for k in kws):
                return name
    return OTHER


def compute() -> None:
    import numpy as np
    os.environ.update({"FM_PERSONAS": "foodcom", "FM_DATASET": "ubereats", "FM_CITY": "Houston, TX"})
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import nbio
    from c1_report import matrices, pick       # the exact picks behind c1_summary.csv

    catalog = nbio.load_catalog()
    groups = nbio.generate_groups(n_groups=N_GROUPS, seed=SEED, adversarial_frac=ADVERSARIAL_FRAC)

    # ── the example group ────────────────────────────────────────────────────
    changed = []
    for g in groups:
        S, _ = matrices(g, catalog)
        if not S:
            continue
        p = {"average": pick(S, "average"), "fair": pick(S, "fair", LAM),
             "least_misery": pick(S, "least_misery")}
        if p["fair"] != p["average"]:
            gain = min(S[p["fair"]].values()) - min(S[p["average"]].values())
            changed.append((round(gain, 12), g["group_id"], g, S, p))
    changed.sort(key=lambda t: (t[0], t[1]))
    gain, gid, g, S, p = changed[len(changed) // 2]
    names = {o["option_id"]: html.unescape(o["name"]) for o in catalog}
    members = g["members"]
    s_avg = [S[p["average"]][m["member_id"]] for m in members]
    # one plain-words taste note per diner: the taste they like most relative to
    # the whole Food.com pool (largest z-score of the unit-normalised vector)
    pool = np.array([q["taste_vector"] for q in nbio.load_personas()], dtype=float)
    pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    pmu, psd = pool.mean(axis=0), pool.std(axis=0)

    def note(t):
        v = np.asarray(t, float) / np.linalg.norm(t)
        return NOTE[DIMS[int(np.argmax((v - pmu) / psd))]]

    worst = min(range(len(members)), key=lambda i: s_avg[i])
    rows = []
    for i, m in enumerate(members):
        rows.append({
            "group_id": gid, "n_changed_groups": len(changed), "rank_in_changed": len(changed) // 2 + 1,
            "diner": "ABCDEF"[i], "taste_note": note(m["taste"]), "diet": m["diet"], "allergies": ";".join(m["allergies"]) or "none",
            "sat_average": round(S[p["average"]][m["member_id"]], 4),
            "sat_fair": round(S[p["fair"]][m["member_id"]], 4),
            "sat_least_misery": round(S[p["least_misery"]][m["member_id"]], 4),
            "worst_off_at_average": int(i == worst),
            "pick_average": names[p["average"]], "pick_fair": names[p["fair"]],
            "pick_least_misery": names[p["least_misery"]],
            "n_feasible": len(S),
        })
    _write_csv(INTER / "fig_example_group.csv", rows)

    # ── the taste map: PCA of unit-normalised taste (s_i is a cosine) ────────
    X = np.array([o["taste"] for o in catalog], dtype=float)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    mu = Xn.mean(axis=0)
    _, sv, vt = np.linalg.svd(Xn - mu, full_matrices=False)
    # fix the arbitrary SVD signs: the largest |loading| of each axis is positive
    for k in range(2):
        if vt[k, np.argmax(np.abs(vt[k]))] < 0:
            vt[k] *= -1
    var = sv ** 2 / (sv ** 2).sum()
    axes = [{"axis": f"PC{k + 1}", "explained_var": round(float(var[k]), 4),
             **{d: round(float(vt[k, j]), 4) for j, d in enumerate(DIMS)}} for k in range(2)]
    _write_csv(INTER / "fig_taste_axes.csv", axes)

    def proj(v):
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        return (v - mu) @ vt[:2].T

    tmap = []
    for o in catalog:
        x, y = proj(o["taste"])
        role = ("average_pick" if o["option_id"] == p["average"] else
                "fair_pick" if o["option_id"] == p["fair"] else "")
        tmap.append({"kind": "restaurant", "id": o["option_id"], "cuisine": _cuisine(o.get("category")),
                     "pc1": round(float(x), 5), "pc2": round(float(y), 5), "role": role})
    for i, m in enumerate(members):
        x, y = proj(m["taste"])
        tmap.append({"kind": "diner", "id": "ABCDEF"[i], "cuisine": "",
                     "pc1": round(float(x), 5), "pc2": round(float(y), 5),
                     "role": "worst_off_at_average" if i == worst else ""})
    _write_csv(INTER / "fig_taste_map.csv", tmap)


# ════════════════════════════════════════════════════════════════════════════
# 2. PLOT -- reads only figdata/ CSVs
# ════════════════════════════════════════════════════════════════════════════

def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


CM = 1 / 2.54
W_TEXT, W_FULL = 13.86 * CM, 18.46 * CM          # MDPI \textwidth and full width
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAIR, AVERAGE, LEASTM = "#eb6834", "#2a78d6", "#4a3aa7"     # validated all-pairs set
FAIR_INK = "#b8481c"                                          # darker accent for text
SHUFFLED = "#b9b8b2"
CUISINE_COL = ["#eda100", "#008300", "#7a6fd0", "#e87ba4"]   # validated all-pairs set
CUISINE_MK = ["^", "s", "D", "o"]
OTHER_COL = "#c9c8c2"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 9.5,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42, "svg.hashsalt": "fm",
        "figure.dpi": 100, "savefig.dpi": 300, "text.color": INK,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 6,
        "path.simplify": False,
    })


def _title(fig, title, subtitle=None, y=0.985):
    fig.text(0.012, y, title, ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, y - 0.065 * (2.6 / fig.get_figheight()) * 2.0, subtitle,
                 ha="left", va="top", fontsize=7.8, color=INK2)


def _save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                "Producer": None, "Creator": None})
    fig.savefig(out / f"{stem}.png", dpi=200, metadata={"Software": None})
    print(f"  wrote {stem}.pdf / .png")


def _pct(v: float) -> float:
    """cosine similarity -> 0-100 'taste match' points"""
    return 100.0 * v


# ── Fig 2 ────────────────────────────────────────────────────────────────────
def fig2(out: Path):
    import matplotlib.pyplot as plt
    rows = _read(INTER / "fig_taste_map.csv")
    axes = {r["axis"]: r for r in _read(INTER / "fig_taste_axes.csv")}
    rest = [r for r in rows if r["kind"] == "restaurant"]
    din = [r for r in rows if r["kind"] == "diner"]

    def ends(ax):
        """Plain words for each end of an axis: the dimension with the largest
        loading on that side, plus a second one if it loads >= 0.75 as much."""
        load = {d: float(axes[ax][d]) for d in DIMS}
        out = []
        for sign in (1, -1):
            side = sorted((d for d in DIMS if sign * load[d] > 0), key=lambda d: -abs(load[d]))
            words = [WORD[side[0]]]
            if len(side) > 1 and abs(load[side[1]]) >= 0.75 * abs(load[side[0]]):
                words.append(WORD[side[1]].replace("more ", ""))
            out.append(" & ".join(words))
        return out

    fig, ax = plt.subplots(figsize=(W_FULL, 10.2 * CM))
    fig.subplots_adjust(left=0.035, right=0.70, bottom=0.08, top=0.84)
    # restaurants: "other" first, underneath
    oth = [r for r in rest if r["cuisine"] == OTHER]
    ax.scatter([float(r["pc1"]) for r in oth], [float(r["pc2"]) for r in oth], s=14, marker="o",
               facecolor=OTHER_COL, edgecolor="white", linewidth=0.5, zorder=2)
    counts = {}
    for (name, _), col, mk in zip(CUISINES, CUISINE_COL, CUISINE_MK):
        sel = [r for r in rest if r["cuisine"] == name]
        counts[name] = len(sel)
        ax.scatter([float(r["pc1"]) for r in sel], [float(r["pc2"]) for r in sel], s=16, marker=mk,
                   facecolor=col, edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    counts[OTHER] = len(oth)

    # the two picks for the example group
    for role, col, lab in (("average_pick", AVERAGE, "Average's pick"), ("fair_pick", FAIR, "Fair's pick")):
        r = next(r for r in rest if r["role"] == role)
        x, y = float(r["pc1"]), float(r["pc2"])
        ax.scatter([x], [y], s=120, facecolor="none", edgecolor=col, linewidth=1.8, zorder=6)
        dx, dy = (-0.07, 0.13) if role == "fair_pick" else (0.06, 0.16)
        ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), fontsize=7.5, color=INK, zorder=7,
                    ha="right" if dx < 0 else "left", va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.9, shrinkA=1, shrinkB=5))

    # the diners
    for r in din:
        x, y = float(r["pc1"]), float(r["pc2"])
        worst = r["role"] == "worst_off_at_average"
        ax.scatter([x], [y], s=110, marker="*", facecolor=INK,
                   edgecolor="white", linewidth=0.8, zorder=8)
        ax.text(x + 0.018, y + 0.012, r["id"], fontsize=8, fontweight="bold", zorder=9,
                color=FAIR_INK if worst else INK, ha="left", va="bottom")

    xs = [float(r["pc1"]) for r in rows]
    ys = [float(r["pc2"]) for r in rows]
    xpad = (max(xs) - min(xs)) * 0.06
    ypad = (max(ys) - min(ys)) * 0.10
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad * 1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)

    # plain-word axes, from the loadings
    p1, n1 = ends("PC1")
    p2, n2 = ends("PC2")
    v1 = 100 * float(axes["PC1"]["explained_var"])
    v2 = 100 * float(axes["PC2"]["explained_var"])
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    arrow = dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7)
    ax.annotate("", xy=(x1, y0), xytext=(x0, y0), arrowprops=arrow, annotation_clip=False)
    ax.text(x1, y0 - (y1 - y0) * 0.035, f"{p1} →", ha="right", va="top", fontsize=7.8, color=INK2)
    ax.text(x0, y0 - (y1 - y0) * 0.035, f"← {n1}", ha="left", va="top", fontsize=7.8, color=INK2)
    ax.annotate("", xy=(x0, y1), xytext=(x0, y0), arrowprops=arrow, annotation_clip=False)
    ax.text(x0 + (x1 - x0) * 0.012, y1, f"↑ {p2}", ha="left", va="top", fontsize=7.8, color=INK2)
    ax.text(x0 + (x1 - x0) * 0.012, y0 + (y1 - y0) * 0.02, f"↓ {n2}", ha="left", va="bottom",
            fontsize=7.8, color=INK2)

    # legend panel on the right, drawn by hand so it can carry counts
    lx, ly, step = 0.725, 0.80, 0.058
    fig.text(lx, ly, "Restaurants (Houston, n = %d)" % len(rest), fontsize=7.8, fontweight="bold",
             color=INK, va="center")
    items = [(n, c, m) for (n, _), c, m in zip(CUISINES, CUISINE_COL, CUISINE_MK)] + [(OTHER, OTHER_COL, "o")]
    for k, (name, col, mk) in enumerate(items):
        yy = ly - step * (k + 1)
        fig.lines.append(plt.Line2D([lx + 0.008], [yy], marker=mk, color=col, markersize=4.6,
                                    markeredgecolor="white", markeredgewidth=0.5, linestyle="none",
                                    transform=fig.transFigure))
        fig.text(lx + 0.025, yy, f"{name}  ({counts[name]})", fontsize=7.5, color=INK2, va="center")
    yy = ly - step * (len(items) + 1) - 0.03
    fig.text(lx, yy, "One friend group (n = %d)" % len(din), fontsize=7.8, fontweight="bold",
             color=INK, va="center")
    for k, (mk, kw, txt) in enumerate((
            ("*", dict(color=INK, markersize=7.5), "a diner's own taste (A–%s)" % din[-1]["id"]),
            ("o", dict(color="none", markeredgecolor=AVERAGE, markersize=7.5, markeredgewidth=1.6),
             "restaurant Average picks"),
            ("o", dict(color="none", markeredgecolor=FAIR, markersize=7.5, markeredgewidth=1.6),
             "restaurant Fair picks (λ = 5)"))):
        y2 = yy - step * (k + 1)
        fig.lines.append(plt.Line2D([lx + 0.008], [y2], marker=mk, linestyle="none",
                                    transform=fig.transFigure, **kw))
        fig.text(lx + 0.025, y2, txt, fontsize=7.5, color=INK2, va="center")
    fig.text(lx, 0.04, "Close together = similar taste.\nMap = PCA of the 6 tastes; the two\n"
             f"axes hold {v1:.0f}% and {v2:.0f}% of the spread.",
             fontsize=7, color=MUTED, va="bottom", linespacing=1.35)

    fig.text(0.035, 0.975, "Most restaurants taste alike; sweet places stand apart",
             fontsize=9.5, fontweight="bold", va="top", color=INK)
    fig.text(0.035, 0.915, "300 real Houston restaurants placed by the taste of their menus. The five stars "
             "are one real group of friends.", fontsize=7.8, va="top", color=INK2)
    _save(fig, out, "fig2_taste_map")
    plt.close(fig)


# ── Fig 3 ────────────────────────────────────────────────────────────────────
def fig3(out: Path):
    import matplotlib.pyplot as plt
    rows = _read(INTER / "fig_example_group.csv")
    rows.sort(key=lambda r: float(r["sat_average"]), reverse=True)   # worst at the bottom
    n = len(rows)

    def who(r):
        bits = []
        if r["diet"].lower() not in ("mix", "non-veg", "nonveg", ""):
            bits.append(r["diet"].lower())
        if r["allergies"] != "none":
            bits.append(r["allergies"].replace(";", ", ") + " allergy")
        return f"{r['diner']} \u00b7 {r['taste_note']}" + (f" ({'; '.join(bits)})" if bits else "")

    fig, ax = plt.subplots(figsize=(W_TEXT, (3.3 + 0.75 * n) * CM))
    H = 3.6 + 0.75 * n
    fig.subplots_adjust(left=0.20, right=0.97, bottom=1.25 / H, top=1 - 2.35 / H)
    lo = min(min(_pct(float(r["sat_average"])), _pct(float(r["sat_fair"]))) for r in rows)
    xmin = 5 * int((lo - 3) // 5)
    for i, r in enumerate(rows):
        y = n - 1 - i
        a, f = _pct(float(r["sat_average"])), _pct(float(r["sat_fair"]))
        worst = r["worst_off_at_average"] == "1"
        if worst:
            ax.axhspan(y - 0.42, y + 0.42, color="#f6e3da", lw=0, zorder=0)
        ax.plot([a, f], [y, y], color=FAIR if worst else "#d8d7d1", lw=2.2 if worst else 1.6,
                zorder=2, solid_capstyle="round")
        ax.scatter([a], [y], s=34, marker="o", facecolor="white", edgecolor=AVERAGE, linewidth=1.5, zorder=4)
        ax.scatter([f], [y], s=40, marker="D", facecolor=FAIR, edgecolor="white", linewidth=0.8, zorder=5)
        d = f - a
        ax.text(max(a, f) + 0.7, y, f"{d:+.1f}".replace("-", "\u2212") + ("   the worst-off friend" if worst else ""),
                va="center", ha="left", fontsize=7.5,
                color=FAIR_INK if worst else INK2, fontweight="bold" if worst else "normal")
    ax.set_yticks(range(n))
    labels = [who(r) for r in rows][::-1]
    ax.set_yticklabels(labels)
    for t, r in zip(ax.get_yticklabels(), rows[::-1]):
        if r["worst_off_at_average"] == "1":
            t.set_fontweight("bold"); t.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(xmin, 100)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("Taste match with the chosen restaurant (0–100; axis starts at %d)" % xmin)


    # legend row (direct, under the title)
    top = 1 - 1.45 / H
    fig.lines.append(plt.Line2D([0.20], [top], marker="o", markerfacecolor="white", markeredgecolor=AVERAGE,
                                markeredgewidth=1.5, markersize=5.5, linestyle="none", transform=fig.transFigure))
    fig.text(0.215, top, f"Average picks {rows[0]['pick_average']}", va="center", fontsize=7.5, color=INK2)
    fig.lines.append(plt.Line2D([0.20], [top - 0.32 / H * 1.3], marker="D", markerfacecolor=FAIR, markeredgecolor="white",
                                markersize=5.5, linestyle="none", transform=fig.transFigure))
    fig.text(0.215, top - 0.32 / H * 1.3, f"Fair (λ = 5) picks {rows[0]['pick_fair']}", va="center",
             fontsize=7.5, color=INK2)
    fig.text(0.012, 1 - 0.25 / H, "The fair pick asks the happiest friends to give a little so the "
             "least happy one gains", fontsize=9.5, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 1 - 0.75 / H, f"One real Houston group of {n} (group {rows[0]['group_id']}); each line "
             "joins a friend's score at the two restaurants.", fontsize=7.8, va="top", color=INK2)
    _save(fig, out, "fig3_left_behind")
    plt.close(fig)


# ── Fig 4 ────────────────────────────────────────────────────────────────────
def fig4(out: Path):
    import matplotlib.pyplot as plt
    rows = [r for r in _read(INTER / "c1_summary.csv") if r["formula"] == "whole menu"]
    get = lambda ds, lam: next(r for r in rows if r["dataset"] == ds and float(r["lambda"]) == lam)

    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 8.6 * CM), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.78, wspace=0.10)
    for ax, city in zip(axs, ("Houston, TX", "Seattle, WA")):
        r5, r05 = get(city, 5), get(city, 0.5)
        s5 = get(city + " shuffled", 5)
        lm_cost = -_pct(float(r5["meansat_change_least_misery"]))
        f_cost = -_pct(float(r5["meansat_change"]))
        f05_cost = -_pct(float(r05["meansat_change"]))
        sh_lm = -_pct(float(s5["meansat_change_least_misery"]))
        sh_f = -_pct(float(s5["meansat_change"]))
        f_y, lo, hi = (float(r5[k]) for k in ("pct_of_achievable", "pct_of_achievable_ci_lo",
                                               "pct_of_achievable_ci_hi"))
        f05_y = float(r05["pct_of_achievable"])
        sh_y, sh_lo, sh_hi = (float(s5[k]) for k in ("pct_of_achievable", "pct_of_achievable_ci_lo",
                                                      "pct_of_achievable_ci_hi"))
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        # shuffled control: faded
        ax.plot([0, sh_f, sh_lm], [0, sh_y, 100], color=SHUFFLED, lw=1.0, zorder=2)
        ax.errorbar([sh_f], [sh_y], yerr=[[sh_y - sh_lo], [sh_hi - sh_y]], fmt="D", ms=4.5,
                    color=SHUFFLED, mfc="white", mec=SHUFFLED, mew=1.1, elinewidth=0.9, capsize=0, zorder=3)
        ax.scatter([sh_lm], [100], marker="s", s=26, facecolor="white", edgecolor=SHUFFLED, lw=1.1, zorder=3)
        ax.text(sh_f + 0.03, sh_y - 9, f"shuffled menus\n(control): {sh_y:.0f}%", fontsize=7, color=MUTED,
                ha="left", va="top")
        # real menus
        ax.plot([0, f05_cost, f_cost, lm_cost], [0, f05_y, f_y, 100], color="#d8d7d1", lw=1.0, zorder=2)
        ax.scatter([0], [0], s=42, marker="o", facecolor="white", edgecolor=AVERAGE, lw=1.6, zorder=5)
        ax.scatter([lm_cost], [100], s=38, marker="s", facecolor=LEASTM, edgecolor="white", lw=0.8, zorder=5)
        ax.scatter([f05_cost], [f05_y], s=16, marker="D", facecolor="white", edgecolor=FAIR, lw=1.0, zorder=5)
        ax.errorbar([f_cost], [f_y], yerr=[[f_y - lo], [hi - f_y]], fmt="none", ecolor=FAIR,
                    elinewidth=1.5, capsize=2.5, capthick=1.2, zorder=4)
        ax.scatter([f_cost], [f_y], s=58, marker="D", facecolor=FAIR, edgecolor="white", lw=0.9, zorder=6)
        # direct labels
        ax.text(0.035, -8.5, "Average (the starting point)", fontsize=7.2, color=INK2, ha="left", va="center")
        ax.text(lm_cost - 0.03, 100 - 4, f"Least-Misery\n100% by definition\ncost {lm_cost:.2f}", fontsize=7.2, color=INK2,
                ha="right", va="top")
        ax.text(f05_cost + 0.035, f05_y, "Fair, λ = 0.5", fontsize=7, color=MUTED, ha="left", va="center")
        ax.text(f_cost + 0.045, f_y + 1, f"Fair (λ = 5)\n{f_y:.0f}% [{lo:.0f}–{hi:.0f}]  "
                f"cost {f_cost:.2f}", fontsize=7.5, color=FAIR_INK, fontweight="bold", ha="left", va="center")
        ax.set_title(city.split(",")[0], fontsize=8.5, color=INK, pad=4)
        ax.set_xlim(-0.03, 1.08)
        ax.set_ylim(-13, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axs[0].set_ylabel("Share of the possible gain\nfor the worst-off diner")
    fig.text(0.535, 0.03, "Cost to the group: drop in average taste match (points on a 0–100 scale) →",
             ha="center", fontsize=8, color=INK2)
    fig.text(0.012, 0.975, "Fair gets about two-thirds of the way to Least-Misery, at about a third of its cost",
             fontsize=9.5, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 0.905, "Averages over 200 real-diner groups per city; up and to the left is better. "
             "Grey: the same test on shuffled menus.", fontsize=7.8, va="top", color=INK2)
    _save(fig, out, "fig4_price_of_fairness")
    plt.close(fig)


# ── Fig 5 ────────────────────────────────────────────────────────────────────
LADDER = [  # (prefix in the CSV 'level' column, plain-English label)
    ("B1", "Everyone gets a random safe dish"),
    ("B3", "Everyone shares the middle diner's favourite dish"),
    ("B2", "Everyone shares the best single dish"),
    ("per-diner", "Each person orders their own best safe dish"),
]


def fig5(out: Path):
    import matplotlib.pyplot as plt
    hou = _read(FIGDATA / "ubereats-houston-tx-foodcom" / "table5_compensation.csv")
    sea = _read(FIGDATA / "ubereats-seattle-wa-foodcom" / "table5_compensation.csv")
    pick = lambda rows, pre: next(r for r in rows if r["level"].startswith(pre))

    H = 8.6
    fig, ax = plt.subplots(figsize=(W_TEXT, H * CM))
    fig.subplots_adjust(left=0.03, right=0.965, bottom=1.25 / H, top=1 - 2.2 / H)
    n = len(LADDER)
    for i, (pre, lab) in enumerate(LADDER):
        y = n - 1 - i                       # first rung at the top, best at the bottom
        h, s = pick(hou, pre), pick(sea, pre)
        hv, hci = _pct(float(h["min_sat"])), _pct(float(h["min_sat_ci95"]))
        sv = _pct(float(s["min_sat"]))
        ours = pre == "per-diner"
        ax.barh(y + 0.02, hv, height=0.30, color=FAIR if ours else "#8e8d87", edgecolor="none", zorder=3)
        ax.errorbar([hv], [y + 0.02], xerr=[[hci], [hci]], fmt="none", ecolor=INK, elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=4)
        ax.text(hv - hci - 1.0, y + 0.02, f"{hv:.0f}", va="center", ha="right", fontsize=7.8, color="white",
                fontweight="bold", zorder=5)
        ax.barh(y - 0.2, sv, height=0.10, color="#f3b497" if ours else "#d3d2cc", edgecolor="none", zorder=3)
        ax.text(sv + 1.0, y - 0.2, f"{sv:.0f}", va="center", ha="left", fontsize=7, color=MUTED, zorder=5)
        ax.text(0, y + 0.22, f"{i + 1}.  {lab}", ha="left", va="bottom", fontsize=7.8,
                color=INK if ours else INK2, fontweight="bold" if ours else "normal")
    ven = pick(hou, "venue")
    vv = _pct(float(ven["min_sat"]))
    ax.plot([vv, vv], [-0.45, n - 0.3], color=AVERAGE, lw=0.9, zorder=2)
    ax.text(vv + 0.8, n - 0.3, f"the restaurant's\nwhole-menu score: {vv:.0f}", fontsize=7, color=AVERAGE,
            ha="right", va="bottom", linespacing=1.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, n + 0.05)
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("Taste match of the least-happy diner in the group (0\u2013100)")
    # key
    ky = 1 - 1.55 / H
    fig.lines.append(plt.Line2D([0.035], [ky], marker="s", color="#8e8d87", markersize=5, linestyle="none",
                                transform=fig.transFigure))
    fig.text(0.05, ky, "Houston (bar, with 95% CI)", fontsize=7.4, color=INK2, va="center")
    fig.lines.append(plt.Line2D([0.33, 0.355], [ky, ky], color="#d3d2cc", lw=2.6, transform=fig.transFigure))
    fig.text(0.365, ky, "Seattle (thin bar, same test)", fontsize=7.4, color=INK2, va="center")
    fig.text(0.012, 1 - 0.25 / H, "Letting each friend order their own dish lifts the least-happy diner most",
             fontsize=9.5, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 1 - 0.75 / H, "Same chosen restaurant, four ways to order; averages over 200 real-diner "
             "groups.", fontsize=7.8, va="top", color=INK2)
    _save(fig, out, "fig5_own_dish")
    plt.close(fig)


# ── Fig 6 ────────────────────────────────────────────────────────────────────
PLATEAU_TOL = 0.001   # MinSat "stops improving": first lambda within this of the city's best MinSat


def fig6(out: Path):
    import matplotlib.pyplot as plt
    cities = (("Houston", "ubereats-houston-tx-foodcom"), ("Seattle", "ubereats-seattle-wa-foodcom"))
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 8.4 * CM), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.77, wspace=0.08)
    ticks = [0, 0.5, 1, 2, 5, 10, 25, 50, 100]
    for k, (ax, (city, folder)) in enumerate(zip(axs, cities)):
        rows = sorted(_read(FIGDATA / folder / "fig1_lambda_sweep.csv"), key=lambda r: float(r["lambda"]))
        lam = [float(r["lambda"]) for r in rows]
        mean = [_pct(float(r["mean_sat"])) for r in rows]
        mins = [_pct(float(r["min_sat"])) for r in rows]
        best = max(float(r["min_sat"]) for r in rows)
        plat = next(r for r in rows if float(r["min_sat"]) >= best - PLATEAU_TOL)
        lp, mp = float(plat["lambda"]), _pct(float(plat["min_sat"]))
        r5 = next(r for r in rows if float(r["lambda"]) == LAM)

        ax.set_xscale("symlog", linthresh=0.25, linscale=0.45)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        ax.plot([LAM, LAM], [85, 96.6], color="#f6d3c2", lw=5, zorder=1, solid_capstyle="butt")
        ax.fill_between(lam, mins, mean, color="#f0efec", lw=0, zorder=1)
        ax.plot(lam, mean, color=INK2, lw=1.6, zorder=3)
        ax.scatter(lam, mean, s=16, marker="o", facecolor="white", edgecolor=INK2, lw=1.1, zorder=4)
        ax.plot(lam, mins, color=FAIR, lw=2.0, zorder=3)
        ax.scatter(lam, mins, s=20, marker="D", facecolor=FAIR, edgecolor="white", lw=0.6, zorder=4)
        # where MinSat stops improving
        ax.annotate(f"stops improving\nfrom \u03bb \u2248 {lp:g}  ({mp:.1f})", xy=(lp, mp), xytext=(lp, mp - 2.6),
                    ha="center", va="top", fontsize=7, color=FAIR_INK,
                    arrowprops=dict(arrowstyle="-", color=FAIR_INK, lw=0.7, shrinkA=1, shrinkB=4))
        # lambda = 5 values
        m5, n5 = _pct(float(r5["min_sat"])), _pct(float(r5["mean_sat"]))
        ax.text(LAM * 0.82, m5 + 0.35, f"{m5:.1f}", ha="right", va="bottom", fontsize=7.2, color=FAIR_INK,
                fontweight="bold")
        ax.text(LAM * 0.82, n5 - 0.3, f"{n5:.1f}", ha="right", va="top", fontsize=7.2, color=INK2)
        ax.text(LAM, 96.7, "\u03bb = 5 (headline)", ha="center", va="bottom", fontsize=7, color=FAIR_INK)
        # start values and direct series labels
        ax.text(-0.075, mins[0], f"{mins[0]:.1f}", ha="right", va="center", fontsize=7.2, color=FAIR_INK)
        ax.text(-0.075, mean[0], f"{mean[0]:.1f}", ha="right", va="center", fontsize=7.2, color=INK2)
        if k == 0:
            ax.text(0.3, mean[0] + 0.45, "everyone, on average", fontsize=7.5, color=INK2, va="bottom")
            ax.text(0.3, mins[0] - 0.45, "the least-happy diner", fontsize=7.5, color=FAIR_INK, va="top",
                    fontweight="bold")
        ax.set_title(city, fontsize=8.5, color=INK, pad=4)
        ax.set_xlim(-0.22, 130)
        ax.set_ylim(85, 97.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.tick_params(axis="x", which="both", length=2.5)
    axs[0].set_ylabel("Taste match (0\u2013100; axis starts at 85)")
    fig.text(0.53, 0.035, "Fairness dial \u03bb (0 = Average; higher = more weight on equality) \u2192",
             ha="center", fontsize=8, color=INK2)
    fig.text(0.012, 0.975, "Turning up the fairness dial lifts the least-happy diner, then levels off; "
             "the group gives up little", fontsize=9.5, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 0.905, "Averages over 200 real-diner groups per city. The shaded gap between the "
             "two lines is how far the least-happy diner trails the group.", fontsize=7.8, va="top",
             color=INK2)
    _save(fig, out, "fig6_lambda_sweep")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: the paper's figures/)")
    ap.add_argument("--plot-only", action="store_true", help="skip the compute step; plot from existing CSVs")
    a = ap.parse_args()
    if not a.plot_only:
        print("1. compute intermediates")
        compute()
    print(f"2. plot into {a.out}")
    _style()
    for f in (fig2, fig3, fig4, fig5, fig6):
        f(a.out)


if __name__ == "__main__":
    main()
