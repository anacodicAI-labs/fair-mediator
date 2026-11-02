"""E3-E5: mediation quality, veto 2x2 ablation, mediation under a
proxy-truth gap with richer reactions and LLM-parser replay."""
import sys, csv, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
import pandas as pd

OUT = os.environ.get("FM_REV_OUT", "figdata_rev")
D = data()


def write(name, rows):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def summarise(res, Us, bars, label, city, extra=None):
    conv = [r for r in res if r["converged"]]
    picks = [(r["pick"], U) for r, U in zip(res, Us) if r["pick"] is not None]
    row = dict(city=city, arm=label, n=len(res), agree=len(conv),
               pct_agree=round(100 * len(conv) / len(res), 1),
               rounds_agree=round(np.mean([r["rounds"] for r in conv]), 2) if conv else "--",
               rounds_capped=round(np.mean([r["rounds"] for r in res]), 2),
               exhausted=sum(r["reason"] == "exhausted" for r in res),
               final_minsat=round(np.mean([U[p].min() for p, U in picks]), 4),
               final_mean=round(np.mean([U[p].mean() for p, U in picks]), 4),
               final_jain=round(np.mean([jain(U[p]) for p, U in picks]), 4))
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------- E3: quality
rows = []
for city in CITIES:
    G = D[city]
    Ss = [g["S"] for g in G]
    bars = [S.min(1).max() - MARGIN for S in Ss]

    def oneshot(rule, lam=0.5, label=None):
        res = []
        for S, b in zip(Ss, bars):
            p = pick(S, rule, lam)
            res.append(dict(converged=S[p].min() >= b, rounds=1, pick=p, reason="oneshot"))
        return summarise(res, Ss, bars, label or rule, city)

    rows.append(oneshot("average", label="one-shot Average"))
    rows.append(oneshot("fair", 0.5, "one-shot Fair (lambda=0.5)"))
    rows.append(oneshot("fair", 5, "one-shot Fair (lambda=5)"))
    rows.append(oneshot("least_misery", label="one-shot Least-Misery"))
    # oracle: best mean among options that clear the bar (known utilities)
    res = []
    for S, b in zip(Ss, bars):
        ok = S.min(1) >= b
        p = pick(S, "average", mask=ok)
        res.append(dict(converged=True, rounds=1, pick=p, reason="oracle"))
    rows.append(summarise(res, Ss, bars, "oracle: best mean subject to the bar", city))
    rows.append(summarise([mediate(S) for S in Ss], Ss, bars, "mediator (paper)", city))
    rows.append(summarise([mediate(S, veto_remove=True, veto_grad=True) for S in Ss], Ss, bars,
                          "mediator + veto (paper)", city))
    # E4: veto 2x2 ablation
    rows.append(summarise([mediate(S, veto_remove=True) for S in Ss], Ss, bars,
                          "removal only (g=0.5)", city))
    rows.append(summarise([mediate(S, veto_grad=True) for S in Ss], Ss, bars,
                          "larger gradient only (g=1.5, no removal)", city))
    rows.append(summarise([mediate(S, memory=True) for S in Ss], Ss, bars,
                          "memory: never re-propose an objected pick", city))
for r in rows:
    print(r)
write("e3_mediation_quality.csv", rows)

# dominated mediated picks (Section 4.5 contradiction)
dom_rows = []
for city in CITIES:
    nd, nconv, nonuni = 0, 0, 0
    for g in D[city]:
        S = g["S"]
        r = mediate(S)
        p = r["pick"]
        dominated = np.any(np.all(S >= S[p], axis=1) & np.any(S > S[p], axis=1))
        nd += dominated; nonuni += r["rounds"] > 1
    dom_rows.append(dict(city=city, mediated_picks_dominated=int(nd), groups_reweighted=int(nonuni)))
    print(dom_rows[-1])
write("e3_mediated_dominance.csv", dom_rows)

# ------------------------------------------------- E5: proxy-truth gap
P = pd.read_csv(os.path.join(os.environ.get("FM_REPO", "fair-mediator"), "data/objections/parser_predictions.csv"))
POOL = {lab: P[P.gold_label == lab].pred_label.tolist() for lab in GRAD}


def parser_replay(label, rng):
    """Map an intended label to the label the cached gpt-oss-20b parse of a
    real hand-labelled free-text reaction with that gold label produced."""
    opts = POOL[label]
    return opts[rng.integers(len(opts))]


def worst_only(u, bar, worst, rng):
    lab = ["accept"] * len(u); lab[worst] = "strong_dislike"; return lab


def graded(u, bar, worst, rng, delta=0.02):
    """Every diner below the bar objects; mildly (reject) if within delta of
    the bar, strongly otherwise."""
    return ["accept" if x >= bar else ("reject" if bar - x <= delta else "strong_dislike") for x in u]


def truth(g, kind, rng):
    S = g["S"]
    if kind == "proxy is exact":
        return S
    if kind.startswith("noise"):
        sigma = float(kind.split("=")[1])
        return S + rng.normal(0, sigma, S.shape)
    if kind == "order-aware (top-3 dishes)":
        return g["D3"]
    raise ValueError(kind)


ARMS = [
    ("one-shot Average", None),
    ("one-shot Fair (lambda=5)", None),
    ("one-shot Least-Misery", None),
    ("mediator, worst-off objects (paper rule)", dict(react_fn=worst_only)),
    ("mediator, graded multi-diner reactions", dict(react_fn=graded)),
    ("  + LLM parser replay", dict(react_fn=graded, parser=parser_replay)),
    ("  + LLM parser replay + memory", dict(react_fn=graded, parser=parser_replay, memory=True)),
    ("  + LLM parser replay + veto token", dict(react_fn=graded, parser=parser_replay,
                                                veto_remove=True, veto_grad=True)),
]
rows5 = []
for city in CITIES:
    for kind in ["proxy is exact", "noise sd=0.01", "noise sd=0.02", "order-aware (top-3 dishes)"]:
        rng0 = np.random.default_rng(11)
        Us = [truth(g, kind, rng0) for g in D[city]]
        Ss = [g["S"] for g in D[city]]
        bars = [U.min(1).max() - MARGIN for U in Us]
        for label, kw in ARMS:
            if kw is None:
                rule = {"one-shot Average": ("average", 0.5), "one-shot Fair (lambda=5)": ("fair", 5),
                        "one-shot Least-Misery": ("least_misery", 0.5)}[label]
                res = []
                for S, U, b in zip(Ss, Us, bars):
                    p = pick(S, rule[0], rule[1])
                    res.append(dict(converged=U[p].min() >= b, rounds=1, pick=p, reason="oneshot"))
            else:
                rng = np.random.default_rng(23)
                res = [mediate(S, U=U, bar=b, rng=rng, **kw) for S, U, b in zip(Ss, Us, bars)]
            rows5.append(summarise(res, Us, bars, label, city, dict(truth=kind)))
            print(rows5[-1])
write("e5_proxy_gap.csv", rows5)
