"""Aggregate the Houston judge log into Tables 7 and 8 (same logic as 07-judge cells 17-23).
The judge ensemble is gpt-4o-mini, qwen3.8-27b, allam-2-7b; the partial gpt-oss-20b
rows (stopped by free-tier caps) are excluded."""
import os, sys, json, itertools
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); os.chdir(R)
import pandas as pd, nbio
LOG = "runs/judge_v1_seed7_houston-tx-foodcom.jsonl"
OUT = "figdata/ubereats-houston-tx-foodcom"
MODELS = ["gpt-4o-mini", "qwen/qwen3.8-27b", "allam-2-7b"]
SELECTORS = ["average", "least_misery", "most_pleasure", "fair"]
ORDER_TOLERANCE, UNSTABLE_SD = 1.0, 1.0
df = pd.DataFrame([json.loads(l) for l in open(LOG)])
df = df[df.model.isin(MODELS)].drop_duplicates(["model","group_id","selector","repeat","reverse_order"])
print("calls per model (all):", df.model.value_counts().to_dict())
# Balanced design: keep only groups every judge rated in full (40 calls each), so
# every model sees every item. Qwen stopped early on Groq's free-tier cap.
cnt = df[df.ok].groupby(["group_id","model"]).size().unstack(fill_value=0)
FULL = cnt.index[(cnt.reindex(columns=MODELS, fill_value=0) == 40).all(axis=1)]
df = df[df.group_id.isin(FULL)]
print(f"balanced groups: {len(FULL)}; calls per model:", df.model.value_counts().to_dict())
ok = df[df.ok]
print(f"drop rate {1-len(ok)/len(df):.3%} ({len(df)-len(ok)} of {len(df)})")
rank = ok.groupby(["model","selector"]).fairness.mean().unstack()[SELECTORS]
print(rank.round(3).to_string())
rhos = {}
for a, b in itertools.combinations(MODELS, 2):
    rhos[(a,b)] = nbio.spearman(rank.loc[a].tolist(), rank.loc[b].tolist()); print(f"rho {a} {b} {rhos[(a,b)]:+.3f}")
print(f"mean rho {sum(rhos.values())/3:+.3f}")
o = ok.groupby(["group_id","selector","reverse_order"])[["fairness","satisfaction","consensus"]].mean()
gaps = (o.xs(False, level="reverse_order") - o.xs(True, level="reverse_order")).abs()
flag = (gaps > ORDER_TOLERANCE).any(axis=1); DISC = set(gaps.index[flag])
print("order gaps:", gaps.mean().round(3).to_dict(), f"discarded {flag.sum()} of {len(gaps)}")
sd = ok.groupby(["group_id","selector","model"])[["fairness","satisfaction","consensus"]].std()
uns = (sd > UNSTABLE_SD).any(axis=1)
print("mean sd:", sd.mean().round(3).to_dict(), f"unstable {uns.sum()} of {len(sd)} ({uns.mean():.1%})")
keep = ok[~ok.set_index(["group_id","selector"]).index.isin(DISC)]
out = []
for sel in SELECTORS:
    s = keep[keep.selector == sel]; pi = s.groupby(["group_id","model"])[["fairness","satisfaction","consensus"]].mean().groupby("group_id").mean()
    row = {"selector": sel, "n_items": len(pi), "n_calls": len(s)}
    for d in ("fairness","satisfaction","consensus"):
        m, ci, _ = nbio.mean_ci(pi[d].tolist()); row[f"{d}_mean"], row[f"{d}_ci95"] = round(m,3), round(ci,3)
    row["drop_rate"] = round(1 - len(s)/max(1, len(df[df.selector == sel])), 4); out.append(row)
t7 = pd.DataFrame(out).sort_values("fairness_mean", ascending=False); t7.to_csv(f"{OUT}/table7_judge.csv", index=False)
print(t7.to_string())
pm = keep.groupby(["model","selector"])[["fairness","satisfaction","consensus"]].mean().round(3).reset_index()
pm.to_csv(f"{OUT}/table7_judge_per_model.csv", index=False)
# arithmetic metrics of the same picks, for comparison
ar = df.drop_duplicates(["group_id","selector"]).groupby("selector")[["min_sat","jain","mean_sat"]].mean().round(4)
print(ar.to_string()); ar.to_csv(f"{OUT}/table7_judge_arith.csv")
claims = pd.read_csv("data/judge/human_claims.csv"); t7i = t7.set_index("selector")
# "no_difference" claims agree when the two fairness CIs overlap; directional claims
# need the judge's means to point the same way and be separated.
r8 = []
for _, c in claims.iterrows():
    a, b = t7i.loc[c.test_selector_a], t7i.loc[c.test_selector_b]
    gap = abs(a.fairness_mean - b.fairness_mean)
    sep = gap > a.fairness_ci95 + b.fairness_ci95
    jd = "no_difference" if not sep else ("a_fairer" if a.fairness_mean > b.fairness_mean else "b_fairer")
    r8.append({"claim_id": c.claim_id, "source_paper": c.source_paper,
               "comparison": f"{c.test_selector_a} vs {c.test_selector_b}",
               "human_direction": c.human_direction, "judge_direction": jd,
               "agrees": bool(jd == c.human_direction), "margin": round(gap, 3),
               "page_cite": c.page_cite, "verified": c.verified})
t8 = pd.DataFrame(r8); t8.to_csv(f"{OUT}/table8_judge_validation.csv", index=False); print(t8.to_string())
