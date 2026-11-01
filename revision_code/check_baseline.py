import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lib import *

D = data()
for city in CITIES + [c + " shuffled" for c in CITIES]:
    Ms = [g["S"] for g in D[city]]
    for lam in (0.5, 5):
        r = c1_summary(Ms, lam)
        print(f"{city:22s} lam={lam:<4} gain={r['gain']:.4f} [{r['gain_lo']:.4f},{r['gain_hi']:.4f}] "
              f"ach={r['achievable']:.4f} pct={r['pct']:.1f} [{r['pct_lo']:.1f},{r['pct_hi']:.1f}] "
              f"cost={r['cost']:.4f} lm={r['cost_lm']:.4f} moved={r['moved']:.1f}")
for city in CITIES:
    for name, kw in [("mediator", {}), ("veto", dict(veto_remove=True, veto_grad=True))]:
        res = [mediate(g["S"], **kw) for g in D[city]]
        conv = [r for r in res if r["converged"]]
        print(city, name, len(conv), np.mean([r["rounds"] for r in conv]),
              np.mean([r["rounds"] for r in res]))
    one = sum(1 for g in D[city] if g["S"][pick(g["S"], "fair", 0.5)].min() >= g["S"].min(1).max() - 0.01)
    print(city, "one-shot", one)
