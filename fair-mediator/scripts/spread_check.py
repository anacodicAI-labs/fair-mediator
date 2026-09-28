"""Task 6 — check the spread before anything else (task doc §7.3).

    python scripts/spread_check.py                 # today's catalog + every built city
    python scripts/spread_check.py "Houston, TX"   # just one city

spread = mean over the 6 tastes of the std of restaurant taste vectors. Also
printed: how far apart restaurants sit (max per-dim range), taste coverage,
and the spread of pairwise cosine between restaurants, since cosine is what
satisfaction actually uses (Equation 2).
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402


def report(label: str, catalog) -> dict:
    sp = nbio.taste_spread(catalog)
    vecs = [nbio.resolve_taste_vector(o["taste"]) for o in catalog]
    ranges = {d: max(v[i] for v in vecs) - min(v[i] for v in vecs) for i, d in enumerate(nbio.TASTE_DIMS)}
    cos = [nbio.calculate_cosine_similarity(a, b) for a, b in combinations(vecs, 2)]
    cos_min = min(cos) if cos else 1.0
    cos_mean = sum(cos) / len(cos) if cos else 1.0
    print(f"\n{label}  ({sp['n_restaurants']} restaurants, taste coverage {nbio.taste_coverage(catalog):.1%})")
    print(f"  spread (mean per-dim std)   {sp['spread']:.4f}")
    print("  per-dim std    " + "  ".join(f"{d} {v:.3f}" for d, v in sp["per_dim_std"].items()))
    print("  per-dim range  " + "  ".join(f"{d} {v:.3f}" for d, v in ranges.items()))
    print(f"  restaurant-vs-restaurant cosine: mean {cos_mean:.3f}, min {cos_min:.3f}  "
          f"(1.0 = identical taste direction)")
    return sp


def main():
    base = report("potluck (today: data/Menu, 5 restaurants)", nbio._load_catalog_potluck())
    if len(sys.argv) > 1:
        cities = sys.argv[1:]
    else:
        cities = [json.loads(p.read_text())["meta"]["city"]
                  for p in sorted(nbio.EXTERNAL_CACHE.glob("catalog_ubereats_*.json"))]
    if not cities:
        print("\nno Uber Eats catalog built yet — run scripts/build_ubereats_catalog.py --city ...")
    for c in cities:
        sp = report(f"ubereats: {c}", nbio.load_catalog_ubereats(c))
        ratio = sp["spread"] / base["spread"] if base["spread"] else float("inf")
        print(f"  -> {ratio:.1f}x today's spread")


if __name__ == "__main__":
    main()
