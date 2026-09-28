"""Hand-check sheets and their scores — the two measured components of the
food catalog (task doc: "spot-check 100 dishes by hand", "hand-check 100
matches and report the accuracy").

    python scripts/handcheck.py bitter-sheet            # writes validation/bitter_spotcheck.csv
    python scripts/handcheck.py score-bitter            # after filling looks_bitter (y/n)
    python scripts/handcheck.py score-match "Houston, TX"   # after filling correct (y/p/n)

The match sheet itself is written by build_ubereats_catalog.py.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402

VALIDATION = ROOT / "validation"
BITTER_SHEET = VALIDATION / "bitter_spotcheck.csv"
BITTER_I = nbio.TASTE_DIMS.index("bitter")


def _wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for k successes in n — sane at small n and near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _fmt(k, n):
    lo, hi = _wilson(k, n)
    return f"{k}/{n} = {k / n:.0%}  (95% CI {lo:.0%}-{hi:.0%})" if n else "n/a"


def bitter_sheet(seed: int = 7, n: int = 100) -> None:
    """50 recipes the list scores bitter, 50 it scores not bitter, seeded.
    Stratified so both error types get measured: false bitter (precision)
    and missed bitter (false-negative rate)."""
    if BITTER_SHEET.exists():
        sys.exit(f"{BITTER_SHEET.relative_to(ROOT)} exists — not overwriting labels.")
    names = {}
    with open(nbio.flavour_csv_path(), newline="") as f:
        csv.field_size_limit(1 << 24)
        for row in csv.DictReader(f):
            names[int(row["id"])] = row["name"]
    ids = sorted(nbio._recipes())
    pos = [i for i in ids if nbio.recipe_taste(i)[BITTER_I] > 0]
    neg = [i for i in ids if nbio.recipe_taste(i)[BITTER_I] == 0]
    rng = random.Random(seed)
    picked = rng.sample(pos, n // 2) + rng.sample(neg, n - n // 2)
    rng.shuffle(picked)                       # labeller should not see the stratum order
    VALIDATION.mkdir(exist_ok=True)
    with open(BITTER_SHEET, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recipe_id", "recipe_name", "ingredients", "bitter_score",
                    "bitter_ingredients", "looks_bitter", "notes"])
        for i in picked:
            ings = nbio.recipe_ingredients(i)
            hits = [g for g in ings if nbio.ingredient_bitter(g)]
            w.writerow([i, names.get(i, ""), "; ".join(ings), nbio.recipe_taste(i)[BITTER_I],
                        "; ".join(hits), "", ""])
    print(f"-> {BITTER_SHEET.relative_to(ROOT)}  ({len(pos):,} of {len(ids):,} recipes score bitter > 0; "
          f"sheet has {n // 2} of each)")
    print("   fill looks_bitter with y (a noticeably bitter dish) or n, then run score-bitter")


def score_bitter() -> None:
    rows = list(csv.DictReader(open(BITTER_SHEET)))
    lab = [r for r in rows if r["looks_bitter"].strip().lower() in ("y", "n")]
    if not lab:
        sys.exit("no rows labelled yet (looks_bitter = y/n).")
    pos = [r for r in lab if float(r["bitter_score"]) > 0]
    neg = [r for r in lab if float(r["bitter_score"]) == 0]
    tp = sum(r["looks_bitter"].strip().lower() == "y" for r in pos)
    fn = sum(r["looks_bitter"].strip().lower() == "y" for r in neg)
    print(f"labelled {len(lab)}/{len(rows)}")
    print(f"  precision   (scored bitter AND looks bitter)   {_fmt(tp, len(pos))}")
    print(f"  missed rate (scored 0 BUT looks bitter)        {_fmt(fn, len(neg))}")


def score_match(city: str) -> None:
    path = VALIDATION / f"match_check_{nbio.city_slug(city)}.csv"
    rows = list(csv.DictReader(open(path)))
    lab = [r for r in rows if r["correct"].strip().lower() in ("y", "p", "n")]
    if not lab:
        sys.exit(f"no rows labelled yet in {path.name} (correct = y/p/n).")
    print(f"{city}: labelled {len(lab)}/{len(rows)}")
    y = sum(r["correct"].strip().lower() == "y" for r in lab)
    yp = sum(r["correct"].strip().lower() in ("y", "p") for r in lab)
    print(f"  exact dish          {_fmt(y, len(lab))}")
    print(f"  exact or close      {_fmt(yp, len(lab))}")
    # Accuracy above each threshold: what raising the cut-off would buy.
    print("  threshold  kept  exact-or-close among kept")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        kept = [r for r in lab if float(r["match_score"]) >= t]
        k = sum(r["correct"].strip().lower() in ("y", "p") for r in kept)
        print(f"    {t:.1f}     {len(kept):>4}  {_fmt(k, len(kept))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bitter-sheet")
    sub.add_parser("score-bitter")
    sm = sub.add_parser("score-match")
    sm.add_argument("city")
    a = ap.parse_args()
    {"bitter-sheet": bitter_sheet, "score-bitter": score_bitter}.get(a.cmd, lambda: score_match(a.city))()


if __name__ == "__main__":
    main()
