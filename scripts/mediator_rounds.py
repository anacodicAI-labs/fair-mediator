"""Rounds to agreement, reported both ways, for every margin and city.

rounds_agree_only: mean rounds over the groups that reach agreement.
rounds_capped:     mean rounds with groups that never agree counted at the cap.
Uses nbio.mediate with the group-relative bar (nbio.consensus_bar), the same
200 seeded groups and Fair(0.5) feasible set as the harness.
Writes figdata/ubereats-foodcom/mediator_rounds.csv.
    FM_PERSONAS=foodcom FM_DATASET=ubereats python scripts/mediator_rounds.py
"""
from __future__ import annotations
import csv, os, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402


def main():
    os.environ.setdefault("FM_PERSONAS", "foodcom")
    os.environ.setdefault("FM_DATASET", "ubereats")
    rows = []
    for city in ("Houston, TX", "Seattle, WA"):
        os.environ["FM_CITY"] = city
        cat = nbio.load_catalog()
        groups = nbio.generate_groups(200, seed=7)
        pre = []
        for g in groups:
            r = nbio.run_group(g, cat, "fair", 0.5)
            if r:
                pre.append(r)
        for margin in (0.005, 0.01, 0.02):
            for arm, veto in (("one-shot", None), ("mediator", False), ("mediator + veto", True)):
                agree_rounds, capped = [], []
                for r in pre:
                    S, ids = r["_S"], r["_member_ids"]
                    bar = nbio.consensus_bar(S, ids, margin)
                    if veto is None:
                        ok = min(S[r["pick_id"]][m] for m in ids) >= bar
                        rounds = 1
                    else:
                        o = nbio.mediate(S, ids, bar=bar, use_veto=veto)
                        ok, rounds = o["converged"], o["rounds"]
                    if ok:
                        agree_rounds.append(rounds)
                    capped.append(rounds if ok else nbio.DEFAULT_MAX_ROUNDS)
                rows.append({"city": city, "margin": margin, "arm": arm, "n_groups": len(pre),
                             "n_agree": len(agree_rounds), "pct_agree": round(100 * len(agree_rounds) / len(pre), 1),
                             "rounds_agree_only": round(st.mean(agree_rounds), 3) if agree_rounds else "",
                             "rounds_capped": round(st.mean(capped), 3)})
                print(rows[-1], flush=True)
    out = ROOT / "figdata" / "ubereats-foodcom" / "mediator_rounds.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
