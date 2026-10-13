"""The mediator's pass mark (task doc, Step 1): relative to the group, not fixed.

    python tests/test_mediation.py      # or: pytest tests/

Uses only a hand-written satisfaction matrix, so it needs no Kaggle data.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402

# Everyone scores high, as real Food.com diners do. The best worst-off option
# is C (0.926), so the pass mark is 0.916 and A, the first pick, falls short.
USERS = ["ann", "ben", "cara"]
S = {
    "A": {"ann": 0.99, "ben": 0.97, "cara": 0.886},
    "B": {"ann": 0.95, "ben": 0.94, "cara": 0.905},
    "C": {"ann": 0.93, "ben": 0.93, "cara": 0.926},
}


def test_consensus_bar_is_best_floor_minus_margin():
    assert abs(nbio.consensus_bar(S, USERS) - (0.926 - nbio.DEFAULT_CONSENSUS_MARGIN)) < 1e-12
    assert abs(nbio.consensus_bar(S, USERS, 0.02) - 0.906) < 1e-12


def test_relative_bar_makes_the_loop_negotiate():
    old = nbio.mediate(S, USERS, bar=0.45)
    assert old["reason"] == "consensus" and old["rounds"] == 1   # the bug: no negotiation
    new = nbio.mediate(S, USERS)
    assert new["reason"] == "consensus" and new["rounds"] > 1
    assert new["worst_sat"] >= nbio.consensus_bar(S, USERS)
    assert nbio.solvable(S, USERS)                               # every group, by construction


def test_consensus_reached_requires_a_bar():
    assert inspect.signature(nbio.consensus_reached).parameters["bar"].default is inspect.Parameter.empty


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
