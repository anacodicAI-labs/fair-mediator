"""The shuffled-menu control is the same on every machine (task doc, Step 2).

    python tests/test_shuffle.py      # or: pytest tests/

Uses a hand-written catalog, so it needs no Kaggle data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402


def _catalog():
    cat = []
    for r in range(6):
        menu = [{"name": f"dish {r}-{k}", "taste": [r / 10, k / 10, 0.1, 0.0, 0.2, 0.3]}
                for k in range(r + 2)]
        menu.append({"name": "house special", "taste": [0.5, 0.5, 0.0, 0.0, 0.1, r / 10]})  # a tie on name
        cat.append({"option_id": f"r{r}", "name": f"R{r}", "menu": menu})
    return cat


def _deal(catalog):
    return {o["option_id"]: [(d["name"], tuple(d["taste"])) for d in o["menu"]]
            for o in nbio.shuffle_menus(catalog, seed=7)}


def test_shuffle_ignores_input_order():
    cat = _catalog()
    scrambled = [{**o, "menu": list(reversed(o["menu"]))} for o in reversed(cat)]
    assert _deal(cat) == _deal(scrambled)


def test_shuffle_keeps_restaurants_and_menu_sizes():
    cat = _catalog()
    out = nbio.shuffle_menus(cat, seed=7)
    assert [o["option_id"] for o in out] == [o["option_id"] for o in cat]
    assert [len(o["menu"]) for o in out] == [len(o["menu"]) for o in cat]
    every = lambda c: sorted((d["name"], tuple(d["taste"])) for o in c for d in o["menu"])
    assert every(out) == every(cat)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
