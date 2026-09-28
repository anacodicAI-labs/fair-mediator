"""The loader contract (task doc §7.2): what every catalog must satisfy before
the harness is allowed to score it.

    python tests/test_loader_contract.py      # or: pytest tests/

Checks every Uber Eats catalog built in data/external_cache/ (or only FM_CITY
when set), plus recipe_taste itself. Anything needing Kaggle data that is not
on this machine is skipped with a message, not failed.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbio  # noqa: E402

MIN_COVERAGE = 0.90


try:                                   # a real skip under pytest, a message otherwise
    import pytest
    Skip = pytest.skip.Exception
except ImportError:
    class Skip(Exception):
        pass


def _finite6(vec) -> bool:
    return (isinstance(vec, (list, tuple)) and len(vec) == 6
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in vec))


def check_catalog(catalog, label: str) -> None:
    assert catalog, f"{label}: catalog is empty"
    ids = [o["option_id"] for o in catalog]
    assert len(ids) == len(set(ids)), f"{label}: duplicate option_id"
    for o in catalog:
        where = f"{label}/{o['option_id']}"
        assert o.get("name"), f"{where}: no name"
        assert _finite6(o["taste"]), f"{where}: restaurant taste is not 6 finite numbers: {o['taste']}"
        assert o.get("menu"), f"{where}: menu is empty"
        for d in o["menu"]:
            assert d.get("name"), f"{where}: dish with no name"
            assert _finite6(d["taste"]), f"{where}/{d.get('name')}: dish taste {d.get('taste')}"
            assert "allergens" in d, f"{where}/{d['name']}: no allergens field"
            a = d["allergens"]
            assert a is None or set(a) <= set(nbio.FDA_ALLERGEN_KEYWORDS), \
                f"{where}/{d['name']}: unknown allergen in {a}"
            # unknown ingredients must stay unknown allergens (fail closed)
            if d.get("ingredients") is None:
                assert a is None, f"{where}/{d['name']}: allergens without ingredients"
    cov = nbio.taste_coverage(catalog)
    assert cov >= MIN_COVERAGE, f"{label}: taste coverage {cov:.1%} < {MIN_COVERAGE:.0%}"


def test_recipe_taste():
    try:
        nbio._recipes()
    except FileNotFoundError as e:
        raise Skip(str(e).splitlines()[0])
    rid = next(iter(nbio._recipes()))
    v = nbio.recipe_taste(rid)
    assert _finite6(v), v
    assert all(0.0 <= x <= 1.0 for x in v), v
    assert nbio.recipe_taste(-1) == [0.0] * 6
    assert nbio.recipe_taste("not an id") == [0.0] * 6
    assert nbio.recipe_ingredients(-1) is None


def test_bitter_list():
    assert nbio.ingredient_bitter("strong brewed coffee") == 0.9
    assert nbio.ingredient_bitter("coffee creamer") == 0.0      # exclusion beats 'coffee'
    assert nbio.ingredient_bitter("white chocolate chips") == 0.0
    assert nbio.ingredient_bitter("sugar") is None


def test_allergen_tags():
    assert nbio.dish_allergens(None) is None
    assert nbio.dish_allergens([]) == []
    assert nbio.dish_allergens(["coconut milk", "eggplant", "water chestnuts"]) == []
    assert nbio.dish_allergens(["soy sauce"]) == ["soy", "wheat"]
    assert nbio.dish_allergens(["peanut butter", "shrimp", "eggs"]) == ["egg", "peanuts", "shellfish"]


def test_ubereats_catalogs():
    city = os.environ.get("FM_CITY")
    paths = ([nbio.ubereats_cache_path(city)] if city
             else sorted(nbio.EXTERNAL_CACHE.glob("catalog_ubereats_*.json")))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise Skip("no Uber Eats catalog built yet (scripts/build_ubereats_catalog.py)")
    import json
    for p in paths:
        c = json.loads(p.read_text())["meta"]["city"]
        check_catalog(nbio.load_catalog_ubereats(c), c)


def test_default_catalog_shape():
    """The Potluck catalog predates the contract: shape only, no coverage
    floor (it is 47%, one reason the old results were null)."""
    for o in nbio.load_catalog():
        assert _finite6(o["taste"]) and o["menu"]
        assert all(_finite6(d["taste"]) for d in o["menu"])


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as s:
            print(f"SKIP  {name}: {s}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
