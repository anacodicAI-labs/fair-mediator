"""nbio — shared plumbing for the Fair Mediator notebooks.

Two jobs, same as the cookbook's nbio.py:

1. DISPLAY  — io() bands, tables, cost metering, run artifacts under runs/.
2. DOMAIN LOGIC that is genuinely shared across notebooks — cosine similarity,
   the four aggregation selectors, and the fairness metrics — so a function is
   defined exactly ONCE and every notebook imports it rather than redefining it.
   This is the one file in the repository allowed to hold logic; everything
   else is a notebook.

Each function below is either original to this paper or ported from a named
contributor's code, with the source commit/file cited in its docstring.
Attribution travels with the code.
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = HERE / "runs"
FIGDATA = HERE / "figdata"

TASTE_DIMS = ["sweet", "salty", "sour", "bitter", "umami", "spicy"]
NEUTRAL_TASTE_VECTOR = [0.5] * 6


# ── setup ────────────────────────────────────────────────────────────────────

def bootstrap() -> Path:
    """Confirm the repo layout and report what is on disk. Call first in every notebook."""
    for d in (DATA, RUNS, FIGDATA):
        d.mkdir(parents=True, exist_ok=True)
    menus = sorted((DATA / "Menu").glob("*.json")) if (DATA / "Menu").exists() else []
    print(f"repo     : {HERE}")
    print(f"data     : {DATA.relative_to(HERE)}  "
          f"(ingredient-flavor.csv: {'found' if (DATA/'ingredient-flavor.csv').exists() else 'MISSING'}, "
          f"{len(menus)} menu files)")
    return HERE


# ── the chain — notebooks hand off through files, not shared kernel state ────

def save_run(stage: str, obj: Any, run_id: str = "demo") -> Path:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stage}.json"
    p.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    print(f"-> wrote {p.relative_to(HERE)}  ({p.stat().st_size:,} bytes)")
    return p


def load_run(stage: str, run_id: str = "demo") -> Any:
    p = RUNS / run_id / f"{stage}.json"
    if not p.exists():
        have = sorted(x.name for x in (RUNS / run_id).glob("*.json")) if (RUNS / run_id).exists() else []
        raise FileNotFoundError(
            f"{p.name} not found for run {run_id!r}.\n"
            f"  present: {have or '(nothing yet)'}\n"
            f"  run the notebook that writes {stage}.json first."
        )
    return json.loads(p.read_text())


def save_figdata(name: str, rows: list[dict]) -> Path:
    """Write one CSV that a paper table/figure reads. This is the ONLY path a
    reported number may travel — see the repository's non-negotiables.

    FM_FIGDATA (a folder, relative to the repo) redirects the output, so runs
    on different datasets do not overwrite each other's CSVs."""
    import csv
    p = HERE / os.environ["FM_FIGDATA"] / name if os.environ.get("FM_FIGDATA") else FIGDATA / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"-> wrote {p.relative_to(HERE)}  ({len(rows)} rows)")
    return p


# ── seeing inside — the IO band ───────────────────────────────────────────────

_W = 74


def io(step: str | int, node: str, before: dict, after: dict, *, kind: str = "py",
       reads: Iterable[str] = ()) -> dict:
    """Print what one step read, added and changed — computed from a state diff
    so it cannot drift from the code, then return `after` so it can be chained."""
    new = sorted(set(after) - set(before))
    changed = sorted(k for k in before if k in after and _repr(before[k]) != _repr(after[k]))
    gone = sorted(set(before) - set(after))
    print("+- STEP " + f"{step}  {node}".ljust(_W - 14) + f"[{kind}]")
    if reads:
        print("| IN      : " + _wrap(", ".join(reads)))
    print("| NEW     : " + (_wrap(", ".join(new)) if new else "-"))
    print("| CHANGED : " + (_wrap(", ".join(changed)) if changed else "-"))
    if gone:
        print("| REMOVED : " + _wrap(", ".join(gone)))
    print("+" + "-" * (_W - 1))
    return after


def _repr(v: Any) -> str:
    try:
        return json.dumps(v, sort_keys=True, default=str)
    except Exception:
        return repr(v)


def _wrap(s: str, indent: str = "|           ") -> str:
    lines = textwrap.wrap(s, width=_W - 12) or [""]
    return ("\n" + indent).join(lines)


def rule(title: str) -> None:
    print("=" * _W); print(title); print("=" * _W)


def taste(vec: Sequence[float], *, width: int = 12) -> str:
    """One taste vector as a labelled bar strip."""
    parts = []
    for name, v in zip(TASTE_DIMS, vec):
        filled = int(round(max(0.0, min(1.0, v)) * width))
        parts.append(f"{name[:5]:<5} {'#' * filled}{'.' * (width - filled)} {v:.2f}")
    return "\n".join("    " + p for p in parts)


def table(rows: list[dict], cols: list[str] | None = None, width: int = 22) -> None:
    if not rows:
        print("(no rows)"); return
    cols = cols or list(rows[0])
    def cell(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    widths = {c: min(width, max(len(str(c)), *(len(cell(r.get(c, ""))) for r in rows))) for c in cols}
    print("  ".join(str(c)[:widths[c]].ljust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        print("  ".join(cell(r.get(c, ""))[:widths[c]].ljust(widths[c]) for c in cols))


def matrix(S: Dict[str, Dict[str, float]], member_ids: Sequence[str], *,
           names: Dict[str, str] | None = None, labels: Dict[str, str] | None = None,
           limit: int = 12, highlight: str | None = None) -> None:
    """Print a satisfaction matrix, options x members."""
    if not S:
        print("(empty — F is empty)"); return
    names = names or {m: m for m in member_ids}
    labels = labels or {}
    head = "option".ljust(30) + "".join(names[i][:9].rjust(10) for i in member_ids) + "      min      mean"
    print(head); print("-" * len(head))
    for oid, row in list(S.items())[:limit]:
        vals = [row.get(i, 0.0) for i in member_ids]
        mark = " <-- pick" if oid == highlight else ""
        label = labels.get(oid, oid)
        label = (label[:27] + "...") if len(label) > 30 else label
        print(label.ljust(30) + "".join(f"{v:10.3f}" for v in vals)
              + f"{min(vals):9.3f}{sum(vals)/len(vals):9.3f}{mark}")
    if len(S) > limit:
        print(f"... [{len(S) - limit} more options]")


# ── cost meter (only 05-mediate/01 calls a model) ────────────────────────────

class BudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, budget_usd: float = 0.50):
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self.calls: list[dict] = []

    def record(self, label: str, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        self.calls.append({"label": label, "cost_usd": cost_usd, "running_total": self.spent_usd})
        if self.spent_usd > self.budget_usd:
            raise BudgetExceeded(f"${self.spent_usd:.4f} > ceiling ${self.budget_usd:.2f}")

    def report(self) -> str:
        return f"spent ${self.spent_usd:.4f} of ${self.budget_usd:.2f} ceiling across {len(self.calls)} call(s)"


def cost_meter(budget_usd: float = 0.50) -> Meter:
    print(f"metering on, ceiling ${budget_usd:.2f}")
    return Meter(budget_usd)


# ═════════════════════════════════════════════════════════════════════════════
# DOMAIN LOGIC — shared across notebooks, defined here exactly once
# ═════════════════════════════════════════════════════════════════════════════

# ── taste similarity ──────────────────────────────────────────────────────────
#
# Ported from services/taste_service.py (potluck, main). `resolve_taste_vector`
# and the neutral-fallback constant are Vincent's (wangvincentx-cmd, commit
# 86124d4, 2026-08-14): "Add neutral default vectors, remove persona
# duplicates, and make fairness lambda adjustable per session".

def calculate_cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """cosine(v1, v2). Potluck's services/integrations/embeddings.py, ported
    without the numpy/sklearn dependency so this notebook needs no install."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def resolve_taste_vector(vec: Optional[Sequence[float]]) -> List[float]:
    """Return `vec` if it has any signal, else the neutral fallback.

    Vincent, potluck main, services/taste_service.py:27, commit 86124d4. Before
    this fix a member with no taste data scored 0.0 against everything rather
    than the neutral midpoint, which silently penalised new/anonymous members.
    """
    if vec and any(v != 0.0 for v in vec):
        return list(vec)
    return list(NEUTRAL_TASTE_VECTOR)


def taste_similarity(user_vec: Sequence[float], item_vec: Sequence[float]) -> float:
    return calculate_cosine_similarity(resolve_taste_vector(user_vec), resolve_taste_vector(item_vec))


# ── C1: the four aggregation selectors ────────────────────────────────────────
#
# Ported verbatim (renamed for this repo's notation) from Vincent's
# backend/services/fairness_service.py (potluck main, commit 3c32c76,
# "Add per-member F0 taste vectors and C1 fairness aggregation selectors",
# 2026-08-02). Docstrings are his.

AggregationMethod = Literal["average", "least_misery", "most_pleasure", "fair"]
DEFAULT_LAMBDA = 0.5


def average_score(scores: Sequence[float]) -> float:
    """(1/n) * sum(s_i) — highest average happiness, can leave individuals behind."""
    return sum(scores) / len(scores) if scores else 0.0


def least_misery_score(scores: Sequence[float]) -> float:
    """min(s_i) — protects the worst-off member, but drags the group to the safest option."""
    return min(scores) if scores else 0.0


def most_pleasure_score(scores: Sequence[float]) -> float:
    """max(s_i) — favors whoever is most excited, ignores everyone else."""
    return max(scores) if scores else 0.0


def variance(scores: Sequence[float]) -> float:
    """Population variance of the per-member scores."""
    if not scores:
        return 0.0
    m = average_score(scores)
    return sum((s - m) ** 2 for s in scores) / len(scores)


def fairness_score(scores: Sequence[float], lam: float = DEFAULT_LAMBDA) -> float:
    """(1/n)*sum(s_i) - lambda*Var(s_i) — the C1 objective. lambda=0 is average;
    raising it trades mean satisfaction for a flatter spread."""
    return average_score(scores) - lam * variance(scores) if scores else 0.0


_SELECTORS = {
    "average": lambda scores, lam: average_score(scores),
    "least_misery": lambda scores, lam: least_misery_score(scores),
    "most_pleasure": lambda scores, lam: most_pleasure_score(scores),
    "fair": lambda scores, lam: fairness_score(scores, lam),
}


def aggregate(scores: Sequence[float], method: AggregationMethod = "fair",
              lam: float = DEFAULT_LAMBDA) -> float:
    """Dispatch to one of the four C1 selectors."""
    if method not in _SELECTORS:
        raise ValueError(f"Unknown aggregation method: {method!r}. Expected one of {sorted(_SELECTORS)}.")
    return _SELECTORS[method](scores, lam)


def rank_candidates(candidate_taste_vectors: Dict[str, Sequence[float]],
                     member_taste_vectors: Sequence[Sequence[float]],
                     method: AggregationMethod = "fair",
                     lam: float = DEFAULT_LAMBDA) -> List[Tuple[str, float]]:
    """Rank candidates (already restricted to the safe set F) by aggregated
    group score. Vincent, fairness_service.py."""
    ranked = []
    for cid, vec in candidate_taste_vectors.items():
        scores = [taste_similarity(mv, vec) for mv in member_taste_vectors]
        ranked.append((cid, aggregate(scores, method, lam)))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


# ── Stage 2: the feasible set F ───────────────────────────────────────────────
#
# Original to this paper (Rashan), ported from potluck's
# backend/services/aggregation.py.

NON_VEGAN_WORDS = ["paneer", "cheese", "butter", "ghee", "cream", "milk", "malai",
                   "curd", "yogurt", "yoghurt", "dahi", "khoya", "mawa", "honey",
                   "kulfi", "lassi"]
NON_VEG_WORDS = ["chicken", "murgh", "murg", "mutton", "gosht", "lamb", "beef",
                 "pork", "fish", "pomfret", "prawn", "prawns", "shrimp", "jheenga",
                 "crab", "lobster", "meat", "bacon", "sausage", "ham", "turkey",
                 "duck", "egg", "eggs", "anda", "omelette", "omelet", "seafood",
                 "salmon", "tuna", "nahari", "nihari", "raan", "kheema", "keema"]


def is_nonveg_text(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in NON_VEG_WORDS)


def dish_is_safe(dish_name: str, diet: str, allergies: Sequence[str]) -> bool:
    """True if one dish breaks neither the member's diet nor their allergy set.

    Known limitation, measured and reported in this paper: this checks the
    dish NAME only. A battered, fried dish with no allergen word in its name
    passes for a gluten allergy. See 02-safety/01 for the demonstration."""
    text = (dish_name or "").lower()
    for allergen in allergies or []:
        if allergen and allergen.lower() in text:
            return False
    diet = (diet or "mix").lower()
    if diet in {"veg", "vegetarian"}:
        return not is_nonveg_text(text)
    if diet == "vegan":
        return not is_nonveg_text(text) and not any(w in text for w in NON_VEGAN_WORDS)
    return True


def safe_dishes_for(member: Dict, menu: Sequence[Dict]) -> List[Dict]:
    return [d for d in menu if dish_is_safe(d.get("name", ""), member.get("diet", "mix"),
                                             member.get("allergies", []))]


def build_feasible_set(catalog: Sequence[Dict], members: Sequence[Dict]) -> List[Dict]:
    """Keep only the options where EVERY member has >=1 safe dish. This is the
    property that makes a constraint violation structurally impossible
    downstream, rather than prompt-dependent — see 05-mediate for the contrast
    with the LLM chat path."""
    feasible = []
    for option in catalog:
        menu = option.get("menu") or []
        safe_by_member = {}
        for member in members:
            safe = safe_dishes_for(member, menu)
            if not safe:
                break
            safe_by_member[member["member_id"]] = safe
        else:
            feasible.append({**option, "safe_dishes": safe_by_member})
    return feasible


# ── Stage 5: fairness metrics ─────────────────────────────────────────────────
#
# Original to this paper (Rashan), ported from potluck's
# backend/services/fairness_metrics.py.

def min_sat(values: Sequence[float]) -> float:
    return min(values) if values else 0.0


def mean_sat(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def jain(values: Sequence[float]) -> float:
    """(sum s_i)^2 / (n * sum s_i^2), in [1/n, 1]. 1.0 = everyone equally satisfied."""
    if not values:
        return 0.0
    total, sq = sum(values), sum(v * v for v in values)
    return 1.0 if sq == 0 else (total * total) / (len(values) * sq)


def gini(values: Sequence[float]) -> float:
    """0.0 = perfect equality. Reacts harder to one outlier than Jain does."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    spread = sum(abs(a - b) for a in values for b in values)
    return spread / (2 * n * n * mean)


def max_envy(matrix_: Dict[str, Dict[str, float]], pick_id: str, member_ids: Sequence[str]) -> float:
    """max_i ( best available to i in F - what i got on the pick )."""
    if pick_id not in matrix_:
        return 0.0
    worst = 0.0
    for mid in member_ids:
        best = max(row.get(mid, 0.0) for row in matrix_.values())
        worst = max(worst, best - matrix_[pick_id].get(mid, 0.0))
    return worst


# ── C2 + C5: the mediation loop ───────────────────────────────────────────────
#
# Ported from Vihaan's mediator_loop_veto_token.py (potluck, branch
# research-tasks, commit f045279, "Add files via upload", 2026-07-17), a Colab
# export. Recoverable at that SHA even after the branch is removed:
#   git show f045279:mediator_loop_veto_token.py
# Renamed/typed for this repo; the objective, the reweight rule, the reaction
# gradients and the veto mechanic are unchanged from the original.

REACTION_GRADIENT = {"accept": 0.0, "reject": 0.2, "strong_dislike": 0.5, "veto": 1.5}


class Mediator:
    """C1's fairness objective plus C2's reweighting loop and C5's veto token.

    Vihaan, mediator_loop_veto_token.py. `calculate_score` is the same
    mean-minus-lambda-variance objective as fairness_score/aggregate above,
    written independently in numpy; kept in his own class shape here rather
    than merged into aggregate(), so the notebook that recovers his prototype
    can assert it reproduces his original printed output line for line.
    """

    def __init__(self, users: Sequence[str], satisfaction: Dict[str, Dict[str, float]],
                 learning_rate: float = 0.2, fairness_lambda: float = 0.5):
        self.users = list(users)
        self.satisfaction = satisfaction
        self.weights = {u: 1 / len(self.users) for u in self.users}
        self.veto_tokens = {u: 1 for u in self.users}
        self.eta = learning_rate
        self.lambda_fairness = fairness_lambda

    def calculate_score(self, option: str) -> float:
        scores = [self.satisfaction[option][u] for u in self.users]
        weights = [self.weights[u] for u in self.users]
        weighted_mean = sum(w * s for w, s in zip(weights, scores))
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return weighted_mean - self.lambda_fairness * var

    def recommend(self, candidates: Sequence[str]) -> Tuple[str, Dict[str, float]]:
        scores = {c: self.calculate_score(c) for c in candidates}
        return max(scores, key=scores.get), scores

    def update_weights(self, reactions: Dict[str, str]) -> None:
        """w_i <- w_i * exp(eta * g(reaction_i)) / Z — exponentiated-gradient reweight."""
        new_weights = {}
        for user, reaction in reactions.items():
            g = REACTION_GRADIENT.get(reaction, 0.0)
            new_weights[user] = self.weights[user] * math.exp(self.eta * g)
        z = sum(new_weights.values())
        self.weights = {u: new_weights[u] / z for u in self.users}

    def apply_veto(self, choice: str, reactions: Dict[str, str], candidates: List[str]) -> List[str]:
        for user, reaction in reactions.items():
            if reaction == "veto":
                if self.veto_tokens[user] > 0:
                    self.veto_tokens[user] -= 1
                    if choice in candidates:
                        candidates.remove(choice)
                # else: no tokens left, the veto has no effect (silent, matches original)
        return candidates

    def run_round(self, candidates: List[str], reactions: Dict[str, str]) -> str:
        """One mediation round: propose -> veto -> re-propose if vetoed -> reweight.

        NOTE ON SEMANTICS: `candidates` is mutated IN PLACE when a veto is
        spent, matching Vihaan's original (mediator_loop_veto_token.py passes
        the caller's list straight into apply_veto, whose .remove() mutates
        it). A vetoed option therefore stays removed for every later round.
        An earlier port here copied the list, which silently made every veto
        last exactly one round and produced veto/no-veto arms that were
        numerically identical -- a porting bug, not a finding."""
        choice, _ = self.recommend(candidates)
        self.apply_veto(choice, reactions, candidates)   # in place, by design
        if choice not in candidates:
            choice, _ = self.recommend(candidates)
        self.update_weights(reactions)
        return choice


DEFAULT_MAX_ROUNDS = 8

# Why a cap exists at all, and what it is NOT
# -------------------------------------------
# Some groups never converge: no option ever lifts the worst-off member over
# the bar, so the loop would run forever. The fix is a STATED cap plus an
# honest non-convergence rate -- not tuning eta/lambda until everything
# converges. A loop that always succeeds because the objective was bent to
# make it succeed measures nothing.
#
# The three ways a mediation can end are kept SEPARATE, because they mean
# different things and an earlier version of the harness conflated them:
#   consensus            the worst-off member cleared the bar
#   round_cap            still going when the cap hit -- might have converged later
#   candidates_exhausted vetoes removed every remaining option -- would NEVER converge
# Reporting "72% converged" without splitting the other 28% hides whether the
# cap was too tight or the group was unsatisfiable.


def mediate(satisfaction: Dict[str, Dict[str, float]], member_ids: Sequence[str], *,
            bar: float = 0.45, max_rounds: int = DEFAULT_MAX_ROUNDS,
            eta: float = 0.5, lam: float = 0.5, use_veto: bool = False,
            reaction: str = "strong_dislike") -> Dict:
    """Run the closed loop to consensus, the round cap, or candidate exhaustion.

    Terminates by construction: every path out of the loop is bounded by
    `max_rounds`. Returns the termination reason so the caller can report the
    three endings separately rather than lumping them into "did not converge".
    """
    med = Mediator(member_ids, satisfaction, learning_rate=eta, fairness_lambda=lam)
    candidates = list(satisfaction.keys())
    history = []

    for rnd in range(1, max_rounds + 1):
        if not candidates:
            return {"converged": False, "reason": "candidates_exhausted", "rounds": rnd - 1,
                    "pick": None, "worst_sat": None, "history": history}
        pick, _ = med.recommend(candidates)
        worst_id = min(member_ids, key=lambda m: satisfaction[pick][m])
        worst = satisfaction[pick][worst_id]
        history.append({"round": rnd, "pick": pick, "worst_member": worst_id,
                        "worst_sat": round(worst, 4), "n_candidates": len(candidates),
                        "weights": dict(med.weights)})
        if consensus_reached(worst, bar):
            return {"converged": True, "reason": "consensus", "rounds": rnd,
                    "pick": pick, "worst_sat": worst, "history": history}
        react = "veto" if (use_veto and med.veto_tokens.get(worst_id, 0) > 0) else reaction
        med.run_round(candidates, {m: (react if m == worst_id else "accept")
                                   for m in member_ids})

    return {"converged": False, "reason": "round_cap", "rounds": max_rounds,
            "pick": history[-1]["pick"] if history else None,
            "worst_sat": history[-1]["worst_sat"] if history else None,
            "history": history}


def solvable(satisfaction: Dict[str, Dict[str, float]], member_ids: Sequence[str],
             bar: float = 0.45) -> bool:
    """Is ANY option good enough to clear the bar for the worst-off member?

    The ceiling control. If this is False the group is unsatisfiable and no
    mediator, however clever, could have converged -- so it must not be
    counted against the mediator."""
    return any(min(satisfaction[o][m] for m in member_ids) >= bar
               for o in satisfaction)


def consensus_reached(worst_score: float, bar: float = 0.6) -> bool:
    """A round-stop rule: the worst-off member's satisfaction on the current
    pick has crossed an acceptance bar. Original to this repo — Vihaan's
    prototype leaves the stop condition to the caller; this names one."""
    return worst_score >= bar


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION HARNESS (PRD-5) — synthetic groups, catalog loading, one-group run
# Original to this paper. Everything here is deterministic given a seed.
# ═════════════════════════════════════════════════════════════════════════════

import csv as _csv
import random as _random


def load_catalog() -> List[Dict]:
    """The option catalog the harness scores. Which dataset is chosen by the
    FM_DATASET environment variable, so the notebooks run unedited on every
    dataset (task doc §7.5):

        FM_DATASET unset / "potluck"  -> the 5 menus in data/Menu/ (default)
        FM_DATASET = "ubereats"       -> load_catalog_ubereats(FM_CITY)
        FM_DATASET = "ubereats-shuffled" -> the same city with dishes dealt out
                                         at random (random-menu control, §7.4)
    """
    dataset = (os.environ.get("FM_DATASET") or "potluck").strip().lower()
    if dataset == "potluck":
        return _load_catalog_potluck()
    if dataset == "ubereats":
        return load_catalog_ubereats(os.environ.get("FM_CITY"))
    if dataset == "ubereats-shuffled":
        return shuffle_menus(load_catalog_ubereats(os.environ.get("FM_CITY")),
                             seed=int(os.environ.get("FM_SHUFFLE_SEED") or 7))
    raise ValueError(f"FM_DATASET={dataset!r}: expected 'potluck', 'ubereats' or 'ubereats-shuffled'")


def shuffle_menus(catalog: Sequence[Dict], seed: int = 7) -> List[Dict]:
    """Random-menu control: the same restaurants and menu sizes, but every dish
    is dealt to a random restaurant, so each one tastes like 'the average of
    everything'. If lambda helps on the real catalog and not on this one, the
    gain comes from real differences between restaurants."""
    dishes = [d for o in catalog for d in o["menu"]]
    _random.Random(seed).shuffle(dishes)
    out, i = [], 0
    for o in catalog:
        menu = dishes[i:i + len(o["menu"])]
        i += len(o["menu"])
        out.append({**o, "menu": menu, "taste": _restaurant_taste(menu)})
    return out


def _restaurant_taste(menu: Sequence[Dict]) -> List[float]:
    """Equation 1: a restaurant tastes like the mean of its dishes, skipping
    dishes whose taste is unknown (all zeros). One definition for every
    dataset, so the rule cannot differ between the two cities or sources."""
    nz = [d["taste"] for d in menu if sum(abs(x) for x in d["taste"]) > 0]
    return [sum(v[i] for v in nz) / len(nz) for i in range(6)] if nz else [0.0] * 6


def _load_catalog_potluck() -> List[Dict]:
    """Build the option catalog from the real menus in data/Menu/.

    Each option carries per-dish taste vectors and a whole-menu average, the
    same construction 01-taste/02 performs — kept here so the harness and the
    notebooks cannot drift apart."""
    flavor = {}
    with open(DATA / "ingredient-flavor.csv") as f:
        for row in _csv.DictReader(f):
            n = (row.get("ingredient") or "").strip().lower()
            if n:
                flavor[n] = {d: float(row.get(d, 0) or 0) for d in TASTE_DIMS}

    def infer(dish: str) -> List[float]:
        m = [flavor[i] for i in flavor if i in dish.lower()]
        return [sum(x[k] for x in m) / len(m) for k in TASTE_DIMS] if m else [0.0] * 6

    def dish_names(node):
        if isinstance(node, dict):
            if ("item" in node or "name" in node) and "price" in node:
                yield (node.get("item") or node.get("name") or "").strip()
                return
            for v in node.values():
                yield from dish_names(v)
        elif isinstance(node, list):
            for v in node:
                yield from dish_names(v)

    catalog = []
    for f in sorted((DATA / "Menu").glob("*.json")):
        names = [n for n in dish_names(json.loads(f.read_text())) if n]
        menu = [{"name": n, "taste": infer(n)} for n in names]
        catalog.append({"option_id": f.stem, "name": f.stem, "menu": menu,
                        "taste": _restaurant_taste(menu)})
    return catalog


def load_personas() -> List[Dict]:
    return json.loads((DATA / "personas.json").read_text())


# ═════════════════════════════════════════════════════════════════════════════
# REAL FOOD DATA — Food.com recipe taste + Uber Eats restaurants
#
# Original to this paper (Vincent, wangvincentx-cmd), per the task doc's food
# side: recipe_taste() is the ONE taste function both catalog (dish taste) and
# personas (user taste = mean over 4+-star recipes) import. No Kaggle data is
# committed — see data/README.md for where each file goes.
# ═════════════════════════════════════════════════════════════════════════════

import ast as _ast
import functools as _functools
import hashlib as _hashlib
import re as _re
import statistics as _statistics

EXTERNAL = DATA / "external"              # gitignored: raw Kaggle-derived files
EXTERNAL_CACHE = DATA / "external_cache"  # gitignored: catalogs built from them
BITTER_CSV = DATA / "bitter-ingredients.csv"

_RECIPES: Optional[Dict[int, Tuple[Dict[str, float], List[str]]]] = None


def flavour_csv_path() -> Path:
    return Path(os.environ.get("FM_FLAVOUR_CSV") or EXTERNAL / "recipes_with_flavour_profiles.csv")


def _recipes() -> Dict[int, Tuple[Dict[str, float], List[str]]]:
    """id -> (5-D flavour profile, ingredient list), loaded once on first use.

    The profile's five dims are, measured on 300 random recipes, the mean of
    data/ingredient-flavor.csv over the recipe ingredients whose exact name is
    in that table (mean abs error 0.0012). bitter is added below by the same
    rule."""
    global _RECIPES
    if _RECIPES is None:
        path = flavour_csv_path()
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Copy recipes_with_flavour_profiles.csv there "
                f"(or set FM_FLAVOUR_CSV) — see data/README.md.")
        _csv.field_size_limit(1 << 24)
        out = {}
        with open(path, newline="") as f:
            for row in _csv.DictReader(f):
                out[int(row["id"])] = (_ast.literal_eval(row["flavor_profile"]),
                                       [i.strip().lower() for i in _ast.literal_eval(row["ingredients"])])
        _RECIPES = out
    return _RECIPES


@_functools.lru_cache(maxsize=1)
def _flavour_table_names() -> frozenset:
    with open(DATA / "ingredient-flavor.csv") as f:
        return frozenset((r.get("ingredient") or "").strip().lower() for r in _csv.DictReader(f))


@_functools.lru_cache(maxsize=1)
def _bitter_patterns() -> Tuple[Tuple[Any, float], ...]:
    """Longest phrase first, so an exclusion ('white chocolate', 0) beats the
    shorter bitter term it contains."""
    with open(BITTER_CSV) as f:
        rows = [(r["ingredient"].strip().lower(), float(r["bitter"])) for r in _csv.DictReader(f)]
    rows.sort(key=lambda r: -len(r[0]))
    return tuple((_re.compile(r"\b" + _re.escape(name) + r"\b"), w) for name, w in rows)


@_functools.lru_cache(maxsize=None)
def ingredient_bitter(ingredient: str) -> Optional[float]:
    """Bitter weight of one ingredient from the hand-made list, or None if the
    list does not mention it. Word-boundary match, longest phrase wins."""
    s = (ingredient or "").strip().lower()
    for pat, w in _bitter_patterns():
        if pat.search(s):
            return w
    return None


def _recipe_bitter(ingredients: Sequence[str]) -> float:
    """Mean bitter weight over the ingredients we know anything about — those
    in the flavour table or the bitter list — mirroring how the other five
    dims average over known ingredients only."""
    known, total = 0, 0.0
    table = _flavour_table_names()
    for ing in ingredients:
        b = ingredient_bitter(ing)
        if b is not None or ing in table:
            known += 1
            total += b or 0.0
    return round(total / known, 4) if known else 0.0


def recipe_taste(recipe_id: Any) -> List[float]:
    """6-D taste [sweet, salty, sour, bitter, umami, spicy] of one Food.com
    recipe. Unknown id -> all zeros (never NaN), which every consumer already
    treats as 'taste unknown, skip'."""
    try:
        rec = _recipes().get(int(recipe_id))
    except (TypeError, ValueError):
        rec = None
    if rec is None:
        return [0.0] * 6
    prof, ings = rec
    vals = {d: float(prof.get(d, 0.0) or 0.0) for d in TASTE_DIMS if d != "bitter"}
    vals["bitter"] = _recipe_bitter(ings)
    return [vals[d] for d in TASTE_DIMS]


def recipe_ingredients(recipe_id: Any) -> Optional[List[str]]:
    """Ingredient list of one recipe, or None when the id is unknown — None
    means 'ingredients unknown', which a safety filter must treat as unsafe."""
    try:
        rec = _recipes().get(int(recipe_id))
    except (TypeError, ValueError):
        return None
    return list(rec[1]) if rec else None


# FDA's nine major food allergens (FALCPA 2004 + FASTER Act 2021). Keyword
# matching on ingredient names, deliberately conservative. Shellfish means
# crustacean shellfish, as in the statute; coconut is not a tree nut (FDA
# guidance, Jan 2025). One definition for catalog tagging and dish_is_safe.
FDA_ALLERGEN_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "milk": {
        "include": ["milk", "buttermilk", "butter", "cream", "cheese", "cheddar", "mozzarella",
                    "parmesan", "parmigiano", "romano", "pecorino", "ricotta", "feta", "gouda",
                    "brie", "camembert", "provolone", "gruyere", "monterey jack", "pepper jack",
                    "colby", "mascarpone", "asiago", "havarti", "fontina", "queso", "velveeta",
                    "yogurt", "yoghurt", "whey", "casein", "ghee", "paneer", "half-and-half",
                    "half and half", "custard", "gelato", "kefir", "creme fraiche", "alfredo",
                    "dulce de leche", "white chocolate", "milk chocolate"],
        "exclude": ["coconut milk", "almond milk", "soy milk", "rice milk", "oat milk",
                    "cashew milk", "coconut cream", "cream of coconut", "cream of tartar",
                    "peanut butter", "almond butter", "cashew butter", "apple butter",
                    "cocoa butter", "shea butter", "sunflower butter", "butternut",
                    "butter beans", "butter bean", "cream soda", "creamed corn",
                    "cream-style corn", "dairy-free", "non-dairy", "vegan"],
    },
    "egg": {
        "include": ["egg", "egg white", "egg yolk", "mayonnaise", "mayo", "meringue",
                    "eggnog", "aioli"],
        "exclude": ["eggplant", "vegan mayo", "egg-free", "eggless"],
    },
    "fish": {
        "include": ["fish", "salmon", "tuna", "cod", "halibut", "tilapia", "trout", "anchovy",
                    "anchovies", "sardine", "mackerel", "haddock", "snapper", "catfish",
                    "swordfish", "monkfish", "whitefish", "mahi", "flounder", "pollock",
                    "herring", "bass", "fish sauce", "worcestershire", "bonito", "dashi", "lox",
                    "caesar dressing", "surimi"],
        "exclude": ["goldfish crackers", "shellfish"],
    },
    "shellfish": {
        "include": ["shrimp", "prawn", "crab", "crabmeat", "lobster", "crawfish", "crayfish",
                    "langoustine", "shellfish"],
        "exclude": ["crab apple", "crabapple"],
    },
    "tree_nuts": {
        "include": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "filbert",
                    "macadamia", "brazil nut", "pine nut", "chestnut", "praline", "marzipan",
                    "nutella", "nut", "mixed nuts", "amaretto"],
        "exclude": ["nutmeg", "butternut", "water chestnut", "coconut", "doughnut", "donut",
                    "peanut", "nutritional yeast"],
    },
    "peanuts": {
        "include": ["peanut", "groundnut"],
        "exclude": [],
    },
    "wheat": {
        "include": ["wheat", "flour", "bread", "breadcrumbs", "bread crumbs", "panko", "pasta",
                    "spaghetti", "macaroni", "noodle", "tortilla", "pita", "couscous", "bulgur",
                    "semolina", "farina", "seitan", "cracker", "biscuit", "pie crust",
                    "pie shell", "puff pastry", "phyllo", "filo", "crouton", "soy sauce",
                    "teriyaki", "cake mix", "dough", "bagel", "bun", "english muffin",
                    "wonton", "dumpling wrapper", "orzo", "gnocchi", "ramen", "udon",
                    "lasagna", "fettuccine", "linguine", "penne", "rigatoni", "ziti",
                    "tortellini", "ravioli", "graham"],
        "exclude": ["rice flour", "corn flour", "almond flour", "coconut flour",
                    "chickpea flour", "tapioca flour", "rice noodle", "rice noodles",
                    "rice pasta", "corn tortilla", "corn tortillas", "buckwheat",
                    "gluten-free", "gluten free", "breadfruit", "wheat-free"],
    },
    "soy": {
        "include": ["soy", "soya", "soybean", "tofu", "edamame", "miso", "tempeh", "tamari",
                    "soy sauce", "teriyaki", "hoisin", "shoyu", "textured vegetable protein"],
        "exclude": [],
    },
    "sesame": {
        "include": ["sesame", "tahini", "benne", "zaatar", "za'atar", "halvah", "halva",
                    "furikake", "gomasio"],
        "exclude": [],
    },
}


@_functools.lru_cache(maxsize=1)
def _allergen_patterns() -> Tuple[Tuple[str, Any, Any], ...]:
    def alt(words):
        if not words:
            return None
        body = "|".join(_re.escape(w) for w in sorted(words, key=len, reverse=True))
        return _re.compile(r"\b(?:" + body + r")(?:s|es)?\b")
    return tuple((a, alt(k["include"]), alt(k["exclude"])) for a, k in FDA_ALLERGEN_KEYWORDS.items())


@_functools.lru_cache(maxsize=None)
def ingredient_allergens(ingredient: str) -> Tuple[str, ...]:
    """FDA allergens one ingredient name signals. Exclusion phrases are blanked
    first, so 'coconut milk' is not milk but 'coconut milk and butter' is."""
    s = (ingredient or "").strip().lower()
    found = []
    for allergen, inc, exc in _allergen_patterns():
        text = exc.sub(" ", s) if exc is not None else s
        if inc is not None and inc.search(text):
            found.append(allergen)
    return tuple(found)


def dish_allergens(ingredients: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Sorted FDA allergens across an ingredient list. None in -> None out:
    unknown ingredients must stay distinguishable from 'no allergens'."""
    if ingredients is None:
        return None
    return sorted({a for ing in ingredients for a in ingredient_allergens(ing)})


def bitter_list_sha() -> str:
    """Fingerprint of the bitter list, stored in built catalogs so a stale
    cache (built before the list changed) is detected, not silently used."""
    return _hashlib.sha256(BITTER_CSV.read_bytes()).hexdigest()[:12]


def city_slug(city: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "-", (city or "").strip().lower()).strip("-")


def ubereats_cache_path(city: str) -> Path:
    return EXTERNAL_CACHE / f"catalog_ubereats_{city_slug(city)}.json"


def load_catalog_ubereats(city: Optional[str] = None) -> List[Dict]:
    """Real Uber Eats restaurants for one city, in the shared catalog shape:
    option_id, name, taste[6], menu[{name, taste[6], allergens, ...}].

    Reads the catalog that scripts/build_ubereats_catalog.py built (the dish ->
    Food.com recipe matching needs pandas/scikit-learn; loading does not).
    Restaurant taste is recomputed here with _restaurant_taste, the same rule
    the default catalog uses."""
    if not city:
        raise ValueError("load_catalog_ubereats needs a city (argument or FM_CITY).")
    path = ubereats_cache_path(city)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with:\n  python scripts/build_ubereats_catalog.py "
            f"--city {city!r}\n(see data/README.md for the Kaggle downloads it needs).")
    built = json.loads(path.read_text())
    meta = built.get("meta", {})
    if meta.get("bitter_list_sha") != bitter_list_sha():
        print(f"[nbio] WARNING: {path.name} was built with a different bitter list; "
              f"rebuild it so bitter values are current.", file=sys.stderr)
    catalog = []
    for r in built["restaurants"]:
        menu = r["menu"]
        catalog.append({**{k: v for k, v in r.items() if k != "menu"},
                        "menu": menu, "taste": _restaurant_taste(menu)})
    return catalog


def taste_coverage(catalog: Sequence[Dict]) -> float:
    """Share of all dishes with a real, non-zero taste vector (task doc §7.2)."""
    dishes = [d for o in catalog for d in o.get("menu") or []]
    if not dishes:
        return 0.0
    return sum(1 for d in dishes if sum(abs(x) for x in d["taste"]) > 0) / len(dishes)


def taste_spread(catalog: Sequence[Dict]) -> Dict[str, Any]:
    """How different the restaurants taste from each other (task doc §7.3):
    per-dim population std of the restaurant vectors, and their mean. The
    same function scores every dataset, so the numbers are comparable."""
    vecs = [o["taste"] for o in catalog]
    per_dim = {d: (_statistics.pstdev([v[i] for v in vecs]) if len(vecs) > 1 else 0.0)
               for i, d in enumerate(TASTE_DIMS)}
    return {"n_restaurants": len(vecs), "per_dim_std": per_dim,
            "spread": sum(per_dim.values()) / len(per_dim)}


def generate_groups(n_groups: int = 200, seed: int = 7, size_range: Tuple[int, int] = (3, 6),
                    adversarial_frac: float = 0.35) -> List[Dict]:
    """Sample groups from the persona pool, with a controlled fraction built to
    force a conflict between mean and spread.

    A group is 'adversarial' when it deliberately mixes diners whose taste
    vectors point in different directions -- without those, every selector
    agrees and the comparison is vacuous (PRD-5 §3). Seeded: the same seed
    reproduces the same groups exactly.
    """
    rng = _random.Random(seed)
    pool = load_personas()
    groups = []
    for gi in range(n_groups):
        size = rng.randint(*size_range)
        adversarial = gi < int(n_groups * adversarial_frac)
        if adversarial:
            # pick diners whose taste vectors are far apart pairwise
            seed_member = rng.choice(pool)
            def dist(p):
                return 1.0 - calculate_cosine_similarity(p["taste_vector"], seed_member["taste_vector"])
            ranked = sorted(pool, key=dist, reverse=True)
            chosen = [seed_member] + ranked[:size - 1]
        else:
            chosen = rng.sample(pool, size)
        members = [{
            "member_id": f"{p['display_name']}_{i}".lower(),
            "display_name": p["display_name"],
            "diet": p["diet_type"],
            "allergies": list(p["allergies"]),
            "taste": list(p["taste_vector"]),
        } for i, p in enumerate(chosen)]
        groups.append({"group_id": f"g{gi:04d}", "adversarial": adversarial, "members": members})
    return groups


def run_group(group: Dict, catalog: Sequence[Dict], method: str = "fair",
              lam: float = DEFAULT_LAMBDA) -> Optional[Dict]:
    """Score one group against the catalog under one selector. Returns None if
    the feasible set is empty (no option is safe for everyone)."""
    members = group["members"]
    F = build_feasible_set(catalog, members)
    if not F:
        return None
    member_ids = [m["member_id"] for m in members]
    S = {o["option_id"]: {m["member_id"]: taste_similarity(m["taste"], o["taste"])
                          for m in members} for o in F}
    cand = {o["option_id"]: o["taste"] for o in F}
    ranked = rank_candidates(cand, [m["taste"] for m in members], method=method, lam=lam)
    pick_id = ranked[0][0]
    vals = [S[pick_id][mid] for mid in member_ids]
    return {
        "group_id": group["group_id"], "adversarial": group["adversarial"],
        "n_members": len(members), "F_size": len(F), "pick_id": pick_id,
        "mean_sat": mean_sat(vals), "min_sat": min_sat(vals),
        "jain": jain(vals), "gini": gini(vals),
        "max_envy": max_envy(S, pick_id, member_ids),
        "coverage": len(F) / len(catalog),
        "_S": S, "_F": F, "_member_ids": member_ids,
    }


def mean_ci(values: Sequence[float], z: float = 1.96) -> Tuple[float, float, float]:
    """(mean, half-width of the 95% CI, sd). Normal approximation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(values) / n
    if n < 2:
        return m, 0.0, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    sd = var ** 0.5
    return m, z * sd / (n ** 0.5), sd


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                       seed: int = 7) -> float:
    """Two-sided paired bootstrap p-value for mean(a) - mean(b) = 0.

    Paired because both arms are evaluated on the SAME groups; a two-sample
    test would ignore that and overstate the variance."""
    rng = _random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return 1.0
    obs = sum(diffs) / n
    centred = [d - obs for d in diffs]
    hits = 0
    for _ in range(n_boot):
        s = sum(centred[rng.randrange(n)] for _ in range(n)) / n
        if abs(s) >= abs(obs):
            hits += 1
    return (hits + 1) / (n_boot + 1)


# ═════════════════════════════════════════════════════════════════════════════
# C4 DISH ASSIGNMENT + LIKE-FOR-LIKE BASELINES (PRD-11)
#
# The naive C4 comparison comes out at +0.390 MinSat, which is NOT an effect
# size: it compares cos(diner, ONE DISH chosen to fit them) against
# cos(diner, WHOLE-MENU AVERAGE). A single item picked to match a person will
# nearly always beat the mean of ~100 items most of which were never meant for
# them. The baselines below isolate the part that is actually compensation.
#
#   B1  random safe dish        <- the one that matters. Anything the real
#                                  assignment gains OVER B1 is the compensation
#                                  effect; everything above B1 is scale artifact.
#   B2  best shared dish        <- the real competitor: one crowd-pleaser for all
#   B3  median-diner's dish     <- same question from a fairness angle
# ═════════════════════════════════════════════════════════════════════════════


def _safe_for(member: Dict, pick: Dict) -> List[Dict]:
    """That member's safe dishes at this option, from the attached safe_dishes
    if build_feasible_set put one there, else recomputed."""
    s = (pick.get("safe_dishes") or {}).get(member["member_id"])
    return s if s else safe_dishes_for(member, pick.get("menu") or [])


def assign_dishes_best(members: Sequence[Dict], pick: Dict) -> Optional[Dict[str, float]]:
    """C4 as proposed: each diner gets their OWN best safe dish.
    Returns {member_id: similarity}, or None if any diner has nothing safe."""
    out = {}
    for m in members:
        safe = _safe_for(m, pick)
        if not safe:
            return None
        out[m["member_id"]] = max(taste_similarity(m["taste"], d["taste"]) for d in safe)
    return out


def assign_dishes_random(members: Sequence[Dict], pick: Dict, seed: int = 7) -> Optional[Dict[str, float]]:
    """B1 -- each diner gets a UNIFORMLY RANDOM dish from their own safe set.

    Still a single dish, still safe, still per-diner -- the only thing removed
    is the matching. The gap between this and assign_dishes_best is therefore
    the compensation effect with the scale artifact divided out."""
    rng = _random.Random(seed)
    out = {}
    for m in members:
        safe = _safe_for(m, pick)
        if not safe:
            return None
        d = safe[rng.randrange(len(safe))]
        out[m["member_id"]] = taste_similarity(m["taste"], d["taste"])
    return out


def assign_dishes_shared_best(members: Sequence[Dict], pick: Dict) -> Optional[Dict[str, float]]:
    """B2 -- ONE dish for the whole table: the dish maximising mean satisfaction
    across members, restricted to dishes safe for EVERY member.

    This is the real competitor to per-diner assignment: a venue could simply
    have one crowd-pleaser."""
    common = None
    for m in members:
        names = {d["name"] for d in _safe_for(m, pick)}
        if not names:
            return None
        common = names if common is None else (common & names)
    if not common:
        return None
    by_name = {d["name"]: d for d in (pick.get("menu") or [])}
    best, best_mean = None, -1.0
    for n in sorted(common):                      # sorted -> deterministic ties
        d = by_name.get(n)
        if not d:
            continue
        mean = sum(taste_similarity(m["taste"], d["taste"]) for m in members) / len(members)
        if mean > best_mean:
            best, best_mean = d, mean
    if best is None:
        return None
    return {m["member_id"]: taste_similarity(m["taste"], best["taste"]) for m in members}


def assign_dishes_median_diner(members: Sequence[Dict], pick: Dict,
                               S_row: Dict[str, float]) -> Optional[Dict[str, float]]:
    """B3 -- one shared dish, chosen as the best dish for the MEDIAN-satisfaction
    diner (by their satisfaction with the venue), then given to everyone."""
    ordered = sorted(members, key=lambda m: S_row.get(m["member_id"], 0.0))
    median_member = ordered[len(ordered) // 2]
    safe = _safe_for(median_member, pick)
    if not safe:
        return None
    # restrict to dishes safe for everyone -- a shared dish must be servable to all
    common = None
    for m in members:
        names = {d["name"] for d in _safe_for(m, pick)}
        common = names if common is None else (common & names)
    cands = [d for d in safe if d["name"] in (common or set())]
    if not cands:
        return None
    best = max(cands, key=lambda d: (taste_similarity(median_member["taste"], d["taste"]), d["name"]))
    return {m["member_id"]: taste_similarity(m["taste"], best["taste"]) for m in members}


# ═════════════════════════════════════════════════════════════════════════════
# C2: THE OBJECTION PARSER (PRD-10)
#
# The one piece that makes the mediation loop conversational. Division of
# labour, and it is the paper's central design claim:
#     LLM  reads the sentence, emits {member, label}
#     CODE does the gradient lookup, the reweight, the argmax, the re-pick
# The model must never emit a weight, a score, or a restaurant.
# ═════════════════════════════════════════════════════════════════════════════

VALID_LABELS = ("accept", "reject", "strong_dislike", "veto")

# Committed BEFORE any ratings were collected (PRD-10 §3 control 5).
# If this string changes, re-collect from scratch under a new version.
OBJECTION_PROMPT_VERSION = "v1"
OBJECTION_PROMPT = """You classify one person's reaction to a restaurant suggestion.

Reply with ONLY a JSON object, no prose, no markdown fence:
{"member": "<name>", "label": "<label>", "confidence": <0.0-1.0>}

label must be exactly one of: accept, reject, strong_dislike, veto
  accept          they are happy with the suggestion, including reluctant or
                  conditional agreement ("fine if they have vegan options")
  reject          mild disagreement, a preference for something else
  strong_dislike  strong negative feeling. SARCASM counts as strong_dislike
                  ("oh great, sushi again" is NOT acceptance)
  veto            they explicitly veto/block it, OR state a hard safety
                  constraint that rules the option out (allergy, diet)

member must be the person whose PREFERENCE is being expressed, which is not
always the speaker. "Cara won't eat that" expresses Cara's preference, so
member is Cara. Use "SELF" when the speaker is talking about themselves.
member must be "SELF" or exactly one of: <<MEMBERS>>

Reaction: <<TEXT>>"""


def parse_objection(text: str, member_names: Sequence[str], *, client=None,
                    model: str = None, temperature: float = 0.0) -> Dict:
    """Free text -> {member, label, confidence, raw}.

    Degrades safely and deliberately:
      * unparseable / invalid label -> "reject" at confidence 0.0, never
        "accept". Silence is not consent: letting a malformed complaint be
        recorded as agreement would let the loop reweight the wrong way.
      * a member name the model invented raises ValueError rather than
        reweighting a diner who is not in the group.
    """
    import json as _json
    fallback = {"member": "SELF", "label": "reject", "confidence": 0.0, "raw": ""}

    if client is None:                       # no client -> explicit, not silent
        return {**fallback, "raw": "<no client supplied>"}

    # .replace, not .format -- the prompt contains literal JSON braces
    prompt = (OBJECTION_PROMPT.replace("<<MEMBERS>>", ", ".join(member_names))
                              .replace("<<TEXT>>", text))
    try:
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {**fallback, "raw": f"<call failed: {type(e).__name__}>"}

    out = {**fallback, "raw": raw}
    try:
        body = raw[raw.index("{"): raw.rindex("}") + 1]
        got = _json.loads(body)
    except Exception:
        return out                            # malformed -> reject, conf 0.0

    label = str(got.get("label", "")).strip().lower()
    if label not in VALID_LABELS:
        return out                            # out-of-vocabulary -> reject

    member = str(got.get("member", "SELF")).strip()
    if member != "SELF" and member not in member_names:
        raise ValueError(
            f"parse_objection: model named {member!r}, who is not in the group "
            f"{list(member_names)}. Refusing to reweight an unknown diner.")

    try:
        conf = float(got.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    out.update(member=member, label=label, confidence=max(0.0, min(1.0, conf)))
    return out


def objection_to_reaction(parsed: Dict) -> str:
    """The label the Mediator's reweight consumes. Kept separate so the
    gradient table stays the single place a label becomes a number."""
    return parsed["label"] if parsed["label"] in REACTION_GRADIENT else "reject"


# ═════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE PROXY (PRD-9)
#
# Stands in for a human study we cannot run before submission. An ensemble of
# models rates each group recommendation for perceived fairness, satisfaction
# and consensus. It is a PROXY, never a measurement of human perception, and
# every table built from it must say so.
#
# NO EXTERNAL DEPENDENCIES: runs on the repo + API keys the team already has.
# ═════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT_VERSION = "v1"

# Committed BEFORE any ratings are collected (PRD-9 §5 control 5).
# If this string changes, bump the version and re-collect from scratch.
JUDGE_PROMPT = """You are evaluating how a group of diners would feel about a restaurant
recommendation made on their behalf.

THE GROUP
<<GROUP>>

THE RECOMMENDATION
<<OPTION>>

HOW WELL IT MATCHES EACH DINER (0.0 = poor match, 1.0 = perfect match)
<<SATS>>

Rate the recommendation on three 1-7 scales, judging how the GROUP would
perceive it, not whether the numbers are arithmetically optimal:
  fairness      1 = deeply unfair to someone, 7 = everyone treated fairly
  satisfaction  1 = the group would be unhappy, 7 = the group would be pleased
  consensus     1 = the group would argue, 7 = the group would readily agree

Reply with ONLY a JSON object, no prose, no markdown fence:
{"fairness": <1-7>, "satisfaction": <1-7>, "consensus": <1-7>, "rationale": "<one sentence>"}"""


def _judge_render(group: Dict, option: Dict, satisfactions: Dict[str, float],
                  reverse_order: bool = False) -> str:
    """Build the prompt. `reverse_order` flips diner presentation order, which
    is the order-bias control (PRD-9 §5 control 2) -- a judge whose rating
    depends on who is listed first is not measuring the recommendation."""
    members = list(group["members"])
    if reverse_order:
        members = members[::-1]
    glines, slines = [], []
    for m in members:
        allergies = ", ".join(m.get("allergies") or []) or "none"
        glines.append(f"  - {m['display_name']}: diet {m.get('diet','mix')}, allergies {allergies}")
        slines.append(f"  - {m['display_name']}: {satisfactions.get(m['member_id'], 0.0):.2f}")
    return (JUDGE_PROMPT
            .replace("<<GROUP>>", "\n".join(glines))
            .replace("<<OPTION>>", f"  {option.get('name', option.get('option_id',''))}")
            .replace("<<SATS>>", "\n".join(slines)))


def judge_recommendation(group: Dict, option: Dict, satisfactions: Dict[str, float], *,
                         client=None, model: str = None, temperature: float = 0.0,
                         reverse_order: bool = False) -> Dict:
    """Rate one (group, recommendation) pair on three 1..7 scales.

    Degradation contract (PRD-9 §2): a malformed reply, an out-of-range
    rating, or an API failure returns ok=False and the item is DROPPED by the
    caller -- never coerced to a midpoint. Substituting 4/7 for a failed call
    would invent data; reporting the drop rate does not.
    """
    import json as _json
    out = {"fairness": None, "satisfaction": None, "consensus": None,
           "rationale": "", "raw": "", "model": model, "ok": False,
           "reverse_order": reverse_order, "prompt_version": JUDGE_PROMPT_VERSION}

    if client is None:
        out["raw"] = "<no client supplied>"
        return out

    try:
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "user",
                       "content": _judge_render(group, option, satisfactions, reverse_order)}])
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        out["raw"] = f"<call failed: {type(e).__name__}>"
        return out

    out["raw"] = raw
    try:
        got = _json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except Exception:
        return out                                   # malformed -> ok=False

    vals = {}
    for k in ("fairness", "satisfaction", "consensus"):
        try:
            v = int(round(float(got[k])))
        except (KeyError, TypeError, ValueError):
            return out                               # missing/unparseable -> drop
        if not 1 <= v <= 7:
            return out                               # out of range -> drop
        vals[k] = v

    out.update(vals, rationale=str(got.get("rationale", ""))[:300], ok=True)
    return out


def judge_ensemble(group: Dict, option: Dict, satisfactions: Dict[str, float], *,
                   client, models: Sequence[str], k: int = 5,
                   temperature: float = 0.0, meter: "Meter" = None,
                   cost_per_call: float = 0.0002) -> List[Dict]:
    """Every model x k repeats x both orders. Returns the raw per-call rows;
    aggregate them with judge_aggregate. Per-call rows are what gets logged --
    the aggregate is cheap to recompute, the calls are not."""
    rows = []
    for model in models:
        for rep in range(k):
            for rev in (False, True):
                r = judge_recommendation(group, option, satisfactions, client=client,
                                         model=model, temperature=temperature,
                                         reverse_order=rev)
                r.update(group_id=group["group_id"], option_id=option.get("option_id"), repeat=rep)
                rows.append(r)
                if meter is not None:
                    meter.record(f"judge:{model}", cost_per_call)
    return rows


def judge_aggregate(rows: Sequence[Dict], order_tolerance: float = 1.0) -> Dict:
    """Collapse per-call rows into one item-level result, applying the
    order-bias control and reporting what it discarded.

    Returns means per dimension, the per-dimension sd across repeats
    (self-consistency), the drop rate, and whether the item survived the
    order-bias check."""
    dims = ("fairness", "satisfaction", "consensus")
    ok = [r for r in rows if r["ok"]]
    out = {"n_calls": len(rows), "n_ok": len(ok),
           "drop_rate": round(1 - len(ok) / len(rows), 4) if rows else 1.0,
           "order_ok": True}
    if not ok:
        return {**out, **{d: None for d in dims}, **{f"{d}_sd": None for d in dims}}

    for d in dims:
        vals = [r[d] for r in ok]
        m, _, sd = mean_ci([float(v) for v in vals])
        out[d] = round(m, 3)
        out[f"{d}_sd"] = round(sd, 3)
        fwd = [r[d] for r in ok if not r["reverse_order"]]
        rev = [r[d] for r in ok if r["reverse_order"]]
        if fwd and rev:
            gap = abs(sum(fwd) / len(fwd) - sum(rev) / len(rev))
            out[f"{d}_order_gap"] = round(gap, 3)
            if gap > order_tolerance:
                out["order_ok"] = False              # discard at the caller
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, used for inter-model agreement on the selector
    ranking (PRD-9 §5 control 1). No scipy dependency."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):                        # average ties
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    if len(a) < 2:
        return float("nan")
    ra, rb = ranks(list(a)), ranks(list(b))
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")
