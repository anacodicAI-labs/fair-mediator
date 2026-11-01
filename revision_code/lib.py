import os
"""Numpy re-implementation of the paper's selectors and mediation loop, plus the
revision experiments' building blocks. Verified against the paper's CSVs by
check_baseline.py before anything new is run."""
import pickle
import numpy as np

CACHE = os.environ.get("FM_REV_CACHE", "cache.pkl")
CITIES = ["Houston, TX", "Seattle, WA"]
LAMBDAS = [0, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]
GRAD = {"accept": 0.0, "reject": 0.2, "strong_dislike": 0.5, "veto": 1.5}
ETA, CAP, MARGIN = 0.5, 8, 0.01

_data = None


def data():
    global _data
    if _data is None:
        with open(CACHE, "rb") as f:
            _data = pickle.load(f)
    return _data


def var(M):
    return M.var(axis=1)


def fair_scores(M, lam, w=None):
    mean = M.mean(axis=1) if w is None else M @ w
    return mean - lam * var(M)


def pick(M, rule, lam=0.5, w=None, mask=None):
    if rule == "average":
        sc = M.mean(axis=1)
    elif rule == "least_misery":
        sc = M.min(axis=1)
    elif rule == "most_pleasure":
        sc = M.max(axis=1)
    elif rule == "fair":
        sc = fair_scores(M, lam, w)
    else:
        raise ValueError(rule)
    if mask is not None:
        sc = np.where(mask, sc, -np.inf)
    return int(np.argmax(sc))          # first max = catalog order, as in nbio


def jain(v):
    v = np.asarray(v, float)
    return float(v.sum() ** 2 / (len(v) * (v ** 2).sum()))


def boot_ci(x, n_boot=10000, seed=7):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ratio_ci(num, den, n_boot=10000, seed=7):
    rng = np.random.default_rng(seed)
    num = np.asarray(num, float); den = np.asarray(den, float)
    idx = rng.integers(0, len(num), (n_boot, len(num)))
    r = 100 * num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def c1_summary(Ms, lam, evals=None):
    """Ms: list of per-group utility matrices used for SELECTION.
    evals: list of matrices used for EVALUATION (default: the same).
    Returns gain, CI, pct of achievable (achievable defined on the eval
    matrix with LM selected on the selection matrix), costs."""
    evals = evals or Ms
    g_fair, g_lm, d_mean_f, d_mean_lm, moved = [], [], [], [], []
    for M, E in zip(Ms, evals):
        a = pick(M, "average"); f = pick(M, "fair", lam); l = pick(M, "least_misery")
        g_fair.append(E[f].min() - E[a].min())
        g_lm.append(E[l].min() - E[a].min())
        d_mean_f.append(E[f].mean() - E[a].mean())
        d_mean_lm.append(E[l].mean() - E[a].mean())
        moved.append(f != a)
    lo, hi = boot_ci(g_fair)
    plo, phi = ratio_ci(g_fair, g_lm)
    return dict(lam=lam, gain=np.mean(g_fair), gain_lo=lo, gain_hi=hi,
                achievable=np.mean(g_lm),
                pct=100 * np.sum(g_fair) / np.sum(g_lm) if np.sum(g_lm) != 0 else float("nan"),
                pct_lo=plo, pct_hi=phi, cost=np.mean(d_mean_f), cost_lm=np.mean(d_mean_lm),
                moved=100 * np.mean(moved))


def mediate(S, *, lam=0.5, eta=ETA, cap=CAP, margin=MARGIN, bar=None,
            veto_remove=False, veto_grad=False, react_fn=None, U=None, rng=None,
            parser=None, memory=False):
    """Closed-loop mediation.

    S        proxy utility matrix the system sees (|F| x n)
    U        'true' utility the diners react with (default: S)
    bar      consensus bar on U (default: LM floor of U minus margin)
    react_fn function(u_row, bar, worst_idx, rng) -> list of intended labels
             (default: the paper's rule -- worst-off diner below bar says
             strong_dislike, everyone else accepts)
    parser   function(label, rng) -> label the loop receives (LLM replay)
    veto_remove  a diner holding a token removes the current pick
    veto_grad    the veto reaction carries g=1.5 instead of 0.5
    memory   a rejected pick is never proposed again (tabu)
    Returns dict(converged, rounds, pick)."""
    U = S if U is None else U
    nF, n = S.shape
    if bar is None:
        bar = U.min(axis=1).max() - margin
    w = np.full(n, 1.0 / n)
    alive = np.ones(nF, bool)
    tokens = np.ones(n, int)
    for rnd in range(1, cap + 1):
        if not alive.any():
            return dict(converged=False, rounds=rnd - 1, pick=None, reason="exhausted")
        p = pick(S, "fair", lam, w, alive)
        u = U[p]
        if u.min() >= bar:
            return dict(converged=True, rounds=rnd, pick=p, reason="consensus")
        worst = int(np.argmin(u))
        if react_fn is None:
            labels = ["accept"] * n
            labels[worst] = "strong_dislike"
        else:
            labels = react_fn(u, bar, worst, rng)
        # a diner who would say strong_dislike and still holds a token says
        # "veto" instead (the original harness rule, generalised to any diner)
        if veto_remove or veto_grad:
            labels = ["veto" if (l == "strong_dislike" and tokens[i] > 0) else l
                      for i, l in enumerate(labels)]
        if parser is not None:
            labels = [parser(l, rng) for l in labels]
        g = np.zeros(n)
        removed = False
        for i, l in enumerate(labels):
            if l == "veto":
                if veto_remove or veto_grad:
                    if tokens[i] > 0:
                        tokens[i] -= 1
                        removed = removed or veto_remove
                    gi = GRAD["veto"] if veto_grad else GRAD["strong_dislike"]
                else:
                    gi = GRAD["veto"]   # original nbio semantics: a parsed veto carries g=1.5
            else:
                gi = GRAD[l]
            g[i] = gi
        if removed or (memory and any(l != "accept" for l in labels)):
            alive[p] = False
        w = w * np.exp(eta * g)
        w = w / w.sum()
    return dict(converged=False, rounds=cap, pick=p, reason="cap")
