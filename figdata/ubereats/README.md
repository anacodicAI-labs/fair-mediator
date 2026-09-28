# C1 on real restaurants: Uber Eats Houston and Seattle

Vincent's food-side tasks (task doc §7), run on 27 September 2026. Every number below comes from a CSV in
this folder or in `figdata/ubereats-*/`, and each can be regenerated with the commands in `data/README.md`.

**The diners are still the 95 invented personas.** Real Food.com users are Vihaan's task. When they land,
rerun the same commands. Nothing on the food side needs to change.

## Data

| | Houston, TX | Seattle, WA | today (Potluck) |
|---|---|---|---|
| restaurants | 300 (seeded sample of 3,290 with ≥10 dishes) | 300 (of 1,460) | 5 |
| dishes | 15,711 | 13,942 | 574 |
| taste coverage | 95.5% | 95.2% | 46.5% |
| **spread** (mean per-taste std) | **0.040** | **0.044** | 0.017 |
| restaurant-vs-restaurant cosine, min | 0.44 | 0.33 | 0.96 |
| matcher: exact or close dish | 76% (67–83%) | 72% (63–80%) | — |

The spread is 2.3–2.6× today's. The task doc's "~0.1" was a loose figure, and the measured baseline is 0.017.
The two-city spreads are still small in absolute terms. Each restaurant tastes like the mean of about 50
dishes, and that averaging pulls restaurants together. See `validation/README.md` for the matcher and bitter
checks.

## C1: does the fairness dial help? (`c1_summary.csv`)

200 seeded groups (seed 7, 35% adversarial), as in 06-evaluate. The gain is the per-group MinSat(fair) −
MinSat(average), with a 95% bootstrap CI over groups. "Moved" means the pick at λ=0 differs from the pick at λ=5.

| dataset | λ changes the pick | MinSat gain, λ=0.5 | MinSat gain, λ=5 | mean-sat cost, λ=5 |
|---|---|---|---|---|
| Potluck (today) | 29% | +0.0003 [+0.0001, +0.0005] | +0.001 [+0.000, +0.002] | −0.004 |
| **Houston** | 31.5% | **+0.012 [+0.006, +0.018]** | **+0.024 [+0.017, +0.031]** | −0.004 |
| Houston, shuffled menus | 25.5% | +0.002 [+0.000, +0.003] | +0.008 [+0.005, +0.011] | −0.002 |
| **Seattle** | 38% | +0.002 [+0.000, +0.004] | **+0.020 [+0.016, +0.026]** | −0.004 |
| Seattle, shuffled menus | 36.5% | +0.003 [+0.001, +0.004] | +0.010 [+0.007, +0.013] | −0.003 |

What the table shows:
- **The gain is now measurable.** At λ=5 the CI is well clear of 0 in both cities, and the gain is about 20× today's.
  The fair pick raises the worst-off diner by about 0.02 on the cosine scale, and mean satisfaction drops by 0.004.
- **The control works as intended.** With the same restaurants and dishes dealt out at random, the gain falls to a
  third (Houston) or a half (Seattle). Most of the effect comes from real differences between restaurants.
  Some of it survives shuffling, because random menus of about 50 dishes still differ a little (shuffled spread 0.023).
- **The dial has to be turned up.** At the harness's default λ=0.5, Seattle shows no gain over its control.
  λ=5 is where both cities separate from their controls. The paper should report the whole λ sweep
  (`fig1_lambda_sweep.csv`), not one λ.
- **The effect is modest in absolute size.** MinSat goes from 0.709 to 0.733. Cosine similarity between non-negative
  6-D vectors lives mostly in the 0.6–0.9 band, so a 0.02 shift there is not negligible. Still, the honest
  wording is "a consistent, significant but small improvement", not "a large one".
- The CIs treat groups as independent. They are not fully independent, because all 200 groups draw from one pool
  of 95 personas, so the intervals are somewhat optimistic.

## Satisfaction formula: decision (task §7.6)

**Decision: A, the whole menu.** Proposed text for the paper:

> We score a diner against a restaurant's whole-menu taste vector (Eq. 1) rather than the mean of the dishes that
> diner can safely eat. On both cities the two formulas give the same conclusions within their confidence intervals
> (Houston, λ=5: MinSat gain +0.024 whole-menu vs +0.022 safe-subset; Seattle +0.020 vs +0.017), so the simpler
> formula, one vector per restaurant, costs nothing. The per-diner view is instead captured where it matters, by
> dish-level compensation (C4), which scores each diner against their own best safe dish.

Caveat to re-check: the two formulas agree partly because the current safety check (`dish_is_safe`, dish name
only) never removes a restaurant. The feasible set is all 300 restaurants for every group. Vihaan's ingredient-level
fail-closed filter will shrink each diner's safe subset. Rerun `python scripts/c1_report.py` once it lands. If
A and B then diverge, revisit this decision.

## Table 3: generalisation (`table3_datasets.csv`)

Formula A, λ=0.5 (the Table 1 setting), 200 groups each, ± half-width of the 95% CI.

| city | selector | mean sat | MinSat | Jain |
|---|---|---|---|---|
| Houston, TX | fair | 0.826 | 0.721 | 0.990 |
| Houston, TX | average | 0.826 | 0.709 | 0.989 |
| Seattle, WA | fair | 0.819 | 0.700 | 0.989 |
| Seattle, WA | average | 0.819 | 0.698 | 0.989 |

The direction is the same in both cities (fair ≥ average on MinSat, at no mean cost), but at λ=0.5 the Seattle gap
is within noise. Consider reporting Table 3 at λ=5, where both cities show a clear gain.

## Harness runs (06-evaluate, unedited)

**λ sweep (`fig1_lambda_sweep.csv`)**: MinSat rises with λ up to λ≈10 (Houston 0.709 → 0.735, Seattle 0.698 → 0.724)
and then falls (0.708 / 0.703 at λ=100), while mean satisfaction falls steadily. A very large λ picks restaurants
that everyone rates equally low. The paper's Figure 1 should show this peak.

**C4 dish compensation (`table5_compensation.csv`)**: MinSat 0.721 → 0.840 in Houston (+0.119) and 0.700 → 0.792 in
Seattle (+0.092), against +0.390 on Potluck. The same scale caveat applies as before (one fitted dish vs a menu mean).

**Mediator and veto (Tables 2 and 4) are degenerate on this data, and need Vihaan.** The consensus bar is an absolute
0.45 on the worst-off diner's satisfaction. On real restaurants every group's first pick already clears it, so one-shot,
mediator and veto all converge in 1 round for 200/200 groups. The bar was tuned to the Potluck scale (MinSat ≈ 0.46)
and does not transfer. It should be relative (for example a percentile of the group's own options, or the best
achievable MinSat in F minus a margin), or re-derived once real users are in.

`figdata/ubereats-houston-tx/`, `figdata/ubereats-seattle-wa/` and `figdata/ubereats-shuffled-houston-tx/` hold
the full harness output per dataset. Cell 24 of the notebook still writes a placeholder `table3_datasets.csv` labelled
"potluck", and those copies were removed. The real Table 3 is the file in this folder. The top-level `figdata/`
still holds the Potluck run until the real personas land.
