# C1 and C4 with real diners: Food.com users on Uber Eats Houston and Seattle

Same runs as `figdata/ubereats/`, with the 95 invented personas replaced by real Food.com users
(`FM_PERSONAS=foodcom`, see `data/README.md` §3). 200 seeded groups (seed 7, 35% adversarial).
The full harness output per city is in `figdata/ubereats-*-foodcom/`.

**Taste is real; diet and allergies are synthetic** (drawn at approximate US adult rates).

## The diners

| | invented (95) | Food.com (12,035) |
|---|---|---|
| spread (mean per-taste std) | 0.265 | 0.039 |
| pairwise cosine, median | 0.80 | 0.97 |

Real diners taste far more alike than the invented ones, so satisfaction sits near the top of the
scale (mean about 0.95). Raw MinSat gains are therefore small, and `pct_of_achievable` is the
comparable measure.

## C1: the fairness dial (`c1_summary.csv`, whole-menu formula, λ=5)

`pct_of_achievable` = the MinSat gain over Average, as a share of the largest gain any option in F
offers (Least-Misery's MinSat minus Average's). Ratio of sums, 95% bootstrap CI over groups.

| dataset | MinSat gain | % of achievable | mean-sat cost | Least-Misery's cost |
|---|---|---|---|---|
| Houston | +0.027 [+0.021, +0.034] | **67% [61, 72]** | −0.0034 | −0.0094 |
| Houston, shuffled menus | +0.004 [+0.003, +0.006] | 37% [28, 44] | −0.0003 | −0.0030 |
| Seattle | +0.022 [+0.017, +0.028] | **62% [56, 67]** | −0.0028 | −0.0094 |
| Seattle, shuffled menus | +0.005 [+0.003, +0.006] | 43% [34, 51] | −0.0002 | −0.0024 |

The fair objective recovers about two-thirds of the possible improvement for the worst-off diner,
at about a third of Least-Misery's cost to the group. On shuffled menus it recovers much less, so
most of the gain comes from real differences between restaurants.

## C4: dish compensation (`../ubereats-*-foodcom/table5_compensation.csv`)

| level | Houston MinSat | Seattle MinSat |
|---|---|---|
| venue whole-menu average | 0.890 | 0.893 |
| B1 random safe dish, per diner | 0.528 | 0.675 |
| B3 median diner's best dish, shared | 0.773 | 0.772 |
| B2 best shared dish, one for all | 0.840 | 0.853 |
| **C4 each diner's own best safe dish** | **0.955** | **0.957** |

C4 beats the best shared dish by +0.114 (Houston) and +0.104 (Seattle). On the five Potluck
restaurants the same gap was +0.0065, so on real menus personalising the dish matters.

## Not yet meaningful

Tables 2 and 4 (mediator, veto): every group clears the fixed 0.45 consensus bar in round 1, so all
arms converge in one round. This needs a group-relative bar. The feasible set is still all 300
restaurants, because `dish_is_safe` checks dish names only.
