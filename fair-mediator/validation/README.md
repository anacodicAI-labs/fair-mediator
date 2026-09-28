# validation/

Hand-labelled sheets that measure the two components of the food catalog (task doc §7, items 2 and 3).
Every sheet is seeded (seed 7) and regenerable; the labels in it are the only hand-made part.

**Who labelled:** all three sheets were labelled by Claude (Opus 5.5), working from the dish name, the
Uber Eats description and the recipe's ingredient list. Nobody has looked at the dishes themselves. Before the numbers go in the
paper, a person should re-label a random 20 rows of each sheet and report the agreement.

Rescore any sheet after editing its labels:

```bash
python scripts/handcheck.py score-bitter
python scripts/handcheck.py score-match "Houston, TX"
python scripts/handcheck.py score-match "Seattle, WA"
```

## bitter_spotcheck.csv: the hand-made bitter list

100 Food.com recipes: 50 that the list scores bitter (> 0) and 50 that it scores 0, shuffled.
`looks_bitter` = y if the dish would taste noticeably bitter. Coffee drinks, dark or bittersweet
chocolate or unsweetened cocoa as the main flavour, and bitter liqueurs count as y. A minor bitter
ingredient in a sweet or savoury dish (lemon zest, some beer in a batter, broccoli in a soup,
semisweet chips) counts as n.

| cut-off on bitter score | recipes at or above | precision | recall |
|---|---|---|---|
| > 0 | 50 | 9/50 = 18% (10–31%) | 9/9 = 100% |
| ≥ 0.10 | 30 | 9/30 = 30% (17–48%) | 9/9 = 100% |
| ≥ 0.20 | 14 | 8/14 = 57% (33–79%) | 8/9 = 89% |
| ≥ 0.30 | 11 | 8/11 = 73% (43–90%) | 8/9 = 89% |

Missed: 0 of the 50 zero-scored recipes looked bitter (0–7%).

Read: the list does not miss bitter dishes, and the size of the score tracks taste. A low score
means "contains a bitter ingredient", not "tastes bitter". The score is a mean over ingredients,
so this is how it should behave. Only 9 of 100 dishes are bitter, so the recall CIs are wide.

## match_check_<city>.csv: Uber Eats dish → Food.com recipe

100 dishes per city, drawn from all dishes (matched or not). `correct` takes one of three values:
y means the same dish. p means close: the same family and a similar taste, for example "Beef and
Gyoza" → "gyoza". n means wrong.

| city | exact (y) | exact or close (y+p) | y+p if the match threshold were 0.6 (coverage would fall to ~78%) |
|---|---|---|---|
| Houston, TX | 41% (32–51%) | **76% (67–83%)** | 86% (76–92%) |
| Seattle, WA | 36% (27–46%) | **72% (63–80%)** | 84% (74–91%) |

The threshold stays at 0.30. Raising it buys accuracy but drops taste coverage below the
contract's 90%.

The errors are mostly one of these kinds:
- **Brand or house names:** "Crazy Roll" → "crazy dip", "Alaska Roll" → "alaska burgers", "MVP Burger" → "mvp chili".
- **Non-Latin or transliterated names:** Vietnamese and Chinese dish names.
- **Non-food items that got past the filter:** Seattle had enamel pins, an ice-cream scoop, a
  growler and bagged chips. Houston had "Silverware", which is now filtered out.

They are counted as wrong, so they already pull these numbers down.
