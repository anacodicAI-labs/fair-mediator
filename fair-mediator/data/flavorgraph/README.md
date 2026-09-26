# FlavorGraph — fetched, not yet integrated

Source: https://github.com/lamypark/FlavorGraph
  nodes.csv <- input/nodes_191120.csv (6,653 ingredients + 1,645 compounds)
  edges.csv <- input/edges_191120.csv (147,180 edges: 111,355 ingr-ingr,
               35,440 ingr-fcomp, 384 ingr-dcomp)

Fetched 2026-09-18. Checked, not yet used:
  - 448 of our 1,000 ingredients (../ingredient-flavor.csv) match by exact
    name; the other 6,205 FlavorGraph ingredients are new coverage.
  - These files carry NO taste-dimension columns. This is a graph, not a
    lookup table — a bitter score has to be derived via BitterDB + the
    ingr-fcomp edges, not read off a column. See potluck.txt STORY DATA-3
    for the exact task and its DoD.

Do not treat this directory as a finished input. It is the raw material for
DATA-3, left for Vihaan/Vasu to complete.
