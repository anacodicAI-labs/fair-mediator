# Contributors — attribution by module

Attribution travels with the code (see `prds/PRD_2` non-negotiables). Each
module below names who wrote the original logic and the source it was ported
from; docstrings in `nbio.py` repeat the same citation next to the code itself.

| Module | Contribution | Original source |
|---|---|---|
| `01-taste/02` (member vectors) | 95 simulated taste vectors | Vihaan, `research-tasks` branch, `usertasteembeddings.py`, commit `f045279` (2026-07-17) |
| `01-taste/02` (persona fixture, neutral fallback) | F0 taste vectors wired into the app, deduped fixture, `resolve_taste_vector` | Vincent (`wangvincentx-cmd`), `potluck` main, `group_session_service.py` + `taste_service.py`, commits `3c32c76`, `86124d4` |
| `03-aggregate/02` (the four selectors) | `average_score`, `least_misery_score`, `most_pleasure_score`, `fairness_score`, `aggregate`, `rank_candidates` | Vincent (`wangvincentx-cmd`), `potluck` main, `backend/services/fairness_service.py`, commit `3c32c76` (2026-08-02) |
| — (referenced, not yet ported) | fairness-aware reranking wired into the live chat graph | Vasu (`vasudev parmar`), `potluck` main, `backend/ai/graphs/team_suggestion_graph.py`, commit `65b3f54` (2026-08-13) |
| architecture figure (lives with the paper in `docs/.../paper/figures/`, not in this repo) | architecture figure | Vasu (`vasudev parmar`), `potluck` main, `docs/images/group-fairness-flow.svg`, commit `794db4d` |
| `02-safety/01`, `04-fairness/01`, `04-fairness/02` | feasible set F, fairness metrics, dish compensation | Vihaan, `potluck` `feature/fairness-core-and-notebooks`, `aggregation.py` / `fairness_metrics.py`, commit `4ae7c1f` |
| `05-mediate/01` | `Mediator` — the objective, the reweight rule, the veto mechanic | Vihaan, `research-tasks` branch, `mediator_loop_veto_token.py`, commit `f045279` (2026-07-17) — recoverable at that SHA even if the branch is removed |

A note on `usertasteembeddings.py`: the original file also pushed the 95 vectors
to a Pinecone index using a hardcoded API key. Only the taste-vector data was
harvested; the key is not reproduced anywhere in this repository. See
`prds/PRD_0` — that key must be rotated regardless of this repository's state.
