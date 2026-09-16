# `selection/` — CORE (Direction 1, ~70%)

The main function of the project: **cross-sectional machine-learning stock selection**.
At each monthly rebalance date it scores every stock in the universe from a
leakage-safe feature matrix, ranks them, and selects the top-N. Evaluated by the
walk-forward engine in `common/walkforward.py` against a market benchmark.

Pipeline (build order = Phases 0–2):

| Module | Responsibility | Phase |
|---|---|---|
| `universe.py` | Fixed ticker universe; multi-ticker price + fundamentals loading with local cache. | 0 |
| `features.py` | Leakage-safe feature matrix (fundamentals + technical indicators) indexed by (date, ticker). | 1 |
| `labels.py`   | Forward-return labels vs universe median (binary target). | 1 |
| `model.py`    | Ranker (gradient boosting) + logistic-regression baseline. | 2 |
| `select.py`   | Top-N cross-sectional selection; adapts to the walk-forward engine. | 2 |

**Invariant:** every feature at date *t* uses only information available at *t*
(fundamentals lagged to their actual report date). This is the single most
important correctness property in the project.
