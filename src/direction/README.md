# `direction/` — Extension B (Direction 2, ~15%)

A per-stock **time-series direction forecast**, run *only on the top-N stocks
the core already selected*, as a short-term up/down confirmation/timing signal.
This keeps Direction 2 a drill-down on the core's output rather than a competing
whole-universe project.

| Module | Responsibility | Status |
|---|---|---|
| `forecaster.py` | `DirectionForecaster` (`gbm` / `logistic` / `baseline`) + leakage-safe feature/label builders. | done |
| `evaluate.py` | Walk-forward directional evaluation (accuracy vs base rate, AUC) with a horizon embargo. | done |

**Modelling choice — framework-light, on purpose.** Single-stock daily direction
is close to a coin flip, and the target machine is an x86 macOS / Intel Mac where
the modern deep-learning stack has no working wheel (the same constraint that
shaped Extension A). So the forecaster uses the **same scikit-learn stack as the
core ranker** rather than an LSTM/GRU — portable, readable, fast, and honest about
the weak signal. The `kind` switch stays pluggable (`gbm` / `logistic` / a trivial
momentum `baseline`).

**Leakage safety.** Features at day *t* use only closes up to *t* (causal). The
label is the sign of the forward return over `horizon_days`, so the last *h* rows
are unlabeled; the evaluator additionally embargoes *h* days between the training
window and the prediction date (a training row's label is only known *h* days
after its feature date).

**Usage.**

```bash
# Confirm the core selector's picks (feed its top-N)
python -m src.cli.direction_cli AAPL MSFT NVDA
python -m src.cli.direction_cli AAPL --model logistic --horizon 10 --period 8y
```

```python
from src.direction.forecaster import DirectionForecaster
from src.direction.evaluate import evaluate_direction

ev = evaluate_direction(prices, "AAPL", kind="gbm", horizon_days=5)   # walk-forward
print(ev.accuracy, ev.base_rate, ev.edge, ev.auc)
call = DirectionForecaster(kind="gbm", horizon_days=5).fit(prices, "AAPL").predict_direction(prices)
print(call.prob_up, call.direction)                                   # current signal
```

**Honest framing.** The read is relative: does out-of-sample accuracy beat the
naive base rate (always predict the majority direction), and is AUC above 0.5? A
small or negative edge is an acceptable, informative result — this is a
*confirmation* layer, not a standalone strategy.

**Removable by design:** `selection/` never imports this package (enforced by a
test); deleting `direction/` leaves the core selector fully working.
