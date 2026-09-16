"""Driver for the four methodology additions.

Reproduces the headline walk-forward run, then adds:
  1. average turnover (already computed, never reported)
  2. random-N selection null (isolates ranking from concentration)
  3. block bootstrap vs the current i.i.d. bootstrap
"""
import json
import sys

import numpy as np

import src.selection.universe as U
from src.cli.select_cli import month_end_rebalances
from src.selection.features import build_feature_matrix, cross_sectional_normalize
from src.selection.labels import make_labels
from src.selection.robustness import bootstrap_return_metrics, default_block_size
from src.selection.select import (
    BENCHMARK_TICKER,
    DEFAULT_COST_PER_TURNOVER,
    locate_in_null,
    random_selection_null,
    run_selection_backtest,
)

TOP_N = 3
START = "2015-01-01"
MIN_TRAIN = 3
N_DRAWS = 500
out = {}


def main() -> None:
    universe = U.DEFAULT_UNIVERSE
    print(f"[1/5] loading {len(universe)} tickers from cache...", flush=True)
    prices = U.load_prices(universe, start=START, end=None)
    fundamentals = U.load_fundamentals(universe)
    spy = U.load_prices([BENCHMARK_TICKER], start=START, end=None)

    rebalance_dates = month_end_rebalances(prices)
    print(f"      {len(rebalance_dates)} monthly rebalance dates", flush=True)

    print("[2/5] building feature matrix + labels...", flush=True)
    fm = build_feature_matrix(prices, fundamentals, rebalance_dates)
    fm = cross_sectional_normalize(fm, method="rank")
    labels = make_labels(prices, rebalance_dates, horizon_months=1)

    prices_eval = dict(prices)
    prices_eval.update(spy)

    print("[3/5] walk-forward selection (gbm, top-3)...", flush=True)
    res = run_selection_backtest(
        fm, labels, prices_eval, rebalance_dates,
        n=TOP_N, model_kind="gbm", min_train_dates=MIN_TRAIN,
        cost_per_turnover=DEFAULT_COST_PER_TURNOVER,
    )
    r = res.strategy_returns.to_numpy(dtype="float64")
    r = r[np.isfinite(r)]
    real_sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(12))
    real_ann = float((1.0 + r).prod() ** (12 / r.size) - 1.0)

    out["universe_size"] = len(universe)
    out["n_rebalances"] = int(res.n_rebalances) if hasattr(res, "n_rebalances") else len(r)
    out["n_return_months"] = int(r.size)
    out["top_n"] = TOP_N
    out["avg_turnover"] = float(res.avg_turnover)
    out["cost_per_turnover_bps"] = DEFAULT_COST_PER_TURNOVER * 1e4
    out["real"] = {
        "ann_return": real_ann,
        "sharpe": real_sharpe,
        "strategy_metrics": {k: float(v) for k, v in res.strategy_metrics.items()},
        "benchmark_metrics": {k: float(v) for k, v in res.benchmark_metrics.items()},
        "spy_metrics": {k: float(v) for k, v in (res.spy_metrics or {}).items()},
        "ranking_metrics": {k: float(v) for k, v in res.ranking_metrics.items()},
        "monthly_win_rate": float(res.monthly_win_rate),
    }

    print(f"[4/5] random-{TOP_N} null, {N_DRAWS} draws...", flush=True)
    null = random_selection_null(
        fm, labels, rebalance_dates, prices_eval,
        n=TOP_N, min_train_dates=MIN_TRAIN, n_draws=N_DRAWS,
        cost_per_turnover=DEFAULT_COST_PER_TURNOVER, seed=0,
    )
    out["random_null"] = {
        "n_draws": null["n_draws"],
        "ann_return_quantiles": null["ann_return_quantiles"],
        "sharpe_quantiles": null["sharpe_quantiles"],
        "ann_return_position": locate_in_null(real_ann, null["null_ann_return"]),
        "sharpe_position": locate_in_null(real_sharpe, null["null_sharpe"]),
    }

    print("[5/5] bootstrap: iid vs block...", flush=True)
    bs = default_block_size(r.size)
    out["bootstrap"] = {
        "n_months": int(r.size),
        "block_size": int(bs),
        "iid": {k: list(map(float, v)) for k, v in
                bootstrap_return_metrics(r, block_size=1).items()},
        "block": {k: list(map(float, v)) for k, v in
                  bootstrap_return_metrics(r, block_size=bs).items()},
    }

    with open("methodology_results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\n=== WROTE methodology_results.json ===")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())
