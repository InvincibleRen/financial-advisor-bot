"""Ablation: does adding OBV + ADX change the core selector's predictions?

Runs the exact walk-forward selection pipeline on the cached universe, once with
the new obv/adx features and once with them stripped, and prints the deltas.
Offline (uses data/prices + data/fundamentals cache).
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from src.selection import universe as U
from src.selection.features import build_feature_matrix, TECHNICAL_FEATURES
from src.selection.labels import make_labels
from src.selection.select import run_selection_backtest
from src.cli.select_cli import month_end_rebalances

NEW = ["adx", "obv"]

prices = U.load_prices(U.DEFAULT_UNIVERSE, start="2015-01-01")
fundamentals = U.load_fundamentals(U.DEFAULT_UNIVERSE)
rebalance_dates = month_end_rebalances(prices)
print(f"Universe: {len(prices)} tickers | rebalances: {len(rebalance_dates)} "
      f"({rebalance_dates[0].date()} -> {rebalance_dates[-1].date()})")

fm_full = build_feature_matrix(prices, fundamentals, rebalance_dates)
labels = make_labels(prices, rebalance_dates, horizon_months=1)
fm_base = fm_full.drop(columns=NEW)

print(f"Features WITH new:    {list(fm_full.columns)}")
print(f"Features WITHOUT new: {list(fm_base.columns)}")

def summarise(tag, res):
    s = res.strategy_metrics
    r = res.ranking_metrics
    return {
        "run": tag,
        "CAGR": s["annualised_return"],
        "Sharpe": s["sharpe_ratio"],
        "MaxDD": s["max_drawdown"],
        "HitRate": s["hit_rate"],
        "precision@N": r.get("precision_at_n"),
        "mean_AUC": r.get("mean_auc"),
        "base_rate": r.get("base_rate"),
        "monthly_win": res.monthly_win_rate,
    }

for model in ("gbm", "logistic"):
    rows = []
    for tag, fm in (("with OBV+ADX", fm_full), ("without", fm_base)):
        res = run_selection_backtest(fm, labels, prices, rebalance_dates,
                                     n=3, model_kind=model)
        rows.append(summarise(tag, res))
    df = pd.DataFrame(rows).set_index("run")
    delta = df.loc["with OBV+ADX"] - df.loc["without"]
    df.loc["delta (with - without)"] = delta
    print(f"\n================= MODEL: {model} (top-3, 10bps cost) =================")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(df.T)
